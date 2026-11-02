// -----------------------------------------------------------------------------
// File   : pbit_array.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   N p-bits with a dense N x N coupling matrix, sampled by systematic-scan
//   (chromatic) Gibbs with one time-multiplexed p-bit engine.
//     * weight RAM : J[i][j], signed, WFRAC fraction bits, address {i, j}
//     * bias RAM   : h[i],    same format
//     * order RAM  : visit order, colour block 0 nodes, then block 1, ...
//     * spins      : N-bit register, bit i = node i, 1 = +1
//   run + n_sweeps starts n_sweeps full sweeps. sweep_done pulses once per
//   sweep, in the cycle where `spins` already holds the post-sweep state.
//   hold (sampled at sweep boundaries) pauses before the next sweep. stop
//   aborts after the current update. Per update: N + 18 cycles (measured).
//
//   Idea source: grid of sampling cells updated by chromatic Gibbs, two
//   colour blocks per iteration [data/An efficient ... 2510.23972v2.pdf,
//   p.5 Sec. III; p.21 Fig. 8]; colour blocks in software
//   [thrml/02_spin_models.py:L129-160], parity colouring
//   [torx/06_ising_sampling_contrastive_divergence.py:L198-208].
//   [INFERRED] sequential updates inside a colour class equal the parallel
//   block update (same-colour nodes are conditionally independent).
//   [FPGA-APPROX] one shared engine instead of one cell per node.
//   Mirror: model/fixed.FixedSampler.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
//   2026-10-30  jero98772  v1.1  upd_* tap for temper_ctrl.v (no behaviour change)
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module pbit_array #
(
    // number of p-bits
    parameter        N        = 16,
    // node index width, clog2(N) (>= 1)
    parameter        IW       = 4,
    // weight / bias width
    parameter        WB       = 8,
    // weight / bias fraction bits (4 -> Q3.4, 6 -> Q1.6)
    parameter        WFRAC    = 4,
    // accumulator width, >= WB + clog2(N+1) + 1
    parameter        AW       = 14,
    // 1 = piecewise-linear sigmoid instead of the LUT
    parameter        USE_PWL  = 0,
    // LFSR reset seed
    parameter [31:0] SEED     = 32'hACE1F00D,
    // initialisation files ("" = zeros / identity order)
    parameter        W_FILE   = "",
    parameter        H_FILE   = "",
    parameter        O_FILE   = "",
    parameter        LUT_FILE = "../model/vectors/sigmoid_lut.hex"
)
(
    input  wire                 clk,
    input  wire                 rst,
    // host programming ports
    input  wire                 w_we,
    input  wire [2*IW-1:0]      w_addr,
    input  wire [WB-1:0]        w_data,
    input  wire                 h_we,
    input  wire [IW-1:0]        h_addr,
    input  wire [WB-1:0]        h_data,
    input  wire                 o_we,
    input  wire [IW-1:0]        o_addr,
    input  wire [IW-1:0]        o_data,
    input  wire                 spin_we,
    input  wire [N-1:0]         spin_data,
    input  wire                 seed_load,
    input  wire [31:0]          seed_in,
    // run control
    input  wire                 run,
    input  wire [31:0]          n_sweeps,
    input  wire                 hold,
    input  wire                 stop,
    input  wire [7:0]           beta,
    output wire                 busy,
    output wire                 sweep_done,
    output wire [N-1:0]         spins,
    // update tap (one cycle per p-bit update, at the spin write): node,
    // old and new bit, local field; used by temper_ctrl.v for energies
    output wire                 upd_we,
    output wire                 upd_old,
    output wire                 upd_new,
    output wire signed [AW-1:0] upd_field
);

initial begin
    // |h + sum +-J| <= (N+1) * 2^(WB-1) must fit in AW signed bits
    if ((1 << IW) < N || (N + 1) * (1 << (WB - 1)) > (1 << (AW - 1))) begin
        $error("pbit_array: IW or AW too small for N");
        $finish;
    end
end

// ------------------------------------------------------------------ memories
(* ram_style = "block" *)
reg [WB-1:0] wmem [(2**(2*IW))-1:0];
reg [WB-1:0] hmem [(2**IW)-1:0];
reg [IW-1:0] omem [(2**IW)-1:0];

integer m;
initial begin
    for (m = 0; m < 2**(2*IW); m = m + 1) begin
        wmem[m] = {WB{1'b0}};
    end
    for (m = 0; m < 2**IW; m = m + 1) begin
        hmem[m] = {WB{1'b0}};
        omem[m] = m[IW-1:0];
    end
    if (W_FILE != "") $readmemh(W_FILE, wmem, 0, (2**(2*IW))-1);
    if (H_FILE != "") $readmemh(H_FILE, hmem, 0, (2**IW)-1);
    if (O_FILE != "") $readmemh(O_FILE, omem, 0, (2**IW)-1);
end

wire [2*IW-1:0] mac_addr;
reg  [WB-1:0]   w_q = {WB{1'b0}};
reg  [WB-1:0]   h_q = {WB{1'b0}};
reg  [IW-1:0]   o_q = {IW{1'b0}};

// state encoding: binary, 3 bits (8 states)
localparam [2:0] S_IDLE = 3'd0,
                 S_ORD  = 3'd1,
                 S_BIAS = 3'd2,
                 S_MAC  = 3'd3,
                 S_MACW = 3'd4,
                 S_PBW  = 3'd5,
                 S_WR   = 3'd6,
                 S_HOLD = 3'd7;

localparam integer   LASTI = N - 1;
localparam [IW-1:0] LAST  = LASTI[IW-1:0];       // last position in the order

reg [2:0]    state_reg = S_IDLE,      state_next;
reg [IW-1:0] k_reg     = {IW{1'b0}},  k_next;        // position in order
reg [IW-1:0] node_reg  = {IW{1'b0}},  node_next;
reg [31:0]   left_reg  = 32'd0,       left_next;     // sweeps remaining
reg [N-1:0]  spins_reg = {N{1'b0}},   spins_next;
reg          swd_reg   = 1'b0,        swd_next;
reg          mac_start;
reg          pb_start;

wire signed [AW-1:0] field;
wire                 mac_done;
wire                 pb_bit;
wire                 pb_done;
wire                 pb_take;
wire [15:0]          rnd;
wire signed [9:0]    pb_logit;       // debug taps, unused here
wire [15:0]          pb_p;

always @(posedge clk) begin
    if (w_we) wmem[w_addr] <= w_data;
    w_q <= wmem[mac_addr];
end

always @(posedge clk) begin
    if (h_we) hmem[h_addr] <= h_data;
    h_q <= hmem[o_q];
end

always @(posedge clk) begin
    if (o_we) omem[o_addr] <= o_data;
    o_q <= omem[k_reg];
end

// ------------------------------------------------------------------ control
always @* begin
    state_next = state_reg;
    k_next     = k_reg;
    node_next  = node_reg;
    left_next  = left_reg;
    spins_next = spins_reg;
    swd_next   = 1'b0;
    mac_start  = 1'b0;
    pb_start   = 1'b0;

    case (state_reg)
        S_IDLE: begin
            if (spin_we) begin
                spins_next = spin_data;
            end
            if (run && n_sweeps != 32'd0) begin
                left_next  = n_sweeps;
                k_next     = {IW{1'b0}};
                state_next = S_ORD;
            end
        end
        S_ORD: begin                       // o_q <= omem[k] this cycle
            state_next = S_BIAS;
        end
        S_BIAS: begin                      // o_q valid, h_q <= hmem[o_q]
            node_next  = o_q;
            state_next = S_MAC;
        end
        S_MAC: begin                       // h_q valid
            mac_start  = 1'b1;
            state_next = S_MACW;
        end
        S_MACW: begin
            if (mac_done) begin
                pb_start   = 1'b1;
                state_next = S_PBW;
            end
        end
        S_PBW: begin
            if (pb_done) begin
                spins_next[node_reg] = pb_bit;
                state_next = S_WR;
            end
        end
        S_WR: begin
            if (k_reg == LAST) begin
                k_next    = {IW{1'b0}};
                left_next = left_reg - 32'd1;
                if (left_reg == 32'd1 || stop) begin
                    state_next = S_IDLE;
                end else if (hold) begin
                    state_next = S_HOLD;
                end else begin
                    state_next = S_ORD;
                end
            end else begin
                k_next     = k_reg + 1'b1;
                state_next = stop ? S_IDLE : S_ORD;
            end
        end
        S_HOLD: begin
            if (stop) begin
                state_next = S_IDLE;
            end else if (!hold) begin
                state_next = S_ORD;
            end
        end
        default: begin
            state_next = S_IDLE;
        end
    endcase

    // sweep_done is registered together with the last spin write
    if (state_reg == S_PBW && pb_done && k_reg == LAST) begin
        swd_next = 1'b1;
    end
end

always @(posedge clk) begin
    state_reg <= state_next;
    k_reg     <= k_next;
    node_reg  <= node_next;
    left_reg  <= left_next;
    spins_reg <= spins_next;
    swd_reg   <= swd_next;

    if (rst) begin
        state_reg <= S_IDLE;
        k_reg     <= {IW{1'b0}};
        left_reg  <= 32'd0;
        spins_reg <= {N{1'b0}};
        swd_reg   <= 1'b0;
    end
end

// ------------------------------------------------------------------ datapath
coupling_row #(
    .N  (N),
    .IW (IW),
    .WB (WB),
    .AW (AW)
) coupling_row_inst (
    .clk    (clk),
    .rst    (rst),
    .start  (mac_start),
    .row    (node_reg),
    .h_in   (h_q),
    .spins  (spins_reg),
    .w_addr (mac_addr),
    .w_data (w_q),
    .field  (field),
    .done   (mac_done)
);

pbit #(
    .AW       (AW),
    .WFRAC    (WFRAC),
    .USE_PWL  (USE_PWL),
    .LUT_FILE (LUT_FILE)
) pbit_inst (
    .clk       (clk),
    .rst       (rst),
    .start     (pb_start),
    .field     (field),
    .beta      (beta),
    .rnd       (rnd),
    .bit_out   (pb_bit),
    .done      (pb_done),
    .rnd_take  (pb_take),
    .logit_dbg (pb_logit),
    .p_dbg     (pb_p)
);

noise_bank #(
    .LANES     (1),
    .SEED      (SEED),
    .RO_JITTER (0)
) noise_bank_inst (
    .clk       (clk),
    .rst       (rst),
    .en        (pb_take),
    .seed_load (seed_load),
    .seed_in   (seed_in),
    .rnd       (rnd)
);

assign busy       = (state_reg != S_IDLE);
assign sweep_done = swd_reg;
assign spins      = spins_reg;
assign upd_we     = (state_reg == S_PBW) && pb_done;
assign upd_old    = spins_reg[node_reg];
assign upd_new    = pb_bit;
assign upd_field  = field;

endmodule

`resetall
