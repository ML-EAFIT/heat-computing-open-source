// -----------------------------------------------------------------------------
// File   : sparse_array.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-29
// -----------------------------------------------------------------------------
// Description:
//   N p-bits on a sparse graph of degree <= D, sampled by chromatic Gibbs with
//   P p-bit lanes running in lock step (block-parallel update).
//     * a sweep is R rounds; round r updates up to P nodes of ONE colour
//       block, lane e takes the node at block position r*P + e
//     * node RAM  (per lane): round r -> {valid, node, h}; valid = 0 idles
//       the lane in a short last round of a block
//     * nbr RAM   (per lane): {round, slot} -> {j, J_node,j}, D slots,
//       unused slots {0, 0} add nothing
//     * spins     : N-bit register, bit i = node i, 1 = +1
//   Per round: all lanes read node words, stream D neighbour words through
//   weight_mac (D + 2 cycles), run pbit, then commit the P new bits one per
//   cycle. Same-round nodes share a colour block, hence are never coupled, so
//   the result equals the sequential sweep (bit-exact, model/fixed.py
//   SparseParallelSampler; layout from model/fixed.engine_layout).
//   Cycles per round: D + P + 15 (measured by tb_sparse_array).
//
//   Idea source: sparse local connectivity and block-parallel Gibbs updates
//   of a two-colourable grid [data/An efficient ... 2510.23972v2.pdf, p.5
//   Sec. III, Eq. 11]; G8/G12 patterns [same pdf, p.24 Table II].
//   [FPGA-APPROX] P shared lanes instead of one cell per node; padding of
//   colour blocks to whole rounds.
//
//   NOISE_CTL = 1 adds two run-time noise options (model/fixed.py
//   SparseParallelSampler q16 / share):
//     nz_tau   : finite RNG correlation time. In S_PB every valid lane draws
//                d; d >= nz_q means the RNG has not decorrelated, so the lane
//                keeps its spin (no write, its p-bit draw is not consumed).
//                [DOCUMENTED motivation: tau0, DTM p.6 Fig. 4b]
//     nz_share : all lanes read lane 0's draw, and only lane 0 advances, so the
//                cells of one round see identical (fully correlated) noise.
//   With NOISE_CTL = 0 (default) the ports are ignored.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-29  jero98772  v1.0  Initial version
//   2026-10-30  jero98772  v1.1  NOISE_CTL: RNG persistence and shared noise
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module sparse_array #
(
    // number of p-bits
    parameter        N           = 16,
    // node index width, clog2(N) (>= 1)
    parameter        IW          = 4,
    // lanes (parallel p-bit engines), 1..4 (one noise lane each)
    parameter        P           = 4,
    // maximum degree (neighbour slots used per node)
    parameter        D           = 4,
    // slot address width, 2^DW >= D (>= 1)
    parameter        DW          = 2,
    // round address width, 2^LW >= rounds per sweep (>= 1)
    parameter        LW          = 2,
    // power-on rounds per sweep
    parameter        ROUNDS      = 4,
    // weight / bias width
    parameter        WB          = 8,
    // weight / bias fraction bits (4 -> Q3.4, 6 -> Q1.6)
    parameter        WFRAC       = 4,
    // accumulator width, >= WB + clog2(D+1) + 1
    parameter        AW          = 12,
    // 1 = piecewise-linear sigmoid instead of the LUT
    parameter        USE_PWL     = 0,
    // LFSR reset seed
    parameter [31:0] SEED        = 32'hACE1F00D,
    // RAM images: <prefix><lane>.hex ("" = zeros)
    parameter        NODE_PREFIX = "",
    parameter        NBR_PREFIX  = "",
    parameter        LUT_FILE    = "../model/vectors/sigmoid_lut.hex",
    // 1 = build the nz_tau / nz_share noise options
    parameter        NOISE_CTL   = 0
)
(
    input  wire                 clk,
    input  wire                 rst,
    // host programming ports (lane select l, write only while idle)
    input  wire                 m_we,       // node word
    input  wire [1:0]           m_lane,
    input  wire [LW-1:0]        m_addr,
    input  wire [IW+WB:0]       m_data,     // {valid, node, h}
    input  wire                 n_we,       // neighbour word
    input  wire [1:0]           n_lane,
    input  wire [LW+DW-1:0]     n_addr,     // {round, slot}
    input  wire [IW+WB-1:0]     n_data,     // {j, J}
    input  wire                 r_we,       // rounds per sweep
    input  wire [LW:0]          r_data,
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
    // noise options (NOISE_CTL = 1, change only while idle)
    input  wire                 nz_tau,
    input  wire                 nz_share,
    input  wire [15:0]          nz_q,
    output wire                 busy,
    output wire                 sweep_done,
    output wire [N-1:0]         spins
);

initial begin
    if ((1 << IW) < N || (1 << DW) < D || (1 << LW) < ROUNDS || P < 1 || P > 4 ||
        (D + 1) * (1 << (WB - 1)) > (1 << (AW - 1))) begin
        $error("sparse_array: bad P, or IW/DW/LW/AW too small");
        $finish;
    end
end

// state encoding: binary, 3 bits (8 states)
localparam [2:0] S_IDLE = 3'd0,
                 S_NODE = 3'd1,
                 S_MAC  = 3'd2,
                 S_MACW = 3'd3,
                 S_PB   = 3'd4,
                 S_PBW  = 3'd5,
                 S_CMT  = 3'd6,
                 S_HOLD = 3'd7;

localparam integer  DI    = D;
localparam [DW:0]   NSLOT = DI[DW:0];
localparam integer  LASTL = P - 1;
localparam [1:0]    LASTP = LASTL[1:0];
localparam integer  RI    = ROUNDS;

reg [2:0]    state_reg  = S_IDLE,         state_next;
reg [LW-1:0] ptr_reg    = {LW{1'b0}},     ptr_next;      // round
reg [DW:0]   slot_reg   = {(DW+1){1'b0}}, slot_next;     // next slot to issue
reg          vld_reg    = 1'b0,           vld_next;      // nbr word valid
reg [1:0]    cidx_reg   = 2'd0,           cidx_next;     // lane being committed
reg [31:0]   left_reg   = 32'd0,          left_next;     // sweeps remaining
reg [N-1:0]  spins_reg  = {N{1'b0}},      spins_next;
reg          swd_reg    = 1'b0,           swd_next;
reg [LW:0]   rounds_reg = RI[LW:0];
reg [P-1:0]  keep_reg   = {P{1'b0}};                     // lane keeps its spin
reg          mac_load;
reg          pb_go;
wire [P-1:0] keep_next;
wire [P-1:0] keep_eff;                                     // keep_reg, or 0 without NOISE_CTL

wire [P-1:0]    lane_valid;
wire [P*IW-1:0] lane_node;
wire [P-1:0]    lane_bit;
wire [P-1:0]    lane_done;
wire [P-1:0]    lane_take;
wire [P*16-1:0] rnd;
wire [P*16-1:0] lane_rnd;
wire [P-1:0]    nz_en;
wire            tau_on   = (NOISE_CTL != 0) && nz_tau;
wire            share_on = (NOISE_CTL != 0) && nz_share;

wire [LW-1:0] last_round = rounds_reg[LW-1:0] - 1'b1;

always @(posedge clk) begin
    if (r_we) rounds_reg <= r_data;
end

// ------------------------------------------------------------------ lanes
genvar g;
generate
    for (g = 0; g < P; g = g + 1) begin : gen_lane
        localparam [7:0] DIG = 8'h30 + g;

        reg [IW+WB:0]   nmem [(2**LW)-1:0];
        reg [IW+WB-1:0] wmem [(2**(LW+DW))-1:0];
        reg [IW+WB:0]   nq = {(IW+WB+1){1'b0}};
        reg [IW+WB-1:0] wq = {(IW+WB){1'b0}};

        integer m;
        initial begin
            for (m = 0; m < 2**LW; m = m + 1) begin
                nmem[m] = {(IW+WB+1){1'b0}};
            end
            for (m = 0; m < 2**(LW+DW); m = m + 1) begin
                wmem[m] = {(IW+WB){1'b0}};
            end
            if (NODE_PREFIX != "") $readmemh({NODE_PREFIX, DIG, ".hex"}, nmem, 0, (2**LW)-1);
            if (NBR_PREFIX != "")  $readmemh({NBR_PREFIX, DIG, ".hex"}, wmem, 0, (2**(LW+DW))-1);
        end

        always @(posedge clk) begin
            if (m_we && m_lane == g) nmem[m_addr] <= m_data;
            nq <= nmem[ptr_reg];
        end

        always @(posedge clk) begin
            if (n_we && n_lane == g) wmem[n_addr] <= n_data;
            wq <= wmem[{ptr_reg, slot_reg[DW-1:0]}];
        end

        wire signed [AW-1:0] field;
        wire [IW-1:0]        j_w = wq[IW+WB-1:WB];
        wire signed [9:0]    logit_dbg;          // debug taps, unused here
        wire [15:0]          p_dbg;

        assign lane_valid[g]            = nq[IW+WB];
        assign lane_node[g*IW +: IW]    = nq[IW+WB-1:WB];

        weight_mac #(
            .WB (WB),
            .AW (AW)
        ) weight_mac_inst (
            .clk    (clk),
            .rst    (rst),
            .load   (mac_load),
            .h      (nq[WB-1:0]),
            .add_en (vld_reg),
            .w      (wq[WB-1:0]),
            .s      (spins_reg[j_w]),
            .acc    (field)
        );

        pbit #(
            .AW       (AW),
            .WFRAC    (WFRAC),
            .USE_PWL  (USE_PWL),
            .LUT_FILE (LUT_FILE)
        ) pbit_inst (
            .clk       (clk),
            .rst       (rst),
            .start     (pb_go && nq[IW+WB]),
            .field     (field),
            .beta      (beta),
            .rnd       (lane_rnd[g*16 +: 16]),
            .bit_out   (lane_bit[g]),
            .done      (lane_done[g]),
            .rnd_take  (lane_take[g]),
            .logit_dbg (logit_dbg),
            .p_dbg     (p_dbg)
        );

        // noise routing: own lane, or lane 0 for every lane when shared
        wire [15:0] d_w = share_on ? rnd[15:0] : rnd[g*16 +: 16];

        assign lane_rnd[g*16 +: 16] = d_w;

        // refreshed every round (S_PB), so it is 0 whenever tau is off
        assign keep_next[g] = pb_go ? (tau_on && lane_valid[g] && (d_w >= nz_q)) : keep_reg[g];

        if (g == 0) begin : gen_en0
            // lane 0 serves every lane when shared; lane 0 is always valid
            assign nz_en[g] = (pb_go && tau_on) || (lane_take[g] && !keep_eff[g]);
        end else begin : gen_enk
            assign nz_en[g] = !share_on &&
                              ((pb_go && tau_on && lane_valid[g]) || (lane_take[g] && !keep_eff[g]));
        end
    end
endgenerate

noise_bank #(
    .LANES     (P),
    .SEED      (SEED),
    .RO_JITTER (0)
) noise_bank_inst (
    .clk       (clk),
    .rst       (rst),
    .en        (nz_en),
    .seed_load (seed_load),
    .seed_in   (seed_in),
    .rnd       (rnd)
);

// ------------------------------------------------------------------ control
// a kept lane commits nothing: fold keep into the lane's write-valid bit, so the
// spin write stays one indexed enable (keeps the 256-bit write decoder small)
generate
    if (NOISE_CTL != 0) begin : gen_keep
        assign keep_eff = keep_reg;
    end else begin : gen_nokeep
        assign keep_eff = {P{1'b0}};
    end
endgenerate
wire [P-1:0]  lane_wr = lane_valid & ~keep_eff;
wire          c_valid = lane_wr[cidx_reg];
wire [IW-1:0] c_node  = lane_node[cidx_reg*IW +: IW];
wire          c_bit   = lane_bit[cidx_reg];

always @* begin
    state_next = state_reg;
    ptr_next   = ptr_reg;
    slot_next  = slot_reg;
    vld_next   = 1'b0;
    cidx_next  = cidx_reg;
    left_next  = left_reg;
    spins_next = spins_reg;
    swd_next   = 1'b0;
    mac_load   = 1'b0;
    pb_go      = 1'b0;

    case (state_reg)
        S_IDLE: begin
            if (spin_we) begin
                spins_next = spin_data;
            end
            if (run && n_sweeps != 32'd0 && rounds_reg != {(LW+1){1'b0}}) begin
                left_next  = n_sweeps;
                ptr_next   = {LW{1'b0}};
                state_next = S_NODE;
            end
        end
        S_NODE: begin                          // nq <= nmem[ptr] this cycle
            slot_next  = {(DW+1){1'b0}};
            state_next = S_MAC;
        end
        S_MAC: begin                           // nq valid: load h, issue slot 0
            mac_load   = 1'b1;
            vld_next   = 1'b1;
            slot_next  = slot_reg + 1'b1;
            state_next = S_MACW;
        end
        S_MACW: begin                          // wq of slot (slot_reg - 1) valid
            if (slot_reg < NSLOT) begin
                vld_next  = 1'b1;
                slot_next = slot_reg + 1'b1;
            end else begin
                state_next = S_PB;             // last word is added this cycle
            end
        end
        S_PB: begin                            // fields final
            pb_go      = 1'b1;
            // with nz_tau the keep draw d is taken here, before the p-bit draw
            state_next = S_PBW;
        end
        S_PBW: begin                           // lane 0 is always valid
            if (lane_done[0]) begin
                state_next = S_CMT;
            end
        end
        S_CMT: begin
            if (c_valid) begin
                spins_next[c_node] = c_bit;
            end
            cidx_next = cidx_reg + 2'd1;
            if (cidx_reg == LASTP) begin
                cidx_next = 2'd0;
                if (ptr_reg == last_round) begin
                    swd_next  = 1'b1;          // registered with the last write
                    ptr_next  = {LW{1'b0}};
                    left_next = left_reg - 32'd1;
                    if (left_reg == 32'd1 || stop) begin
                        state_next = S_IDLE;
                    end else if (hold) begin
                        state_next = S_HOLD;
                    end else begin
                        state_next = S_NODE;
                    end
                end else begin
                    ptr_next   = ptr_reg + 1'b1;
                    state_next = stop ? S_IDLE : S_NODE;
                end
            end
        end
        S_HOLD: begin
            if (stop) begin
                state_next = S_IDLE;
            end else if (!hold) begin
                state_next = S_NODE;
            end
        end
        default: begin
            state_next = S_IDLE;
        end
    endcase
end

always @(posedge clk) begin
    state_reg <= state_next;
    ptr_reg   <= ptr_next;
    slot_reg  <= slot_next;
    vld_reg   <= vld_next;
    cidx_reg  <= cidx_next;
    left_reg  <= left_next;
    spins_reg <= spins_next;
    swd_reg   <= swd_next;
    keep_reg  <= keep_next;

    if (rst) begin
        state_reg <= S_IDLE;
        ptr_reg   <= {LW{1'b0}};
        vld_reg   <= 1'b0;
        left_reg  <= 32'd0;
        spins_reg <= {N{1'b0}};
        swd_reg   <= 1'b0;
        keep_reg  <= {P{1'b0}};
    end
end

assign busy       = (state_reg != S_IDLE);
assign sweep_done = swd_reg;
assign spins      = spins_reg;

endmodule

`resetall
