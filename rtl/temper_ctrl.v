// -----------------------------------------------------------------------------
// File   : temper_ctrl.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-30
// -----------------------------------------------------------------------------
// Description:
//   Parallel tempering (replica exchange) controller for one pbit_array.
//   Holds K <= 4 replicas of the spin state, their energies and a beta
//   ladder, and drives the array (spin load, 1-sweep runs, beta) from the
//   side; the array itself is unchanged. One round:
//     for k = 0..K-1: load bank[k], run 1 sweep at ladder[rung_of[k]],
//                     store the spins, e[k] += s_old * field on every flip
//                     (pbit_array upd_* tap; e = E/2 in weight units)
//     swap phase:     even rounds rung pairs (0,1), (2,3), odd rounds (1,2);
//                     accept with sigma((b_a - b_b) * 2 (e_x - e_y)) using a
//                     second pbit (field = energy difference, beta = ladder
//                     gap) and its own LFSR (noise lane 1); an accepted swap
//                     exchanges the rung labels.
//   round_done pulses after the swap phase with spins_out = the replica on
//   rung 0 (the target beta). reset_all: banks <- 0, e <- 0, identity
//   labels (with seed_load: reseed the swap LFSR). resync: every bank <- the
//   rung-0 replica, e <- 0 (needed after h / J change, energies are relative).
//   Bit-exact mirror: model/tempering.py (Tempering, TemperingSampler).
//
//   Idea source: replica exchange is the standard cure for the slow mixing
//   between solutions seen in docs/05 (factor 3). Annealing by beta
//   [thrml/03_codon_optimization.py:L311-344]; Boltzmann law of the array
//   [data/An efficient ... 2510.23972v2.pdf, p.5 Eqs. 10-11].
//   [FPGA-APPROX] Barker acceptance via the sigmoid LUT, time-multiplexed
//   replicas, incremental energies.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-30  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module temper_ctrl #
(
    // p-bits, field width of the array
    parameter        N        = 16,
    parameter        AW       = 14,
    // fraction bits of the weights (for the swap p-bit)
    parameter        WFRAC    = 4,
    // replicas, 2..4
    parameter        K        = 4,
    // energy register width
    parameter        EW       = 24,
    // power-on ladder {b3, b2, b1, b0}, rung 0 = target
    parameter [31:0] LADDER   = 32'h0C_14_20_30,
    parameter [31:0] SEED     = 32'hACE1F00D,
    parameter        LUT_FILE = "../model/vectors/sigmoid_lut.hex"
)
(
    input  wire                 clk,
    input  wire                 rst,
    // configuration (idle only)
    input  wire                 ladder_we,
    input  wire [1:0]           ladder_addr,
    input  wire [7:0]           ladder_data,
    input  wire                 reset_all,
    input  wire                 seed_load,
    input  wire [31:0]          seed_in,
    input  wire                 resync,
    // run control
    input  wire                 run,
    input  wire [31:0]          n_rounds,
    input  wire                 hold,
    output wire                 busy,
    output wire                 round_done,
    output wire [N-1:0]         spins_out,
    // pbit_array side
    output reg                  a_spin_we,
    output wire [N-1:0]         a_spin_data,
    output reg                  a_run,
    output wire [7:0]           a_beta,
    input  wire                 a_busy,
    input  wire                 a_swd,
    input  wire [N-1:0]         a_spins,
    input  wire                 upd_we,
    input  wire                 upd_old,
    input  wire                 upd_new,
    input  wire signed [AW-1:0] upd_field
);

initial begin
    if (K < 2 || K > 4 || EW < AW + 2) begin
        $error("temper_ctrl: K must be 2..4 and EW > AW + 1");
        $finish;
    end
end

// state encoding: binary, 4 bits (10 states)
localparam [3:0] S_IDLE  = 4'd0,
                 S_LOAD  = 4'd1,
                 S_GO    = 4'd2,
                 S_SWEEP = 4'd3,
                 S_STORE = 4'd4,
                 S_SWSET = 4'd5,
                 S_SWGO  = 4'd6,
                 S_SWWT  = 4'd7,
                 S_END   = 4'd8,
                 S_HOLD  = 4'd9;

localparam integer KI   = K;
localparam [2:0]   KLIM = KI[2:0];
localparam [63:0]  PROD = 64'h9E3779B9 * 2;          // lane 1, model/fixed.lane_seed
localparam [31:0]  LS0  = SEED ^ PROD[31:0];
localparam [31:0]  LSD  = (LS0 == 32'd0) ? 32'd1 : LS0;

reg [3:0]           state_reg = S_IDLE, state_next;
reg [1:0]           k_reg     = 2'd0,   k_next;           // replica being swept
reg [2:0]           a_reg     = 3'd0,   a_next;           // lower rung of the pair
reg                 par_reg   = 1'b0,   par_next;         // round parity
reg [31:0]          left_reg  = 32'd0,  left_next;
reg                 rd_reg    = 1'b0,   rd_next;
reg [7:0]           beta_reg  = 8'd0,   beta_next;
reg signed [AW-1:0] sf_reg    = {AW{1'b0}}, sf_next;      // swap field
reg [7:0]           sb_reg    = 8'd0,   sb_next;          // swap beta
reg                 sw_go;

reg [N-1:0]          bank   [0:K-1];
reg signed [EW-1:0]  e      [0:K-1];
reg [1:0]            rep_at [0:K-1];                      // rung -> replica
reg [1:0]            rung   [0:K-1];                      // replica -> rung
reg [7:0]            ladder [0:3];

wire        sw_bit, sw_done, sw_take;
wire [15:0] sw_rnd;
wire signed [9:0] sw_logit;                               // debug taps, unused
wire [15:0] sw_p;

integer m;
initial begin
    for (m = 0; m < K; m = m + 1) begin
        bank[m]   = {N{1'b0}};
        e[m]      = {EW{1'b0}};
        rep_at[m] = m[1:0];
        rung[m]   = m[1:0];
    end
    for (m = 0; m < 4; m = m + 1) begin
        ladder[m] = LADDER[m*8 +: 8];
    end
end

// pair (a, a+1) of the swap phase: replicas, ladder gap, energy gap
wire [1:0]          pa    = a_reg[1:0];
wire [1:0]          pb    = pa + 2'd1;
wire [1:0]          rx    = rep_at[pa];
wire [1:0]          ry    = rep_at[pb];
wire signed [8:0]   dbeta = $signed({1'b0, ladder[pa]}) - $signed({1'b0, ladder[pb]});
wire signed [EW-1:0] de   = e[rx] - e[ry];
wire signed [EW-1:0] de_s = dbeta[8] ? -de : de;
wire [8:0]          db_a  = dbeta[8] ? -dbeta : dbeta;
localparam signed [EW-1:0] FMAX = (1 <<< (AW - 1)) - 1;
localparam signed [EW-1:0] FMIN = -(1 <<< (AW - 1));
wire signed [AW-1:0] de_sat = (de_s > FMAX) ? FMAX[AW-1:0] :
                              (de_s < FMIN) ? FMIN[AW-1:0] : de_s[AW-1:0];

always @* begin
    state_next = state_reg;
    k_next     = k_reg;
    a_next     = a_reg;
    par_next   = par_reg;
    left_next  = left_reg;
    rd_next    = 1'b0;
    beta_next  = beta_reg;
    sf_next    = sf_reg;
    sb_next    = sb_reg;
    a_spin_we  = 1'b0;
    a_run      = 1'b0;
    sw_go      = 1'b0;

    case (state_reg)
        S_IDLE: begin
            if (reset_all) begin
                par_next = 1'b0;
            end
            if (run && n_rounds != 32'd0) begin
                left_next  = n_rounds;
                k_next     = 2'd0;
                state_next = S_LOAD;
            end
        end
        S_LOAD: begin                          // array is idle: load the replica
            a_spin_we  = 1'b1;
            beta_next  = ladder[rung[k_reg]];
            state_next = S_GO;
        end
        S_GO: begin
            a_run      = 1'b1;
            state_next = S_SWEEP;
        end
        S_SWEEP: begin                         // energy updates in the always below
            if (a_swd) begin
                state_next = S_STORE;
            end
        end
        S_STORE: begin
            if (!a_busy) begin
                if ({1'b0, k_reg} == KLIM - 3'd1) begin
                    a_next     = {2'b00, par_reg};
                    state_next = S_SWSET;
                end else begin
                    k_next     = k_reg + 2'd1;
                    state_next = S_LOAD;
                end
            end
        end
        S_SWSET: begin
            if (a_reg + 3'd1 >= KLIM) begin
                state_next = S_END;
            end else begin
                sf_next    = de_sat;
                sb_next    = db_a[7:0];
                state_next = S_SWGO;
            end
        end
        S_SWGO: begin
            sw_go      = 1'b1;
            state_next = S_SWWT;
        end
        S_SWWT: begin                          // label swap in the always below
            if (sw_done) begin
                a_next     = a_reg + 3'd2;
                state_next = S_SWSET;
            end
        end
        S_END: begin
            rd_next   = 1'b1;
            par_next  = ~par_reg;
            left_next = left_reg - 32'd1;
            k_next    = 2'd0;
            if (left_reg == 32'd1) begin
                state_next = S_IDLE;
            end else if (hold) begin
                state_next = S_HOLD;
            end else begin
                state_next = S_LOAD;
            end
        end
        S_HOLD: begin
            if (!hold) begin
                state_next = S_LOAD;
            end
        end
        default: begin
            state_next = S_IDLE;
        end
    endcase
end

always @(posedge clk) begin
    state_reg <= state_next;
    k_reg     <= k_next;
    a_reg     <= a_next;
    par_reg   <= par_next;
    left_reg  <= left_next;
    rd_reg    <= rd_next;
    beta_reg  <= beta_next;
    sf_reg    <= sf_next;
    sb_reg    <= sb_next;

    if (ladder_we && state_reg == S_IDLE) begin
        ladder[ladder_addr] <= ladder_data;
    end

    // replica banks and energies
    if (state_reg == S_IDLE && reset_all) begin
        for (m = 0; m < K; m = m + 1) begin
            bank[m]   <= {N{1'b0}};
            e[m]      <= {EW{1'b0}};
            rep_at[m] <= m[1:0];
            rung[m]   <= m[1:0];
        end
    end else if (state_reg == S_IDLE && resync) begin
        for (m = 0; m < K; m = m + 1) begin
            bank[m] <= bank[rep_at[0]];
            e[m]    <= {EW{1'b0}};
        end
    end
    if (state_reg == S_SWEEP && upd_we && (upd_old != upd_new)) begin
        // e = E/2: flipping s_i changes E/2 by s_old * field_i
        e[k_reg] <= upd_old ? e[k_reg] + {{(EW-AW){upd_field[AW-1]}}, upd_field}
                            : e[k_reg] - {{(EW-AW){upd_field[AW-1]}}, upd_field};
    end
    if (state_reg == S_SWEEP && a_swd) begin
        bank[k_reg] <= a_spins;
    end
    if (state_reg == S_SWWT && sw_done && sw_bit) begin
        rep_at[pa] <= ry;
        rep_at[pb] <= rx;
        rung[rx]   <= pb;
        rung[ry]   <= pa;
    end

    if (rst) begin
        state_reg <= S_IDLE;
        left_reg  <= 32'd0;
        rd_reg    <= 1'b0;
    end
end

// swap decision = one p-bit: P(accept) = sigma(2 * gap * (e_x - e_y))
pbit #(
    .AW       (AW),
    .WFRAC    (WFRAC),
    .USE_PWL  (0),
    .LUT_FILE (LUT_FILE)
) swap_pbit_inst (
    .clk       (clk),
    .rst       (rst),
    .start     (sw_go),
    .field     (sf_reg),
    .beta      (sb_reg),
    .rnd       (sw_rnd),
    .bit_out   (sw_bit),
    .done      (sw_done),
    .rnd_take  (sw_take),
    .logit_dbg (sw_logit),
    .p_dbg     (sw_p)
);

lfsr #(
    .MASK     (32'hA3000000),                  // model/fixed.LFSR_MASKS[1]
    .SEED     (LSD),
    .OUT_BITS (16)
) swap_lfsr_inst (
    .clk       (clk),
    .rst       (rst),
    .en        (sw_take),
    .seed_load (seed_load),
    .seed_in   (seed_in ^ PROD[31:0]),
    .rnd       (sw_rnd)
);

assign busy        = (state_reg != S_IDLE);
assign round_done  = rd_reg;
assign spins_out   = bank[rep_at[0]];
assign a_spin_data = bank[k_reg];
assign a_beta      = beta_reg;

endmodule

`resetall
