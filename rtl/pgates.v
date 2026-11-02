// -----------------------------------------------------------------------------
// File   : pgates.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-25
// -----------------------------------------------------------------------------
// Description:
//   p-gate: a Boolean gate (or a circuit of gates) run as a Boltzmann sampler.
//   The gate's couplings J and biases h (model/pgates.py, exported to
//   model/vectors/<case>/) make the truth-table rows the lowest-energy states,
//   so free-running Gibbs sweeps spend most of their time on valid rows.
//   Any node can be clamped to 0/1. Clamping the inputs runs the gate
//   forward; clamping the output runs it backward (invertible logic):
//     start : latch clamp_mask / clamp_val, rewrite the bias RAM
//             h_i = clamp_mask[i] ? (clamp_val[i] ? +127 : -128) : h0_i
//             (N cycles), then run n_sweeps Gibbs sweeps
//   sweep_done pulses once per sweep with the post-sweep spins; done pulses
//   when the run ends. Bit i of spins = node i, 1 = logic 1 = spin +1.
//
//   Idea source: energy composition of AND / OR / NOT is mentioned
//   [data/A Blueprint ... 2607.16183v2.pdf, p.3]; Boltzmann law of the p-bit
//   array [data/An efficient ... 2510.23972v2.pdf, p.5, Eqs. 10-11].
//   [NOT IN SOURCES] the gadget coefficients (standard Ising gadgets).
//   [INFERRED] clamping = bias offset (docs/02 2.3).
//   [FPGA-APPROX] one time-multiplexed pbit_array per gate circuit.
//   Mirror: model/pgates.py + model/fixed.FixedSampler.
//
//   K > 1: parallel tempering (temper_ctrl.v). K replicas at the betas of
//   LADDER (rung 0 = target, the beta port is unused); n_sweeps counts
//   rounds (K sweeps + swaps), sweep_done pulses per round with the rung-0
//   replica. start resyncs the replicas (all <- rung-0 replica, energies 0),
//   because the new clamp changes every energy. Mirror: model/tempering.py.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-25  jero98772  v1.0  Initial version
//   2026-10-30  jero98772  v1.1  K > 1: parallel tempering (temper_ctrl)
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module pgate #
(
    // p-bits in the circuit (inputs, outputs, internal wires, aux spins)
    parameter        N        = 3,
    parameter        IW       = 2,
    parameter        AW       = 11,
    parameter [31:0] SEED     = 32'hACE1F00D,
    // gate coefficients and visit order (model/export_vectors.py)
    parameter        W_FILE   = "../model/vectors/pgate_and/weights.hex",
    parameter        H_FILE   = "../model/vectors/pgate_and/bias.hex",
    parameter        O_FILE   = "../model/vectors/pgate_and/order.hex",
    parameter        LUT_FILE = "../model/vectors/sigmoid_lut.hex",
    // replicas (1 = plain Gibbs) and their ladder {b3, b2, b1, b0}
    parameter        K        = 1,
    parameter [31:0] LADDER   = 32'h0C_14_20_30
)
(
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire [N-1:0] clamp_mask,
    input  wire [N-1:0] clamp_val,
    input  wire [31:0]  n_sweeps,
    input  wire [7:0]   beta,
    output wire         busy,
    output wire         sweep_done,
    output wire         done,
    output wire [N-1:0] spins
);

// bias file is zero-padded to 2^IW entries
reg [7:0] hbase [0:(2**IW)-1];

initial begin
    $readmemh(H_FILE, hbase);
end

// state encoding: binary, 2 bits
localparam [1:0] S_IDLE = 2'd0,
                 S_BIAS = 2'd1,
                 S_RUN  = 2'd2,
                 S_WAIT = 2'd3;

reg [1:0]    state_reg = S_IDLE,         state_next;
reg [IW:0]   i_reg     = {(IW+1){1'b0}}, i_next;
reg [N-1:0]  mask_reg  = {N{1'b0}},      mask_next;
reg [N-1:0]  val_reg   = {N{1'b0}},      val_next;
reg [31:0]   nsw_reg   = 32'd0,          nsw_next;
reg          done_reg  = 1'b0,           done_next;
reg          h_we, run, resync;
reg  [7:0]   hval;

wire         arr_busy;
wire         run_busy;                      // array, or the tempering controller
wire         a_spin_we, a_run;
wire [N-1:0] a_spin_data;
wire [7:0]   a_beta;
wire         a_swd;
wire [N-1:0] a_spins;
wire         upd_we, upd_old, upd_new;
wire signed [AW-1:0] upd_field;

always @* begin
    state_next = state_reg;
    i_next     = i_reg;
    mask_next  = mask_reg;
    val_next   = val_reg;
    nsw_next   = nsw_reg;
    done_next  = 1'b0;
    h_we       = 1'b0;
    run        = 1'b0;
    resync     = 1'b0;
    // clamped node: extreme bias, otherwise the gate's own bias
    if (mask_reg[i_reg[IW-1:0]]) begin
        hval = val_reg[i_reg[IW-1:0]] ? 8'h7F : 8'h80;
    end else begin
        hval = hbase[i_reg[IW-1:0]];
    end

    case (state_reg)
        S_IDLE: begin
            if (start) begin
                resync     = 1'b1;
                mask_next  = clamp_mask;
                val_next   = clamp_val;
                nsw_next   = n_sweeps;
                i_next     = {(IW+1){1'b0}};
                state_next = S_BIAS;
            end
        end
        S_BIAS: begin                      // bias RAM <- clamps / h0
            h_we   = 1'b1;
            i_next = i_reg + 1'b1;
            if (i_reg == N - 1) begin
                state_next = S_RUN;
            end
        end
        S_RUN: begin
            run        = 1'b1;
            state_next = S_WAIT;
        end
        S_WAIT: begin
            if (!run_busy) begin
                done_next  = 1'b1;
                state_next = S_IDLE;
            end
        end
        default: begin
            state_next = S_IDLE;
        end
    endcase
end

always @(posedge clk) begin
    state_reg <= state_next;
    i_reg     <= i_next;
    mask_reg  <= mask_next;
    val_reg   <= val_next;
    nsw_reg   <= nsw_next;
    done_reg  <= done_next;

    if (rst) begin
        state_reg <= S_IDLE;
        done_reg  <= 1'b0;
    end
end

pbit_array #(
    .N        (N),
    .IW       (IW),
    .WB       (8),
    .WFRAC    (4),
    .AW       (AW),
    .USE_PWL  (0),
    .SEED     (SEED),
    .W_FILE   (W_FILE),
    .H_FILE   (H_FILE),
    .O_FILE   (O_FILE),
    .LUT_FILE (LUT_FILE)
) pbit_array_inst (
    .clk        (clk),
    .rst        (rst),
    .w_we       (1'b0),
    .w_addr     ({2*IW{1'b0}}),
    .w_data     (8'd0),
    .h_we       (h_we),
    .h_addr     (i_reg[IW-1:0]),
    .h_data     (hval),
    .o_we       (1'b0),
    .o_addr     ({IW{1'b0}}),
    .o_data     ({IW{1'b0}}),
    .spin_we    (a_spin_we),
    .spin_data  (a_spin_data),
    .seed_load  (1'b0),
    .seed_in    (32'd0),
    .run        (a_run),
    .n_sweeps   ((K > 1) ? 32'd1 : nsw_reg),
    .hold       (1'b0),
    .stop       (1'b0),
    .beta       (a_beta),
    .busy       (arr_busy),
    .sweep_done (a_swd),
    .spins      (a_spins),
    .upd_we     (upd_we),
    .upd_old    (upd_old),
    .upd_new    (upd_new),
    .upd_field  (upd_field)
);

generate
    if (K > 1) begin : gen_pt
        temper_ctrl #(
            .N        (N),
            .AW       (AW),
            .WFRAC    (4),
            .K        (K),
            .LADDER   (LADDER),
            .SEED     (SEED),
            .LUT_FILE (LUT_FILE)
        ) temper_ctrl_inst (
            .clk         (clk),
            .rst         (rst),
            .ladder_we   (1'b0),
            .ladder_addr (2'd0),
            .ladder_data (8'd0),
            .reset_all   (1'b0),
            .seed_load   (1'b0),
            .seed_in     (32'd0),
            .resync      (resync),
            .run         (run),
            .n_rounds    (nsw_reg),
            .hold        (1'b0),
            .busy        (run_busy),
            .round_done  (sweep_done),
            .spins_out   (spins),
            .a_spin_we   (a_spin_we),
            .a_spin_data (a_spin_data),
            .a_run       (a_run),
            .a_beta      (a_beta),
            .a_busy      (arr_busy),
            .a_swd       (a_swd),
            .a_spins     (a_spins),
            .upd_we      (upd_we),
            .upd_old     (upd_old),
            .upd_new     (upd_new),
            .upd_field   (upd_field)
        );
    end else begin : gen_plain
        assign a_spin_we   = 1'b0;
        assign a_spin_data = {N{1'b0}};
        assign a_run       = run;
        assign a_beta      = beta;
        assign run_busy    = arr_busy;
        assign sweep_done  = a_swd;
        assign spins       = a_spins;
    end
endgenerate

assign busy = (state_reg != S_IDLE);
assign done = done_reg;

endmodule

`resetall
