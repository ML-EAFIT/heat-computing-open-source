// -----------------------------------------------------------------------------
// File   : denoise_demo.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Toy denoising chain in the style of a Denoising Thermodynamic Model.
//   start with x^T: for t = T .. 1
//     1. clamp x^t: bias_i = h_i + (x^t_i ? +g_t : -g_t), g_t = Gamma_t / 2
//        (written into the bias RAM of pbit_array, N cycles)
//     2. initialise the spins to x^t and run K Gibbs sweeps -> x^{t-1}
//   step_done pulses after every step (spins = x^{t-1}); done pulses after
//   t = 1 with x0 = x^0. One pbit_array is reused for every step ("the same
//   hardware, reprogrammed with distinct sets of weights at different times").
//
//   Idea source: [data/An efficient ... 2510.23972v2.pdf, p.4 Eqs. 7-8,
//   Fig. 3; p.5 text; p.23 Eq. D1; p.24 Fig. 9b].
//   [INFERRED] clamping a neighbour = constant bias offset.
//   [FPGA-APPROX] hand-set Hebbian marginal weights, no latents, T = 4.
//   Mirror: model/denoise.py.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module denoise_demo #
(
    // number of data nodes (4 x 4 image)
    parameter        N        = 16,
    parameter        IW       = 4,
    parameter        AW       = 14,
    // denoising steps and sweeps per step
    parameter        T        = 4,
    parameter        K        = 16,
    // inverse temperature, Q4.4
    parameter [7:0]  BETA     = 8'd16,
    parameter [31:0] SEED     = 32'hACE1F00D,
    // marginal-energy weights, base bias, visit order, Gamma_t/2 table
    parameter        W_FILE   = "../model/vectors/denoise/weights.hex",
    parameter        H_FILE   = "../model/vectors/denoise/bias.hex",
    parameter        O_FILE   = "../model/vectors/denoise/order.hex",
    parameter        G_FILE   = "../model/vectors/denoise/gamma.hex",
    parameter        LUT_FILE = "../model/vectors/sigmoid_lut.hex"
)
(
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire [N-1:0] x_t,
    output wire         step_done,
    output wire         done,
    output wire [N-1:0] x_out
);

reg [7:0] hbase [0:N-1];
reg [7:0] gtab  [0:T-1];

initial begin
    $readmemh(H_FILE, hbase, 0, N - 1);
    $readmemh(G_FILE, gtab, 0, T - 1);
end

// state encoding: binary, 3 bits
localparam [2:0] S_IDLE = 3'd0,
                 S_LOAD = 3'd1,
                 S_BIAS = 3'd2,
                 S_RUN  = 3'd3,
                 S_WAIT = 3'd4,
                 S_STEP = 3'd5;

reg [2:0]    state_reg = S_IDLE,     state_next;
reg [7:0]    t_reg     = 8'd0,       t_next;       // current step t (T .. 1)
reg [IW:0]   i_reg     = {(IW+1){1'b0}}, i_next;
reg          sd_reg    = 1'b0,       sd_next;
reg          done_reg  = 1'b0,       done_next;
reg [N-1:0]  xt_reg    = {N{1'b0}},  xt_next;      // clamped x^t
reg          h_we, spin_we, run;
reg  [8:0]   hsum;
reg  [7:0]   hval;

wire         busy, swd;
wire [N-1:0] spins;
wire [7:0]   g = gtab[t_reg - 8'd1];

always @* begin
    state_next = state_reg;
    t_next     = t_reg;
    i_next     = i_reg;
    sd_next    = 1'b0;
    done_next  = 1'b0;
    xt_next    = xt_reg;
    h_we       = 1'b0;
    spin_we    = 1'b0;
    run        = 1'b0;
    hsum       = {hbase[i_reg[IW-1:0]][7], hbase[i_reg[IW-1:0]]} +
                 (xt_reg[i_reg[IW-1:0]] ? {g[7], g} : -{g[7], g});
    // saturate to signed 8 bit
    if (hsum[8] != hsum[7]) begin
        hval = hsum[8] ? 8'h80 : 8'h7F;
    end else begin
        hval = hsum[7:0];
    end

    case (state_reg)
        S_IDLE: begin
            if (start) begin
                xt_next    = x_t;
                t_next     = T;
                state_next = S_LOAD;
            end
        end
        S_LOAD: begin                      // spins <- x^t
            spin_we    = 1'b1;
            i_next     = {(IW+1){1'b0}};
            state_next = S_BIAS;
        end
        S_BIAS: begin                      // bias RAM <- h + clamp field
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
            if (!busy) begin
                state_next = S_STEP;
            end
        end
        S_STEP: begin
            sd_next = 1'b1;
            xt_next = spins;
            if (t_reg == 8'd1) begin
                done_next  = 1'b1;
                state_next = S_IDLE;
            end else begin
                t_next     = t_reg - 8'd1;
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
    t_reg     <= t_next;
    i_reg     <= i_next;
    sd_reg    <= sd_next;
    done_reg  <= done_next;
    xt_reg    <= xt_next;

    if (rst) begin
        state_reg <= S_IDLE;
        sd_reg    <= 1'b0;
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
    .spin_we    (spin_we),
    .spin_data  (xt_reg),
    .seed_load  (1'b0),
    .seed_in    (32'd0),
    .run        (run),
    .n_sweeps   (K),
    .hold       (1'b0),
    .stop       (1'b0),
    .beta       (BETA),
    .busy       (busy),
    .sweep_done (swd),
    .spins      (spins)
);

assign step_done = sd_reg;
assign done      = done_reg;
assign x_out     = spins;

endmodule

`resetall
