// -----------------------------------------------------------------------------
// File   : pmode_top.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-30
// -----------------------------------------------------------------------------
// Description:
//   Stand-alone Tang Nano 9K top for pmode_array (continuous Langevin
//   p-modes). The power-on program (couplings, fields, initial state,
//   constants) comes from parameters / files; there is no host link.
//   Button S2 toggles running; while running one Langevin step is taken
//   every 2^PACE_BITS clocks (2^20 / 27 MHz = 26 steps/s, so the LEDs show
//   the dynamics; PACE_BITS = 0 runs at full speed). Button S1: reset.
//   LEDs (active low): led[k] = sign of x_k (lit = x_k > 0) for k < 5,
//   led[5] = running. UART pins are idle.
//   Mirror of the dynamics: model/pmode.py.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-30  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module pmode_top #
(
    parameter        M          = 4,
    parameter        MW         = 2,
    parameter [7:0]  LAM        = 8'd32,
    parameter [3:0]  KSHIFT     = 4'd4,
    parameter [15:0] SIG        = 16'd1448,
    parameter        PACE_BITS  = 20,
    parameter        DEB_BITS   = 18,
    parameter [31:0] SEED       = 32'hACE1F00D,
    parameter        A_FILE     = "",
    parameter        H_FILE     = "",
    parameter        X_FILE     = "",
    parameter        LUT_FILE   = "../model/vectors/sigmoid_lut.hex",
    parameter        GAUSS_FILE = "../model/vectors/gauss_lut.hex"
)
(
    input  wire       clk,
    input  wire       btn_rst_n,
    input  wire       btn_run_n,
    input  wire       uart_rxd,
    output wire       uart_txd,
    output wire [5:0] led
);

localparam PB = (PACE_BITS < 1) ? 1 : PACE_BITS;

reg [3:0]          por_reg   = 4'd0;
reg                rst_s1    = 1'b0, rst_s2 = 1'b0;     // CDC: button -> clk
reg                run_s1    = 1'b1, run_s2 = 1'b1;
reg [DEB_BITS-1:0] deb_reg   = {DEB_BITS{1'b0}};
reg                run_stable = 1'b1, run_prev = 1'b1;
reg                on_reg    = 1'b0;
reg [PB-1:0]       pace_reg  = {PB{1'b0}};
wire               rst = (por_reg != 4'hF) || !rst_s2;
wire               busy, step_done;
wire [16*M-1:0]    x_flat;
wire               tick = (PACE_BITS == 0) ? 1'b1 : (pace_reg == {PB{1'b0}});

always @(posedge clk) begin
    rst_s1 <= btn_rst_n;
    rst_s2 <= rst_s1;
    run_s1 <= btn_run_n;
    run_s2 <= run_s1;
    if (por_reg != 4'hF) por_reg <= por_reg + 4'd1;
    if (run_s2 != run_stable) begin
        deb_reg <= deb_reg + 1'b1;
        if (&deb_reg) begin
            run_stable <= run_s2;
            deb_reg    <= {DEB_BITS{1'b0}};
        end
    end else begin
        deb_reg <= {DEB_BITS{1'b0}};
    end
    run_prev <= run_stable;
    if (run_prev && !run_stable) on_reg <= ~on_reg;     // press toggles
    pace_reg <= pace_reg + 1'b1;
    if (rst) on_reg <= 1'b0;
end

pmode_array #(
    .M          (M),
    .MW         (MW),
    .SEED       (SEED),
    .A_FILE     (A_FILE),
    .H_FILE     (H_FILE),
    .X_FILE     (X_FILE),
    .LUT_FILE   (LUT_FILE),
    .GAUSS_FILE (GAUSS_FILE)
) pmode_array_inst (
    .clk       (clk),
    .rst       (rst),
    .a_we      (1'b0),
    .a_addr    ({(2*MW){1'b0}}),
    .a_data    (8'd0),
    .h_we      (1'b0),
    .h_addr    ({MW{1'b0}}),
    .h_data    (8'd0),
    .x_we      (1'b0),
    .x_addr    ({MW{1'b0}}),
    .x_data    (16'd0),
    .lam       (LAM),
    .kshift    (KSHIFT),
    .sig       (SIG),
    .run       (on_reg && tick && !busy),
    .n_steps   (32'd1),
    .busy      (busy),
    .step_done (step_done),
    .x_flat    (x_flat)
);

genvar g;
generate
    for (g = 0; g < 5; g = g + 1) begin : gen_led
        if (g < M) begin : gen_site
            assign led[g] = x_flat[16*g + 15];          // active low: lit when x > 0
        end else begin : gen_none
            assign led[g] = 1'b1;
        end
    end
endgenerate

assign led[5]   = ~on_reg;
assign uart_txd = 1'b1;

endmodule

`resetall
