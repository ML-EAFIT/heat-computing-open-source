// -----------------------------------------------------------------------------
// File   : pbit.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   One p-bit update engine:
//     logit = sat10((beta * field + 2^(S-1)) >>> S), S = WFRAC - 2
//                                                  (= 2*beta*field in Q5.5)
//     p     = sigmoid(logit)                       (LUT or PWL, 16 bit)
//     bit   = (rnd < p)                            (1 = spin +1)
//   beta is unsigned Q4.4 and field has WFRAC fraction bits. Serial
//   shift-add multiplier (8 cycles, no DSP). start -> done takes 12 cycles.
//   rnd_take pulses together with done so the noise lane advances exactly
//   once per update.
//
//   Idea source: P(x_i=+1) = sigma(2 beta (sum_j J_ij x_j + h_i))
//   [data/An efficient ... 2510.23972v2.pdf, p.5 Eq. 11]; the RNG compares
//   noise with a programmable threshold [same, p.37 App. K, p.38 Fig. 15a].
//   [FPGA-APPROX] rnd < p is the inverse-CDF form of that comparator.
//   Mirror: model/fixed.FixedSampler.update.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module pbit #
(
    // field (accumulator) width, signed
    parameter AW       = 14,
    // fraction bits of weights / bias / field (>= 3)
    parameter WFRAC    = 4,
    // 1 = piecewise-linear sigmoid, 0 = BSRAM LUT
    parameter USE_PWL  = 0,
    // sigmoid table for the LUT variant
    parameter LUT_FILE = "../model/vectors/sigmoid_lut.hex"
)
(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    input  wire signed [AW-1:0] field,
    input  wire        [7:0]    beta,
    input  wire        [15:0]   rnd,
    output wire                 bit_out,
    output wire                 done,
    output wire                 rnd_take,
    output wire signed [9:0]    logit_dbg,
    output wire        [15:0]   p_dbg
);

localparam PW    = AW + 9;                  // product width
localparam SHIFT = WFRAC - 2;               // 4 + WFRAC frac bits, x2, -> 5

initial begin
    if (WFRAC < 3) begin
        $error("pbit: WFRAC must be >= 3");
        $finish;
    end
end

// state encoding: binary, 3 bits (5 states)
localparam [2:0] S_IDLE  = 3'd0,
                 S_MUL   = 3'd1,
                 S_LOGIT = 3'd2,
                 S_LUT   = 3'd3,
                 S_CMP   = 3'd4;

reg [2:0]           state_reg = S_IDLE,         state_next;
reg signed [PW-1:0] prod_reg  = {PW{1'b0}},     prod_next;
reg signed [PW-1:0] mcand_reg = {PW{1'b0}},     mcand_next;
reg [7:0]           bsh_reg   = 8'd0,           bsh_next;
reg [2:0]           cnt_reg   = 3'd0,           cnt_next;
reg signed [9:0]    logit_reg = 10'sd0,         logit_next;
reg                 bit_reg   = 1'b0,           bit_next;
reg                 done_reg  = 1'b0,           done_next;
reg                 take_reg  = 1'b0,           take_next;
reg                 lut_en;
reg signed [PW-1:0] rounded;

wire [15:0] p_w;

always @* begin
    state_next = state_reg;
    prod_next  = prod_reg;
    mcand_next = mcand_reg;
    bsh_next   = bsh_reg;
    cnt_next   = cnt_reg;
    logit_next = logit_reg;
    bit_next   = bit_reg;
    done_next  = 1'b0;
    take_next  = 1'b0;
    lut_en     = 1'b0;
    rounded    = (prod_reg + (1 <<< (SHIFT - 1))) >>> SHIFT;

    case (state_reg)
        S_IDLE: begin
            if (start) begin
                prod_next  = {PW{1'b0}};
                mcand_next = {{(PW-AW){field[AW-1]}}, field};
                bsh_next   = beta;
                cnt_next   = 3'd0;
                state_next = S_MUL;
            end
        end
        S_MUL: begin
            if (bsh_reg[0]) begin
                prod_next = prod_reg + mcand_reg;
            end
            mcand_next = mcand_reg <<< 1;
            bsh_next   = bsh_reg >> 1;
            cnt_next   = cnt_reg + 3'd1;
            if (cnt_reg == 3'd7) begin
                state_next = S_LOGIT;
            end
        end
        S_LOGIT: begin
            if (rounded > 511) begin
                logit_next = 10'sd511;
            end else if (rounded < -512) begin
                logit_next = -10'sd512;
            end else begin
                logit_next = rounded[9:0];
            end
            state_next = S_LUT;
        end
        S_LUT: begin
            lut_en     = 1'b1;
            state_next = S_CMP;
        end
        S_CMP: begin
            bit_next   = (rnd < p_w);
            done_next  = 1'b1;
            take_next  = 1'b1;
            state_next = S_IDLE;
        end
        default: begin
            state_next = S_IDLE;
        end
    endcase
end

always @(posedge clk) begin
    state_reg <= state_next;
    prod_reg  <= prod_next;
    mcand_reg <= mcand_next;
    bsh_reg   <= bsh_next;
    cnt_reg   <= cnt_next;
    logit_reg <= logit_next;
    bit_reg   <= bit_next;
    done_reg  <= done_next;
    take_reg  <= take_next;

    if (rst) begin
        state_reg <= S_IDLE;
        done_reg  <= 1'b0;
        take_reg  <= 1'b0;
    end
end

generate
    if (USE_PWL != 0) begin : gen_pwl
        sigmoid_pwl sigmoid_pwl_inst (
            .clk   (clk),
            .en    (lut_en),
            .logit (logit_reg),
            .p     (p_w)
        );
    end else begin : gen_lut
        sigmoid_lut #(
            .AW        (10),
            .PW        (16),
            .INIT_FILE (LUT_FILE)
        ) sigmoid_lut_inst (
            .clk   (clk),
            .en    (lut_en),
            .logit (logit_reg),
            .p     (p_w)
        );
    end
endgenerate

assign bit_out   = bit_reg;
assign done      = done_reg;
assign rnd_take  = take_reg;
assign logit_dbg = logit_reg;
assign p_dbg     = p_w;

endmodule

`resetall
