// -----------------------------------------------------------------------------
// File   : pmode_array.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-30
// -----------------------------------------------------------------------------
// Description:
//   M continuous p-modes (Langevin soft spins) with one time-multiplexed
//   engine. One step updates every site from the old state (Jacobi):
//     x' = x - 2^-k dV/dx + sig * xi
//     dV/dx_i = -(1 - s_i^2)(sum_j A_ij s_j + h_i) + lambda x_i (x_i^2 - 1)
//     s = tanh x = 2 sigma(2x) - 1, read from the p-bit sigmoid LUT
//     xi = inverse-CDF Gaussian table (second LUT ROM), addressed by the top
//          10 bits of a 16-bit draw of noise lane 0
//   Q-formats and rounding: model/pmode.py (bit-exact mirror). x is Q4.12
//   (16-bit, saturated), A and h Q3.4, lambda Q4.4, sig = sqrt(2 T eps) Q4.12.
//   Cycles per step: 2M (tanh) + M (2M + 5) (sites) + 1 (measured by tb_pmode).
//
//   Idea source: overdamped Langevin soft-spin gate
//   [torx/helpers/_langevin.py:L46-143]; continuous pmodes / Gaussian sites
//   [data/A Framework ... 2608.01612v2.pdf, p.21]; double-well neurons
//   [data/A Blueprint ... 2607.16183v2.pdf, p.3-6].
//   [FPGA-APPROX] fixed point, LUT tanh and Gaussian, ULA without Metropolis.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-30  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module pmode_array #
(
    // sites (2^MW >= M, MW >= 1)
    parameter        M          = 4,
    parameter        MW         = 2,
    parameter [31:0] SEED       = 32'hACE1F00D,
    // initial couplings {i, j}, fields, states ("" = zeros)
    parameter        A_FILE     = "",
    parameter        H_FILE     = "",
    parameter        X_FILE     = "",
    parameter        LUT_FILE   = "../model/vectors/sigmoid_lut.hex",
    parameter        GAUSS_FILE = "../model/vectors/gauss_lut.hex"
)
(
    input  wire              clk,
    input  wire              rst,
    // programming (idle only)
    input  wire              a_we,
    input  wire [2*MW-1:0]   a_addr,
    input  wire [7:0]        a_data,
    input  wire              h_we,
    input  wire [MW-1:0]     h_addr,
    input  wire [7:0]        h_data,
    input  wire              x_we,
    input  wire [MW-1:0]     x_addr,
    input  wire [15:0]       x_data,
    // dynamics constants
    input  wire [7:0]        lam,          // Q4.4
    input  wire [3:0]        kshift,       // eps = 2^-kshift
    input  wire [15:0]       sig,          // sqrt(2 T eps), Q4.12
    // run control
    input  wire              run,
    input  wire [31:0]       n_steps,
    output wire              busy,
    output wire              step_done,
    output wire [16*M-1:0]   x_flat        // site i in [16 i +: 16]
);

initial begin
    if ((1 << MW) < M || M < 1) begin
        $error("pmode_array: MW too small");
        $finish;
    end
end

// state encoding: binary, 4 bits (11 states)
localparam [3:0] S_IDLE = 4'd0,
                 S_T1   = 4'd1,
                 S_T2   = 4'd2,
                 S_MA   = 4'd3,
                 S_MB   = 4'd4,
                 S_G1   = 4'd5,
                 S_G2   = 4'd6,
                 S_G3   = 4'd7,
                 S_G4   = 4'd8,
                 S_G5   = 4'd9,
                 S_CMT  = 4'd10;

localparam integer MI   = M;
localparam [MW:0]  MLIM = MI[MW:0];

reg [3:0]           state_reg = S_IDLE, state_next;
reg [MW:0]          i_reg     = {(MW+1){1'b0}}, i_next;   // site being updated / tanh index
reg [MW:0]          j_reg     = {(MW+1){1'b0}}, j_next;   // MAC index
reg [31:0]          left_reg  = 32'd0, left_next;
reg                 sd_reg    = 1'b0,  sd_next;
reg signed [39:0]   acc_reg   = 40'sd0, acc_next;         // Q.19 field
reg signed [7:0]    aq_reg    = 8'sd0;
reg signed [16:0]   sq_reg    = 17'sd0;
reg signed [39:0]   f_reg     = 40'sd0;                   // Q.12 field
reg signed [39:0]   g1_reg    = 40'sd0;                   // Q.15, 1 - s^2
reg signed [39:0]   x2_reg    = 40'sd0;                   // Q.12, x^2
reg signed [39:0]   cpl_reg   = 40'sd0;
reg signed [39:0]   q_reg     = 40'sd0;
reg signed [39:0]   qt_reg    = 40'sd0;
reg signed [39:0]   nz_reg    = 40'sd0;
reg signed [15:0]   xi_reg    = 16'sd0;
reg                 lut_en, g_en, take;
reg  signed [9:0]   lut_idx;

reg signed [7:0]    amem  [0:(2**(2*MW))-1];
reg signed [7:0]    hmem  [0:(2**MW)-1];
reg signed [15:0]   xmem  [0:(2**MW)-1];
reg signed [15:0]   xnew  [0:(2**MW)-1];
reg signed [16:0]   smem  [0:(2**MW)-1];                  // tanh, Q.15

integer m;
initial begin
    for (m = 0; m < 2**(2*MW); m = m + 1) amem[m] = 8'sd0;
    for (m = 0; m < 2**MW; m = m + 1) begin
        hmem[m] = 8'sd0;
        xmem[m] = 16'sd0;
        xnew[m] = 16'sd0;
        smem[m] = 17'sd0;
    end
    if (A_FILE != "") $readmemh(A_FILE, amem, 0, (2**(2*MW))-1);
    if (H_FILE != "") $readmemh(H_FILE, hmem, 0, (2**MW)-1);
    if (X_FILE != "") $readmemh(X_FILE, xmem, 0, (2**MW)-1);
end

wire [15:0] p_w;                   // sigmoid LUT
wire [15:0] g_w;                   // Gaussian LUT
wire [15:0] rnd;
wire [MW-1:0] i_s = i_reg[MW-1:0];
wire [MW-1:0] j_s = j_reg[MW-1:0];

// x of the current site, and sat((x + 32) >>> 6) as LUT logit (Q5.5 of 2x)
wire signed [15:0] xi_w   = xmem[i_s];
wire signed [16:0] x_rnd  = {xi_w[15], xi_w} + 17'sd32;
wire signed [10:0] x_lg   = x_rnd[16:6];
wire signed [9:0]  x_lgs  = (x_lg > 11'sd511) ? 10'sd511 : (x_lg < -11'sd512) ? -10'sd512 : x_lg[9:0];

// gradient pieces (all signed 40-bit, >>> = floor like Python >>)
wire signed [39:0] x40     = {{24{xi_w[15]}}, xi_w};
wire signed [39:0] s40     = {{23{smem[i_s][16]}}, smem[i_s]};
wire signed [39:0] lam40   = {32'd0, lam};
wire signed [39:0] sig40   = {24'd0, sig};
wire signed [39:0] gss40   = {{24{xi_reg[15]}}, xi_reg};
wire signed [39:0] grad    = qt_reg - cpl_reg;
wire signed [39:0] xn      = x40 - (grad >>> kshift) + nz_reg;
wire signed [15:0] xn_sat  = (xn > 40'sd32767) ? 16'sh7FFF : (xn < -40'sd32768) ? 16'sh8000 : xn[15:0];

always @* begin
    state_next = state_reg;
    i_next     = i_reg;
    j_next     = j_reg;
    left_next  = left_reg;
    sd_next    = 1'b0;
    acc_next   = acc_reg;
    lut_en     = 1'b0;
    g_en       = 1'b0;
    take       = 1'b0;
    lut_idx    = x_lgs;

    case (state_reg)
        S_IDLE: begin
            if (run && n_steps != 32'd0) begin
                left_next  = n_steps;
                i_next     = {(MW+1){1'b0}};
                state_next = S_T1;
            end
        end
        S_T1: begin                                 // LUT <- tanh address of x_i
            lut_en     = 1'b1;
            state_next = S_T2;
        end
        S_T2: begin                                 // smem[i] <- p - 32768 (below)
            if (i_reg == MLIM - 1'b1) begin
                i_next     = {(MW+1){1'b0}};
                j_next     = {(MW+1){1'b0}};
                acc_next   = {{17{hmem[0][7]}}, hmem[0], 15'd0};
                state_next = S_MA;
            end else begin
                i_next     = i_reg + 1'b1;
                state_next = S_T1;
            end
        end
        S_MA: begin                                 // aq, sq <- A[i][j], s[j] (below)
            state_next = S_MB;
        end
        S_MB: begin
            acc_next = acc_reg + aq_reg * sq_reg;
            if (j_reg == MLIM - 1'b1) begin
                state_next = S_G1;
            end else begin
                j_next     = j_reg + 1'b1;
                state_next = S_MA;
            end
        end
        S_G1: begin                                 // F, 1 - s^2, x^2; Gaussian LUT read
            g_en       = 1'b1;
            take       = 1'b1;
            state_next = S_G2;
        end
        S_G2: begin                                 // xi valid next; cpl, q
            state_next = S_G3;
        end
        S_G3: begin                                 // quart, noise
            state_next = S_G4;
        end
        S_G4: begin
            state_next = S_G5;
        end
        S_G5: begin                                 // xnew[i] <- sat(x - grad >> k + noise)
            if (i_reg == MLIM - 1'b1) begin
                state_next = S_CMT;
            end else begin
                i_next     = i_reg + 1'b1;
                j_next     = {(MW+1){1'b0}};
                acc_next   = {{17{hmem[i_s + 1'b1][7]}}, hmem[i_s + 1'b1], 15'd0};
                state_next = S_MA;
            end
        end
        S_CMT: begin
            sd_next   = 1'b1;
            left_next = left_reg - 32'd1;
            i_next    = {(MW+1){1'b0}};
            if (left_reg == 32'd1) begin
                state_next = S_IDLE;
            end else begin
                state_next = S_T1;
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
    j_reg     <= j_next;
    left_reg  <= left_next;
    sd_reg    <= sd_next;
    acc_reg   <= acc_next;

    if (state_reg == S_IDLE) begin
        if (a_we) amem[a_addr] <= a_data;
        if (h_we) hmem[h_addr] <= h_data;
        if (x_we) xmem[x_addr] <= x_data;
    end
    if (state_reg == S_T2) begin
        smem[i_s] <= {1'b0, p_w} - 17'sd32768;
    end
    if (state_reg == S_MA) begin
        aq_reg <= amem[{i_s, j_s}];
        sq_reg <= smem[j_s];
    end
    if (state_reg == S_G1) begin
        f_reg  <= acc_reg >>> 7;
        g1_reg <= ((40'sd1 <<< 30) - s40 * s40) >>> 15;
        x2_reg <= (x40 * x40) >>> 12;
    end
    if (state_reg == S_G2) begin
        cpl_reg <= (f_reg * g1_reg) >>> 15;
        q_reg   <= (x40 * (x2_reg - 40'sd4096)) >>> 12;
    end
    if (state_reg == S_G3) begin
        xi_reg  <= g_w;
        qt_reg  <= (lam40 * q_reg) >>> 4;
    end
    if (state_reg == S_G4) begin
        nz_reg  <= (sig40 * gss40) >>> 12;
    end
    if (state_reg == S_G5) begin
        xnew[i_s] <= xn_sat;
    end
    if (state_reg == S_CMT) begin
        for (m = 0; m < M; m = m + 1) begin
            xmem[m] <= xnew[m];
        end
    end

    if (rst) begin
        state_reg <= S_IDLE;
        left_reg  <= 32'd0;
        sd_reg    <= 1'b0;
    end
end

// tanh via the p-bit sigmoid table: s = 2 sigma(2x) - 1 = (p - 32768) / 32768
sigmoid_lut #(
    .AW        (10),
    .PW        (16),
    .INIT_FILE (LUT_FILE)
) tanh_lut_inst (
    .clk   (clk),
    .en    (lut_en),
    .logit (lut_idx),
    .p     (p_w)
);

// Gaussian inverse-CDF table: same ROM module, address = top 10 bits of the draw
sigmoid_lut #(
    .AW        (10),
    .PW        (16),
    .INIT_FILE (GAUSS_FILE)
) gauss_lut_inst (
    .clk   (clk),
    .en    (g_en),
    .logit ({~rnd[15], rnd[14:6]}),
    .p     (g_w)
);

noise_bank #(
    .LANES     (1),
    .SEED      (SEED),
    .RO_JITTER (0)
) noise_bank_inst (
    .clk       (clk),
    .rst       (rst),
    .en        (take),
    .seed_load (1'b0),
    .seed_in   (32'd0),
    .rnd       (rnd)
);

genvar g;
generate
    for (g = 0; g < M; g = g + 1) begin : gen_out
        assign x_flat[16*g +: 16] = xmem[g];
    end
endgenerate

assign busy      = (state_reg != S_IDLE);
assign step_done = sd_reg;

endmodule

`resetall
