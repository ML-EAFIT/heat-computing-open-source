// -----------------------------------------------------------------------------
// File   : noise_bank.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   LANES decorrelated noise sources, one 16-bit draw per lane per en pulse.
//   Each lane is a 32-bit Galois LFSR with its own maximal-length polynomial
//   (different m-sequences, not shifted copies of one) and its own seed
//   lane_seed = seed ^ (0x9E3779B9 * (k+1)) (mirror: model/fixed.lane_seed).
//
//   Optional RO_JITTER = 1 XORs a synchronized ring-oscillator bit into bit 0
//   of every lane. This is a combinational loop. It is NOT simulatable and
//   breaks bit-exact agreement with the Python model. Use only on hardware
//   as an entropy experiment.
//
//   Idea source: arrays of RNGs, one per sampling cell
//   [thrml/00_probabilistic_computing.py:L14]; physical RNG in
//   [data/An efficient ... 2510.23972v2.pdf, p.37 App. K].
//   [FPGA-APPROX] pseudo-random lanes stand in for physical shot noise.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module noise_bank #
(
    // number of independent lanes (1..4)
    parameter        LANES     = 4,
    // base seed
    parameter [31:0] SEED      = 32'hACE1F00D,
    // 1 = XOR ring-oscillator jitter into the lanes (hardware only)
    parameter        RO_JITTER = 0
)
(
    input  wire               clk,
    input  wire               rst,
    input  wire [LANES-1:0]   en,
    input  wire               seed_load,
    input  wire [31:0]        seed_in,
    output wire [LANES*16-1:0] rnd
);

initial begin
    if (LANES < 1 || LANES > 4) begin
        $error("noise_bank: LANES must be 1..4");
        $finish;
    end
end

// per-lane maximal masks, lane 0 in the low word (see model/fixed.LFSR_MASKS)
localparam [127:0] MASKS = {32'h80000057, 32'h80200003, 32'hA3000000, 32'hB4BCD35C};

wire jitter_bit;

generate
    if (RO_JITTER != 0) begin : gen_ro
        // three-inverter ring oscillator, kept through synthesis
        (* keep = "true" *) wire ro_a;
        (* keep = "true" *) wire ro_b;
        (* keep = "true" *) wire ro_c;
        assign ro_a = ~ro_c;
        assign ro_b = ~ro_a;
        assign ro_c = ~ro_b;
        // CDC: free-running ring oscillator -> clk (two-flop synchronizer)
        reg ro_s1_reg = 1'b0;
        reg ro_s2_reg = 1'b0;
        always @(posedge clk) begin
            ro_s1_reg <= ro_c;
            ro_s2_reg <= ro_s1_reg;
        end
        assign jitter_bit = ro_s2_reg;
    end else begin : gen_no_ro
        assign jitter_bit = 1'b0;
    end
endgenerate

genvar g;
generate
    for (g = 0; g < LANES; g = g + 1) begin : gen_lane
        localparam [63:0] PROD   = 64'h9E3779B9 * (g + 1);
        localparam [31:0] LSEED0 = SEED ^ PROD[31:0];
        localparam [31:0] LSEED  = (LSEED0 == 32'd0) ? 32'd1 : LSEED0;

        wire [15:0] lane_rnd;
        wire [31:0] lane_seed_in = seed_in ^ PROD[31:0];

        lfsr #(
            .MASK     (MASKS[g*32 +: 32]),
            .SEED     (LSEED),
            .OUT_BITS (16)
        ) lfsr_inst (
            .clk       (clk),
            .rst       (rst),
            .en        (en[g]),
            .seed_load (seed_load),
            .seed_in   (lane_seed_in),
            .rnd       (lane_rnd)
        );

        assign rnd[g*16 +: 16] = {lane_rnd[15:1], lane_rnd[0] ^ jitter_bit};
    end
endgenerate

endmodule

`resetall
