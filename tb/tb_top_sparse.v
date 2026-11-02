// -----------------------------------------------------------------------------
// File   : tb_top_sparse.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-29
// -----------------------------------------------------------------------------
// Description:
//   End-to-end test of top.v with SPARSE = 1 (N = 256, P = 4 lanes, BSRAM up
//   counters) through its UART only (fast baud: 8 clocks/bit). The chip boots
//   with empty lane RAMs; the testbench programs the 16 x 16 G12 spin glass
//   (G rounds, M node words, E non-zero neighbour words, read from
//   model/vectors/sp_g12_glass16_p4/), sets a constant beta (B), reseeds (X),
//   clears the readout (C), runs 300 sweeps (R), reads the up counters (U),
//   the spins (Q), streams 3 sweeps (S), dumps the histogram (D), reads the
//   spins (Q) and sends an unknown command. Every received byte goes to
//   +dump=<file>; tb/compare.py --top-sparse-dump replays the session in
//   model/fixed.SparseParallelSampler and requires byte-for-byte equality.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-29  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_top_sparse;

parameter VEC  = "../model/vectors/";
parameter CASE = "../model/vectors/sp_g12_glass16_p4/";
localparam CPB = 8;

`include "case.vh"

localparam IW = 8;
localparam NB = CASE_N / 8;

reg        clk = 1'b0;
reg        btn_rst_n = 1'b1;
reg        btn_run_n = 1'b1;
reg        rxd = 1'b1;
wire       txd;
wire [5:0] led;
wire [7:0] rx_byte;
wire       rx_vld;

reg [IW+8:0]  nodew [0:(CASE_P << CASE_LW)-1];
reg [IW+7:0]  nbrw  [0:(CASE_P << (CASE_LW + CASE_DW))-1];
integer   e, r, s, a, fd, nrx, nexp;
reg [IW+8:0]  nw;
reg [IW+7:0]  ew;
reg [1023:0] dump_file;

always #5 clk = ~clk;

top #(
    .N(CASE_N), .IW(IW), .WFRAC(CASE_WFRAC), .AW(8 + 4 + 1), .HIST_BITS(12), .USE_PWL(0),
    .CLK_HZ(8000000), .BAUD(1000000), .DEB_BITS(4), .DEMO_SWEEPS(16),
    .T_FILE({CASE, "beta_table.hex"}), .LUT_FILE({VEC, "sigmoid_lut.hex"}),
    .SPARSE(1), .P(CASE_P), .D(CASE_D), .DW(CASE_DW), .LW(CASE_LW), .ROUNDS(1),
    .NODE_PREFIX(""), .NBR_PREFIX(""), .UP_RAM(1)
) dut (
    .clk(clk), .btn_rst_n(btn_rst_n), .btn_run_n(btn_run_n),
    .uart_rxd(rxd), .uart_txd(txd), .led(led)
);

// host-side receiver
uart_rx #(.CLKS_PER_BIT(CPB)) host_rx (.clk(clk), .rst(1'b0), .rxd(txd), .out_data(rx_byte), .out_valid(rx_vld));

always @(posedge clk) begin
    if (rx_vld) begin
        $fdisplay(fd, "%02x", rx_byte);
        nrx = nrx + 1;
    end
end

task send;
    input [7:0] b;
    integer k;
    begin
        rxd = 1'b0;
        repeat (CPB) @(posedge clk);
        for (k = 0; k < 8; k = k + 1) begin
            rxd = b[k];
            repeat (CPB) @(posedge clk);
        end
        rxd = 1'b1;
        repeat (CPB + 2) @(posedge clk);
    end
endtask

task send32;
    input [31:0] w;
    begin
        send(w[31:24]); send(w[23:16]); send(w[15:8]); send(w[7:0]);
    end
endtask

// expect `n` more reply bytes and wait for them
task expect_rx;
    input integer n;
    begin
        nexp = nexp + n;
        while (nrx < nexp) @(posedge clk);
    end
endtask

initial begin
    for (e = 0; e < CASE_P; e = e + 1) begin
        // lane files are loaded one after the other into the flat arrays
        case (e)
            0: begin $readmemh({CASE, "node_e0.hex"}, nodew, 0 << CASE_LW, (1 << CASE_LW) - 1);
                     $readmemh({CASE, "nbr_e0.hex"}, nbrw, 0, (1 << (CASE_LW + CASE_DW)) - 1); end
            1: begin $readmemh({CASE, "node_e1.hex"}, nodew, 1 << CASE_LW, (2 << CASE_LW) - 1);
                     $readmemh({CASE, "nbr_e1.hex"}, nbrw, 1 << (CASE_LW + CASE_DW), (2 << (CASE_LW + CASE_DW)) - 1); end
            2: begin $readmemh({CASE, "node_e2.hex"}, nodew, 2 << CASE_LW, (3 << CASE_LW) - 1);
                     $readmemh({CASE, "nbr_e2.hex"}, nbrw, 2 << (CASE_LW + CASE_DW), (3 << (CASE_LW + CASE_DW)) - 1); end
            default: begin $readmemh({CASE, "node_e3.hex"}, nodew, 3 << CASE_LW, (4 << CASE_LW) - 1);
                     $readmemh({CASE, "nbr_e3.hex"}, nbrw, 3 << (CASE_LW + CASE_DW), (4 << (CASE_LW + CASE_DW)) - 1); end
        endcase
    end
    if (!$value$plusargs("dump=%s", dump_file)) dump_file = "build/top_sparse_uart_dump.txt";
    fd   = $fopen(dump_file, "w");
    nrx  = 0;
    nexp = 0;
    repeat (40) @(posedge clk);

    send("P");                                   expect_rx(7);
    send("G"); send(CASE_ROUNDS >> 8); send(CASE_ROUNDS & 255);   expect_rx(1);
    for (e = 0; e < CASE_P; e = e + 1) begin
        for (r = 0; r < CASE_ROUNDS; r = r + 1) begin
            nw = nodew[(e << CASE_LW) + r];
            send("M"); send(e[7:0]); send(r[7:0]); send({7'd0, nw[IW+8]}); send(nw[IW+7:8]); send(nw[7:0]);
            expect_rx(1);
            for (s = 0; s < CASE_D; s = s + 1) begin
                a  = (e << (CASE_LW + CASE_DW)) + (r << CASE_DW) + s;
                ew = nbrw[a];
                if (ew != 0) begin
                    send("E"); send(e[7:0]); send(r[7:0]); send(s[7:0]); send(ew[IW+7:8]); send(ew[7:0]);
                    expect_rx(1);
                end
            end
        end
    end
    send("B"); send(8'd0); send(CASE_BETA0[7:0]); send(CASE_BETA0[7:0]); send(8'd0); send(8'd0); send(8'd1);
    expect_rx(1);
    send("X"); send32(32'h12345678);             expect_rx(1);
    send("C");                                   expect_rx(1);
    send("R"); send32(32'd300);                  expect_rx(1);
    send("U");                                   expect_rx(4 + 4 * CASE_N + 1);
    send("Q");                                   expect_rx(NB + 1);
    send("S"); send32(32'd3);                    expect_rx(3 * NB + 1);
    send("D");                                   expect_rx(4 * 4096 + 1);
    send("Q");                                   expect_rx(NB + 1);
    send("y");                                   expect_rx(1);
    repeat (50) @(posedge clk);
    $fclose(fd);
    $display("tb_top_sparse: %0d bytes received over UART", nrx);
    $display("tb_top_sparse: byte stream written to %0s (checked by compare.py --top-sparse-dump)", dump_file);
    $finish;
end

endmodule

`resetall
