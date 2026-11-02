// -----------------------------------------------------------------------------
// File   : tb_top.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   End-to-end test of top.v through its UART only (fast baud: 8 clocks/bit).
//   The chip boots with the maxcut16 model; the testbench reprograms it with
//   the grid16 model (W/H/O), sets a constant beta (B), reseeds (X), clears
//   the readout (C), runs 2000 sweeps (R), reads the up counters (U), the
//   spins (Q), streams 5 sweeps (S), dumps the 4096-bin histogram (D), presses
//   button S2 (128-sweep demo run), reads the spins again (Q) and sends an
//   unknown command. Every received byte is written to +dump=<file>;
//   tb/compare.py --top-dump replays the same sequence in model/fixed.py and
//   requires byte-for-byte equality.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_top;

parameter VEC = "../model/vectors/";
localparam CPB = 8;

reg        clk = 1'b0;
reg        btn_rst_n = 1'b1;
reg        btn_run_n = 1'b1;
reg        rxd = 1'b1;
wire       txd;
wire [5:0] led;
wire [7:0] rx_byte;
wire       rx_vld;

reg [7:0] wts [0:255];
reg [7:0] bia [0:15];
reg [3:0] ord [0:15];
integer   i, fd, nrx, saw_busy;
reg [1023:0] dump_file;

always #5 clk = ~clk;

top #(
    .N(16), .IW(4), .WFRAC(4), .AW(14), .HIST_BITS(12), .USE_PWL(0),
    .CLK_HZ(8000000), .BAUD(1000000), .DEB_BITS(4), .DEMO_SWEEPS(128),
    .W_FILE({VEC, "maxcut16/weights.hex"}), .H_FILE({VEC, "maxcut16/bias.hex"}),
    .O_FILE({VEC, "maxcut16/order.hex"}), .T_FILE({VEC, "maxcut16/beta_table.hex"}),
    .LUT_FILE({VEC, "sigmoid_lut.hex"})
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

// wait until `n` total bytes have been received
task wait_rx;
    input integer n;
    begin
        while (nrx < n) @(posedge clk);
    end
endtask

initial begin
    $readmemh({VEC, "grid16/weights.hex"}, wts);
    $readmemh({VEC, "grid16/bias.hex"}, bia);
    $readmemh({VEC, "grid16/order.hex"}, ord);
    if (!$value$plusargs("dump=%s", dump_file)) dump_file = "build/top_uart_dump.txt";
    fd  = $fopen(dump_file, "w");
    nrx = 0;
    repeat (40) @(posedge clk);

    send("P");                                   wait_rx(3);
    for (i = 0; i < 256; i = i + 1) begin
        send("W"); send(8'd0); send(i[7:0]); send(wts[i]);
    end
    for (i = 0; i < 16; i = i + 1) begin
        send("H"); send(i[7:0]); send(bia[i]);
        send("O"); send(i[7:0]); send({4'd0, ord[i]});
    end
    wait_rx(3 + 256 + 32);
    send("B"); send(8'd0); send(8'd7); send(8'd7); send(8'd0); send(8'd0); send(8'd1);
    send("X"); send32(32'h12345678);
    send("C");                                   wait_rx(294);
    send("R"); send32(32'd2000);                 wait_rx(295);
    send("U");                                   wait_rx(295 + 68 + 1);
    send("Q");                                   wait_rx(364 + 3);
    send("S"); send32(32'd5);                    wait_rx(367 + 10 + 1);
    send("D");                                   wait_rx(378 + 16384 + 1);
    // button S2: 128-sweep demo run, no UART reply
    saw_busy = 0;
    btn_run_n = 1'b0;
    repeat (200) @(posedge clk);
    btn_run_n = 1'b1;
    for (i = 0; i < 200000 && !(saw_busy && led[5]); i = i + 1) begin
        @(posedge clk);
        if (!led[5]) saw_busy = 1;
    end
    send("Q");                                   wait_rx(16763 + 3);
    send("y");                                   wait_rx(16766 + 1);
    repeat (50) @(posedge clk);
    $fclose(fd);
    $display("tb_top: %0d bytes received over UART, button run observed busy=%0d", nrx, saw_busy);
    $display("tb_top: byte stream written to %0s (checked by compare.py --top-dump)", dump_file);
    $finish;
end

endmodule

`resetall
