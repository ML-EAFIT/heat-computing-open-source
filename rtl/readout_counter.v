// -----------------------------------------------------------------------------
// File   : readout_counter.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   On-chip readout of a p-bit array.
//     * n_samples : number of recorded samples
//     * up[i]     : how many samples had spin i = +1 (magnetisation counters)
//     * hist[x]   : full-state histogram over the low HIST_BITS spins
//                   (2^HIST_BITS x CW BSRAM; HIST_BITS = 0 disables it)
//   sample records `spins` (read-modify-write, 3 cycles, busy while active).
//   clear zeroes every counter (2^HIST_BITS cycles). Host reads: hist_q is
//   valid one cycle after hist_addr; up_q is a registered mux of up[up_sel].
//   Reads are only meaningful while busy = 0.
//   UP_RAM = 1 keeps up[] in one BSRAM instead of N x CW flip-flops: each
//   sample snapshots the spins and a pipelined read-modify-write scan adds
//   them in N + 1 cycles (busy meanwhile; a sweep of sparse_array is always
//   longer). up_q is then the registered RAM output, same 1-cycle latency.
//   A clear that arrives while such a scan is still running (e.g. 'C' sent
//   right after the last sample of a run) is held pending and runs as soon as
//   the scan ends; busy stays high until the zeroing scan is done.
//
//   Idea source: a readout is a sample, histograms/expectations are built
//   from raw samples [torx/04_execution_interface_readouts.py:L98-118, L202];
//   get-state lines per cell [data/An efficient ... 2510.23972v2.pdf, p.25
//   Fig. 10]. [FPGA-APPROX] counting on chip avoids streaming every sample.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
//   2026-10-29  jero98772  v1.1  UP_RAM option (large N)
//   2026-10-31  jero98772  v1.2  UP_RAM: a clear during an up-scan was lost
//                                (stale counters); now held pending
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module readout_counter #
(
    // number of spins
    parameter N         = 16,
    // spin index width, clog2(N)
    parameter IW        = 4,
    // histogram index width (0 = no histogram, max 12)
    parameter HIST_BITS = 8,
    // counter width
    parameter CW        = 32,
    // 1 = up counters in BSRAM (scan per sample), 0 = flip-flops
    parameter UP_RAM    = 0
)
(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 clear,
    input  wire                 sample,
    input  wire [N-1:0]         spins,
    input  wire [((HIST_BITS < 1) ? 1 : HIST_BITS)-1:0] hist_addr,
    output wire [CW-1:0]        hist_q,
    input  wire [IW-1:0]        up_sel,
    output wire [CW-1:0]        up_q,
    output wire [CW-1:0]        n_samples,
    output wire                 busy
);

localparam HB = (HIST_BITS < 1) ? 1 : HIST_BITS;   // port width when disabled

initial begin
    if (HIST_BITS > 12 || HIST_BITS > N) begin
        $error("readout_counter: HIST_BITS must be <= 12 and <= N");
        $finish;
    end
end

// state encoding: binary, 2 bits
localparam [1:0] S_IDLE  = 2'd0,
                 S_RD    = 2'd1,
                 S_WR    = 2'd2,
                 S_CLEAR = 2'd3;

reg [1:0]    state_reg = S_IDLE,      state_next;
reg [HB-1:0] idx_reg   = {HB{1'b0}},  idx_next;
reg [HB:0]   clr_reg   = {(HB+1){1'b0}}, clr_next;
reg [CW-1:0] ns_reg    = {CW{1'b0}},  ns_next;
reg          up_inc;
reg          up_clr;
wire         up_busy;

// ------------------------------------------------------------- control
always @* begin
    state_next = state_reg;
    idx_next   = idx_reg;
    clr_next   = clr_reg;
    ns_next    = ns_reg;
    up_inc     = 1'b0;
    up_clr     = 1'b0;

    case (state_reg)
        S_IDLE: begin
            if (clear) begin
                clr_next   = {(HB+1){1'b0}};
                ns_next    = {CW{1'b0}};
                up_clr     = 1'b1;
                state_next = (HIST_BITS > 0) ? S_CLEAR : S_IDLE;
            end else if (sample && !up_busy) begin
                idx_next   = spins[HB-1:0];
                ns_next    = ns_reg + 1'b1;
                up_inc     = 1'b1;
                state_next = (HIST_BITS > 0) ? S_RD : S_IDLE;
            end
        end
        S_RD: begin                       // hist_ram output valid next cycle
            state_next = S_WR;
        end
        S_WR: begin
            state_next = S_IDLE;
        end
        S_CLEAR: begin
            clr_next = clr_reg + 1'b1;
            if (clr_reg == (1 << HB) - 1) begin
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
    idx_reg   <= idx_next;
    clr_reg   <= clr_next;
    ns_reg    <= ns_next;

    if (rst) begin
        state_reg <= S_IDLE;
        ns_reg    <= {CW{1'b0}};
    end
end

// ------------------------------------------------------------- up counters
generate
    if (UP_RAM != 0) begin : gen_up_ram
        localparam integer  LASTI = N - 1;
        localparam [IW:0]   LASTU = LASTI[IW:0];

        (* ram_style = "block" *)
        reg [CW-1:0] up_mem [(2**IW)-1:0];
        reg [CW-1:0] uq_reg   = {CW{1'b0}};
        reg [N-1:0]  snap_reg = {N{1'b0}};       // spins of the sample, LSB next
        reg [IW:0]   ui_reg   = {(IW+1){1'b0}};  // next counter to read
        reg          scan_reg = 1'b0;
        reg          clrm_reg = 1'b0;            // scan writes zeros
        reg          wv_reg   = 1'b0;            // write stage valid
        reg          pend_reg = 1'b0;            // clear requested during a scan
        reg [IW-1:0] wa_reg   = {IW{1'b0}};
        reg          wb_reg   = 1'b0;

        integer m;
        initial begin
            for (m = 0; m < 2**IW; m = m + 1) begin
                up_mem[m] = {CW{1'b0}};
            end
        end

        wire [IW-1:0] raddr = scan_reg ? ui_reg[IW-1:0] : up_sel;

        always @(posedge clk) begin
            if (wv_reg) begin
                up_mem[wa_reg] <= clrm_reg ? {CW{1'b0}} : (uq_reg + {{(CW-1){1'b0}}, wb_reg});
            end
            uq_reg <= up_mem[raddr];
        end

        always @(posedge clk) begin
            wv_reg <= scan_reg;
            wa_reg <= ui_reg[IW-1:0];
            wb_reg <= snap_reg[0];
            if (scan_reg) begin
                snap_reg <= snap_reg >> 1;
                ui_reg   <= ui_reg + 1'b1;
                if (ui_reg == LASTU) scan_reg <= 1'b0;
                if (up_clr) pend_reg <= 1'b1;
            end else if (up_clr || up_inc || pend_reg) begin
                snap_reg <= spins;
                ui_reg   <= {(IW+1){1'b0}};
                clrm_reg <= up_clr | pend_reg;
                scan_reg <= 1'b1;
                pend_reg <= 1'b0;
            end

            if (rst) begin
                scan_reg <= 1'b0;
                wv_reg   <= 1'b0;
                pend_reg <= 1'b0;
            end
        end

        assign up_busy = scan_reg | wv_reg | pend_reg;
        assign up_q    = uq_reg;
    end else begin : gen_up_ff
        reg [CW-1:0] up_reg [0:N-1];
        reg [CW-1:0] upq_reg = {CW{1'b0}};

        integer u;
        initial begin
            for (u = 0; u < N; u = u + 1) begin
                up_reg[u] = {CW{1'b0}};
            end
        end

        always @(posedge clk) begin
            upq_reg <= up_reg[up_sel];
            for (u = 0; u < N; u = u + 1) begin
                if (up_clr || rst) begin
                    up_reg[u] <= {CW{1'b0}};
                end else if (up_inc && spins[u]) begin
                    up_reg[u] <= up_reg[u] + 1'b1;
                end
            end
        end

        assign up_busy = 1'b0;
        assign up_q    = upq_reg;
    end
endgenerate

// ------------------------------------------------------------- histogram
generate
    if (HIST_BITS > 0) begin : gen_hist
        (* ram_style = "block" *)
        reg [CW-1:0] hist_mem [(2**HB)-1:0];
        reg [CW-1:0] hq_reg = {CW{1'b0}};

        integer m;
        initial begin
            for (m = 0; m < 2**HB; m = m + 1) begin
                hist_mem[m] = {CW{1'b0}};
            end
        end

        wire [HB-1:0] raddr = (state_reg == S_IDLE) ? hist_addr : idx_reg;

        always @(posedge clk) begin
            if (state_reg == S_CLEAR) begin
                hist_mem[clr_reg[HB-1:0]] <= {CW{1'b0}};
            end else if (state_reg == S_WR) begin
                hist_mem[idx_reg] <= hq_reg + 1'b1;
            end
            hq_reg <= hist_mem[raddr];
        end

        assign hist_q = hq_reg;
    end else begin : gen_no_hist
        assign hist_q = {CW{1'b0}};
    end
endgenerate

assign n_samples = ns_reg;
assign busy      = (state_reg != S_IDLE) || up_busy;

endmodule

`resetall
