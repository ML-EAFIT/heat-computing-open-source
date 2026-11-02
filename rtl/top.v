// -----------------------------------------------------------------------------
// File   : top.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Tang Nano 9K top level: p-bit Ising / Max-Cut sampler with a UART host
//   interface, an annealing controller and on-chip readout.
//
//   UART protocol (115200 8N1, all multi-byte arguments big-endian, all
//   multi-byte replies little-endian, every command ends with 'K'):
//     'P'                      -> N, HIST_BITS, 'K'          (ping / info)
//     'W' a_hi a_lo d          weight J[{i,j}] = d            (a = i*2^IW + j)
//     'H' i d                  bias h[i] = d
//     'O' k i                  order[k] = i
//     'T' k b                  beta table[k] = b
//     'B' mode b0 b1 dl s_hi s_lo  schedule config + restart
//     'Z'                      restart schedule
//     'X' s3 s2 s1 s0          reseed LFSR, spins <- all -1
//     'R' n3 n2 n1 n0          run n sweeps, every sweep recorded in readout
//     'S' n3 n2 n1 n0          run n sweeps, stream ceil(N/8) spin bytes per
//                              sweep (sampler pauses while bytes are sent)
//     'C'                      clear readout counters
//     'D'                      dump histogram, 2^HIST_BITS x 4 bytes
//     'U'                      dump n_samples then up[0..N-1], 4 bytes each
//     'Q'                      current spins, ceil(N/8) bytes
//   SPARSE = 1 (sparse_array.v, layout from model/export_vectors.sparse_layout):
//     'P'                      -> N[7:0], HIST_BITS | 0x80, N[15:8], P, D, LW, 'K'
//     'E' l r s j d            lane l, round r, slot s: neighbour {j, J = d}
//     'M' l r f i d            lane l, round r: node {valid = f[0], i, h = d}
//     'G' r_hi r_lo            rounds per sweep
//     'W', 'H', 'O' are accepted and ignored.
//     'Y' f q_hi q_lo          noise options (NOISE_CTL = 1): f[0] = RNG
//                              persistence with refresh prob q/65536 (tau0),
//                              f[1] = all lanes share lane 0's noise
//   TEMPER_K > 1 (dense, parallel tempering, temper_ctrl.v):
//     'L' k b                  beta of ladder rung k (rung 0 = target)
//     'R' / 'S' count rounds (K sweeps + swap phase); readout, 'Q' and the
//     LEDs show the replica on rung 0; the schedule is not used. 'X' also
//     resets the replicas and reseeds the swap lane; 'Z' also resyncs the
//     replicas (all <- rung 0, energies 0): send it after changing W / H.
//   Both builds:
//     'N' n3 n2 n1 n0          stream n raw 16-bit draws of a separate noise
//                              lane (2 bytes each, LE; reseeded by 'X' like
//                              lane 0; with RO_JITTER = 1 it carries the
//                              ring-oscillator bit) for fpga/entropy.py
//   unknown byte -> '?'.
//   Button S2: restart schedule and run DEMO_SWEEPS sweeps (stand-alone
//   anneal demo). Button S1: reset. LEDs (active low): led[4:0] = spins[4:0],
//   led[5] = busy.
//
//   Idea source: host <-> FPGA <-> sampler split of XTR-0 and the Z1
//   "host communicates h_i and J_ij" protocol [data/A Framework ...
//   2608.01612v2.pdf, p.9 Figs. 6-7]. [FPGA-APPROX] everything here.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
//   2026-10-29  jero98772  v1.1  SPARSE option (sparse_array, N <= 256)
//   2026-10-30  jero98772  v1.2  'Y' noise options, 'N' raw noise stream,
//                                TEMPER_K: parallel tempering ('L')
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module top #
(
    // number of p-bits
    parameter        N           = 16,
    // clog2(N)
    parameter        IW          = 4,
    // weight fraction bits
    parameter        WFRAC       = 4,
    // accumulator width
    parameter        AW          = 14,
    // histogram over the low HIST_BITS spins (<= 12)
    parameter        HIST_BITS   = 12,
    // 1 = PWL sigmoid
    parameter        USE_PWL     = 0,
    // clock and baud rate
    parameter        CLK_HZ      = 27000000,
    parameter        BAUD        = 115200,
    // button debounce counter width (2^18 / 27 MHz = 9.7 ms)
    parameter        DEB_BITS    = 18,
    // sweeps run by button S2
    parameter        DEMO_SWEEPS = 128,
    // power-on defaults (model + schedule)
    parameter [31:0] SEED        = 32'hACE1F00D,
    parameter        W_FILE      = "../model/vectors/maxcut16/weights.hex",
    parameter        H_FILE      = "../model/vectors/maxcut16/bias.hex",
    parameter        O_FILE      = "../model/vectors/maxcut16/order.hex",
    parameter        T_FILE      = "../model/vectors/maxcut16/beta_table.hex",
    parameter        LUT_FILE    = "../model/vectors/sigmoid_lut.hex",
    parameter [1:0]  DEF_MODE    = 2'd2,
    parameter [7:0]  DEF_BETA0   = 8'd8,
    parameter [7:0]  DEF_BETA1   = 8'd64,
    parameter [7:0]  DEF_DELTA   = 8'd0,
    parameter [15:0] DEF_SPS     = 16'd8,
    // 1 = sparse_array (P lanes, degree <= D) instead of the dense pbit_array
    parameter        SPARSE      = 0,
    parameter        P           = 4,
    parameter        D           = 4,
    parameter        DW          = 2,
    parameter        LW          = 2,
    parameter        ROUNDS      = 4,
    parameter        NODE_PREFIX = "",
    parameter        NBR_PREFIX  = "",
    // 1 = up counters in BSRAM (needed for large N)
    parameter        UP_RAM      = 0,
    // 1 = sparse_array noise options ('Y'); 1 = ring-oscillator bit in 'N'
    parameter        NOISE_CTL   = 0,
    parameter        RO_JITTER   = 0,
    // replicas for parallel tempering (dense only, 1 = off)
    parameter        TEMPER_K    = 1,
    parameter [31:0] TEMPER_LADDER = 32'h0C_14_20_30
)
(
    input  wire       clk,
    input  wire       btn_rst_n,
    input  wire       btn_run_n,
    input  wire       uart_rxd,
    output wire       uart_txd,
    output wire [5:0] led
);

localparam CPB  = CLK_HZ / BAUD;
localparam NB   = (N + 7) / 8;                 // bytes per spin word
localparam HB   = (HIST_BITS < 1) ? 1 : HIST_BITS;
localparam WW   = ((N + 31) / 32) * 32;         // send-word width (>= 32)

initial begin
    if ((SPARSE == 0 && N > 128) || N > 256) begin
        $error("top: N <= 128 dense (weight RAM must fit in BSRAM), N <= 256 sparse");
        $finish;
    end
end

// ------------------------------------------------------------ reset, buttons
reg [3:0]  por_reg = 4'd0;
reg        rst_s1  = 1'b0, rst_s2 = 1'b0;      // CDC: button -> clk
reg        run_s1  = 1'b1, run_s2 = 1'b1;      // CDC: button -> clk
reg [DEB_BITS-1:0] deb_reg = {DEB_BITS{1'b0}};
reg        run_stable = 1'b1, run_prev = 1'b1;
wire       rst = (por_reg != 4'hF) || !rst_s2;
wire       btn_run_evt = run_prev && !run_stable;  // press = falling edge

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
end

// ------------------------------------------------------------ UART
wire [7:0] rx_data;
wire       rx_valid;
wire       tx_ready;
reg  [7:0] tx_data_reg  = 8'd0, tx_data_next;
reg        tx_valid_reg = 1'b0, tx_valid_next;

uart_rx #(.CLKS_PER_BIT(CPB)) uart_rx_inst (
    .clk       (clk),
    .rst       (rst),
    .rxd       (uart_rxd),
    .out_data  (rx_data),
    .out_valid (rx_valid)
);

uart_tx #(.CLKS_PER_BIT(CPB)) uart_tx_inst (
    .clk      (clk),
    .rst      (rst),
    .in_data  (tx_data_reg),
    .in_valid (tx_valid_reg),
    .in_ready (tx_ready),
    .txd      (uart_txd)
);

// ------------------------------------------------------------ command FSM
// state encoding: binary, 5 bits
localparam [4:0] S_IDLE  = 5'd0,
                 S_ARGS  = 5'd1,
                 S_EXEC  = 5'd2,
                 S_RUN1  = 5'd3,
                 S_RUNW  = 5'd4,
                 S_STRW  = 5'd5,
                 S_STRS  = 5'd6,
                 S_CLR1  = 5'd7,
                 S_CLRW  = 5'd8,
                 S_DADDR = 5'd9,
                 S_DWAIT = 5'd10,
                 S_DLAT  = 5'd11,
                 S_UADDR = 5'd12,
                 S_UWAIT = 5'd13,
                 S_ULAT  = 5'd14,
                 S_SEND  = 5'd15,
                 S_SENDA = 5'd16,
                 S_SENDB = 5'd17,
                 S_ACK   = 5'd18,
                 S_PING2 = 5'd19,
                 S_BRST  = 5'd20,
                 S_PING3 = 5'd21,
                 S_NOISE = 5'd22;

reg [4:0]    state_reg = S_IDLE,     state_next;
reg [4:0]    ret_reg   = S_IDLE,     ret_next;       // state after S_SEND
reg [7:0]    cmd_reg   = 8'd0,       cmd_next;
reg [2:0]    nargs_reg = 3'd0,       nargs_next;
reg [2:0]    aidx_reg  = 3'd0,       aidx_next;
reg [47:0]   args_reg  = 48'd0,      args_next;      // shift register, last byte in [7:0]
reg [WW-1:0] word_reg  = {WW{1'b0}},  word_next;      // bytes being sent, LSB first
reg [5:0]    nby_reg   = 6'd0,       nby_next;
reg [31:0]   cnt_reg   = 32'd0,      cnt_next;       // generic counter
reg [31:0]   tot_reg   = 32'd0,      tot_next;
reg          silent_reg = 1'b0,      silent_next;    // no 'K' (button run)
reg          hold_reg  = 1'b0,       hold_next;
reg [1:0]    mode_reg  = DEF_MODE,   mode_next;
reg [7:0]    b0_reg    = DEF_BETA0,  b0_next;
reg [7:0]    b1_reg    = DEF_BETA1,  b1_next;
reg [7:0]    dl_reg    = DEF_DELTA,  dl_next;
reg [15:0]   sps_reg   = DEF_SPS,    sps_next;
reg [1:0]    nzf_reg   = 2'd0,       nzf_next;       // noise flags {share, tau}
reg [15:0]   nzq_reg   = 16'd0,      nzq_next;
reg          raw_en;
wire [15:0]  raw_rnd;
wire [WW-1:0] raw_w = {{(WW-16){1'b0}}, raw_rnd};

reg          w_we, h_we, o_we, t_we, sch_restart, seed_load, spin_we;
reg          e_we, m_we, g_we, l_we, pt_resync;
reg          arr_run, ro_clear, arr_stop;
reg  [31:0]  arr_nsw;
reg  [HB-1:0] hist_addr;
reg  [IW-1:0] up_sel;

wire [7:0]   beta;
wire         arr_busy, swd;
wire [N-1:0] spins;
wire [WW-1:0] spins_w;
wire [WW-1:0] ns_w, hq_w, uq_w;                       // 32-bit words widened to WW
localparam [WW-1:0] CH_K = {{(WW-8){1'b0}}, 8'h4B};  // 'K' (WW >= 32)
localparam [WW-1:0] CH_Q = {{(WW-8){1'b0}}, 8'h3F};  // '?'
localparam integer   NI   = N;
localparam integer   HBI  = HIST_BITS;
localparam [WW-1:0] N_W  = {{(WW-8){1'b0}}, NI[7:0]};
localparam [WW-1:0] HB_W = {{(WW-8){1'b0}}, (SPARSE != 0) ? 1'b1 : 1'b0, HBI[6:0]};
localparam integer   PI   = P;
localparam integer   DI   = D;
localparam integer   LWI  = LW;
// sparse ping tail: N[15:8], P, D, LW (sent LSB first)
localparam [WW-1:0] SP_W = {{(WW-32){1'b0}}, LWI[7:0], DI[7:0], PI[7:0], NI[15:8]};
wire [31:0]  hist_q, up_q, n_samples;
wire         ro_busy;

function [2:0] nargs_of;
    input [7:0] c;
    begin
        case (c)
            "W":     nargs_of = 3'd3;
            "E", "M": nargs_of = 3'd5;
            "G":     nargs_of = 3'd2;
            "H", "O", "T": nargs_of = 3'd2;
            "B":     nargs_of = 3'd6;
            "X", "R", "S", "N": nargs_of = 3'd4;
            "Y":     nargs_of = 3'd3;
            "L":     nargs_of = 3'd2;
            default: nargs_of = 3'd0;
        endcase
    end
endfunction

always @* begin
    state_next    = state_reg;
    ret_next      = ret_reg;
    cmd_next      = cmd_reg;
    nargs_next    = nargs_reg;
    aidx_next     = aidx_reg;
    args_next     = args_reg;
    word_next     = word_reg;
    nby_next      = nby_reg;
    cnt_next      = cnt_reg;
    tot_next      = tot_reg;
    silent_next   = silent_reg;
    hold_next     = hold_reg;
    mode_next     = mode_reg;
    b0_next       = b0_reg;
    b1_next       = b1_reg;
    dl_next       = dl_reg;
    sps_next      = sps_reg;
    nzf_next      = nzf_reg;
    nzq_next      = nzq_reg;
    raw_en        = 1'b0;
    tx_data_next  = tx_data_reg;
    tx_valid_next = 1'b0;
    w_we          = 1'b0;
    h_we          = 1'b0;
    o_we          = 1'b0;
    e_we          = 1'b0;
    m_we          = 1'b0;
    g_we          = 1'b0;
    l_we          = 1'b0;
    pt_resync     = 1'b0;
    t_we          = 1'b0;
    sch_restart   = 1'b0;
    seed_load     = 1'b0;
    spin_we       = 1'b0;
    arr_run       = 1'b0;
    arr_stop      = 1'b0;
    arr_nsw       = tot_reg;
    ro_clear      = 1'b0;
    hist_addr     = cnt_reg[HB-1:0];
    up_sel        = cnt_reg[IW-1:0];

    case (state_reg)
        S_IDLE: begin
            hold_next = 1'b0;
            if (rx_valid) begin
                cmd_next    = rx_data;
                nargs_next  = nargs_of(rx_data);
                aidx_next   = 3'd0;
                silent_next = 1'b0;
                state_next  = (nargs_of(rx_data) == 3'd0) ? S_EXEC : S_ARGS;
            end else if (btn_run_evt) begin
                cmd_next    = "R";
                tot_next    = DEMO_SWEEPS;
                silent_next = 1'b1;
                sch_restart = 1'b1;
                state_next  = S_RUN1;
            end
        end
        S_ARGS: begin
            if (rx_valid) begin
                args_next = {args_reg[39:0], rx_data};
                aidx_next = aidx_reg + 3'd1;
                if (aidx_reg + 3'd1 == nargs_reg) begin
                    state_next = S_EXEC;
                end
            end
        end
        S_EXEC: begin
            state_next = S_ACK;
            case (cmd_reg)
                "P": begin
                    word_next  = N_W;
                    nby_next   = 6'd1;
                    ret_next   = S_PING2;
                    state_next = S_SEND;
                end
                "W": w_we = 1'b1;
                "H": h_we = 1'b1;
                "O": o_we = 1'b1;
                "E": e_we = 1'b1;
                "M": m_we = 1'b1;
                "G": g_we = 1'b1;
                "T": t_we = 1'b1;
                "B": begin
                    mode_next   = args_reg[41:40];
                    b0_next     = args_reg[39:32];
                    b1_next     = args_reg[31:24];
                    dl_next     = args_reg[23:16];
                    sps_next    = args_reg[15:0];
                    state_next  = S_BRST;           // restart after cfg update
                end
                "Z": begin
                    sch_restart = 1'b1;
                    pt_resync   = 1'b1;
                end
                "L": l_we = 1'b1;
                "X": begin
                    seed_load = 1'b1;
                    spin_we   = 1'b1;
                end
                "R": begin
                    tot_next   = args_reg[31:0];
                    state_next = (args_reg[31:0] == 32'd0) ? S_ACK : S_RUN1;
                end
                "S": begin
                    tot_next   = args_reg[31:0];
                    cnt_next   = 32'd0;
                    hold_next  = 1'b1;
                    state_next = (args_reg[31:0] == 32'd0) ? S_ACK : S_RUN1;
                end
                "Y": begin
                    nzf_next   = args_reg[17:16];
                    nzq_next   = args_reg[15:0];
                end
                "N": begin
                    tot_next   = args_reg[31:0];
                    cnt_next   = 32'd0;
                    state_next = (args_reg[31:0] == 32'd0) ? S_ACK : S_NOISE;
                end
                "C": begin
                    ro_clear   = 1'b1;
                    state_next = S_CLR1;
                end
                "D": begin
                    cnt_next   = 32'd0;
                    state_next = (HIST_BITS > 0) ? S_DADDR : S_ACK;
                end
                "U": begin
                    word_next  = ns_w;
                    nby_next   = 6'd4;
                    cnt_next   = 32'd0;
                    ret_next   = S_UADDR;
                    state_next = S_SEND;
                end
                "Q": begin
                    word_next  = spins_w;
                    nby_next   = NB[5:0];
                    ret_next   = S_ACK;
                    state_next = S_SEND;
                end
                default: begin
                    word_next  = CH_Q;
                    nby_next   = 6'd1;
                    ret_next   = S_IDLE;
                    state_next = S_SEND;
                end
            endcase
        end
        S_BRST: begin
            sch_restart = 1'b1;
            state_next  = S_ACK;
        end
        S_PING2: begin
            word_next  = HB_W;
            nby_next   = 6'd1;
            ret_next   = (SPARSE != 0) ? S_PING3 : S_ACK;
            state_next = S_SEND;
        end
        S_PING3: begin
            word_next  = SP_W;
            nby_next   = 6'd4;
            ret_next   = S_ACK;
            state_next = S_SEND;
        end
        // ------------------------------------------------ run / stream
        S_RUN1: begin
            arr_run    = 1'b1;
            cnt_next   = 32'd0;
            state_next = (cmd_reg == "S") ? S_STRW : S_RUNW;
        end
        S_RUNW: begin
            if (!arr_busy && !ro_busy) begin
                state_next = silent_reg ? S_IDLE : S_ACK;
            end
        end
        S_STRW: begin                     // wait for a sweep, then send it
            hold_next = 1'b1;
            if (swd) begin
                word_next  = spins_w;
                nby_next   = NB[5:0];
                cnt_next   = cnt_reg + 32'd1;
                ret_next   = S_STRS;
                state_next = S_SEND;
            end
        end
        S_STRS: begin                     // release the held sampler
            if (cnt_reg == tot_reg) begin
                hold_next  = 1'b0;
                state_next = S_RUNW;
            end else begin
                hold_next  = 1'b0;
                state_next = S_STRW;
            end
        end
        S_NOISE: begin                    // send one raw draw, advance the lane
            word_next  = raw_w;
            nby_next   = 6'd2;
            raw_en     = 1'b1;
            cnt_next   = cnt_reg + 32'd1;
            ret_next   = (cnt_reg + 32'd1 == tot_reg) ? S_ACK : S_NOISE;
            state_next = S_SEND;
        end
        // ------------------------------------------------ readout
        S_CLR1: begin
            state_next = S_CLRW;
        end
        S_CLRW: begin
            if (!ro_busy) state_next = S_ACK;
        end
        S_DADDR: begin                    // hist_addr = cnt
            state_next = S_DWAIT;
        end
        S_DWAIT: begin
            state_next = S_DLAT;
        end
        S_DLAT: begin
            word_next  = hq_w;
            nby_next   = 6'd4;
            cnt_next   = cnt_reg + 32'd1;
            ret_next   = (cnt_reg == (1 << HB) - 1) ? S_ACK : S_DADDR;
            state_next = S_SEND;
        end
        S_UADDR: begin                    // up_sel = cnt
            if (cnt_reg == N) begin
                state_next = S_ACK;
            end else begin
                state_next = S_UWAIT;
            end
        end
        S_UWAIT: begin
            state_next = S_ULAT;
        end
        S_ULAT: begin
            word_next  = uq_w;
            nby_next   = 6'd4;
            cnt_next   = cnt_reg + 32'd1;
            ret_next   = S_UADDR;
            state_next = S_SEND;
        end
        // ------------------------------------------------ byte sender
        S_SEND: begin
            if (nby_reg == 6'd0) begin
                state_next = ret_reg;
            end else if (tx_ready) begin
                tx_data_next  = word_reg[7:0];
                tx_valid_next = 1'b1;
                word_next     = {8'd0, word_reg[WW-1:8]};
                nby_next      = nby_reg - 6'd1;
                state_next    = S_SENDA;
            end
        end
        S_SENDA: begin                    // tx accepts this cycle
            state_next = S_SENDB;
        end
        S_SENDB: begin                    // tx_ready now low until done
            state_next = S_SEND;
        end
        S_ACK: begin
            word_next  = CH_K;
            nby_next   = 6'd1;
            ret_next   = S_IDLE;
            state_next = S_SEND;
        end
        default: begin
            state_next = S_IDLE;
        end
    endcase
end

always @(posedge clk) begin
    state_reg    <= state_next;
    ret_reg      <= ret_next;
    cmd_reg      <= cmd_next;
    nargs_reg    <= nargs_next;
    aidx_reg     <= aidx_next;
    args_reg     <= args_next;
    word_reg     <= word_next;
    nby_reg      <= nby_next;
    cnt_reg      <= cnt_next;
    tot_reg      <= tot_next;
    silent_reg   <= silent_next;
    hold_reg     <= hold_next;
    mode_reg     <= mode_next;
    b0_reg       <= b0_next;
    b1_reg       <= b1_next;
    dl_reg       <= dl_next;
    sps_reg      <= sps_next;
    nzf_reg      <= nzf_next;
    nzq_reg      <= nzq_next;
    tx_data_reg  <= tx_data_next;
    tx_valid_reg <= tx_valid_next;

    if (rst) begin
        state_reg    <= S_IDLE;
        tx_valid_reg <= 1'b0;
        hold_reg     <= 1'b0;
        mode_reg     <= DEF_MODE;
        b0_reg       <= DEF_BETA0;
        b1_reg       <= DEF_BETA1;
        dl_reg       <= DEF_DELTA;
        sps_reg      <= DEF_SPS;
        nzf_reg      <= 2'd0;
        nzq_reg      <= 16'd0;
    end
end

// ------------------------------------------------------------ datapath
schedule_ctrl #(
    .TABLE_FILE (T_FILE)
) schedule_ctrl_inst (
    .clk        (clk),
    .rst        (rst),
    .restart    (sch_restart),
    .sweep_done (swd),
    .cfg_mode   (mode_reg),
    .cfg_beta0  (b0_reg),
    .cfg_beta1  (b1_reg),
    .cfg_delta  (dl_reg),
    .cfg_sps    (sps_reg),
    .tbl_we     (t_we),
    .tbl_addr   (args_reg[11:8]),
    .tbl_data   (args_reg[7:0]),
    .beta       (beta)
);

generate
    if (SPARSE != 0) begin : gen_sparse
        sparse_array #(
            .N           (N),
            .IW          (IW),
            .P           (P),
            .D           (D),
            .DW          (DW),
            .LW          (LW),
            .ROUNDS      (ROUNDS),
            .WB          (8),
            .WFRAC       (WFRAC),
            .AW          (AW),
            .USE_PWL     (USE_PWL),
            .SEED        (SEED),
            .NODE_PREFIX (NODE_PREFIX),
            .NBR_PREFIX  (NBR_PREFIX),
            .LUT_FILE    (LUT_FILE),
            .NOISE_CTL   (NOISE_CTL)
        ) sparse_array_inst (
            .clk        (clk),
            .rst        (rst),
            .m_we       (m_we),
            .m_lane     (args_reg[33:32]),
            .m_addr     (args_reg[LW+23:24]),
            .m_data     ({args_reg[16], args_reg[IW+7:8], args_reg[7:0]}),
            .n_we       (e_we),
            .n_lane     (args_reg[33:32]),
            .n_addr     ({args_reg[LW+23:24], args_reg[DW+15:16]}),
            .n_data     ({args_reg[IW+7:8], args_reg[7:0]}),
            .r_we       (g_we),
            .r_data     (args_reg[LW:0]),
            .spin_we    (spin_we),
            .spin_data  ({N{1'b0}}),
            .seed_load  (seed_load),
            .seed_in    (args_reg[31:0]),
            .run        (arr_run),
            .n_sweeps   (arr_nsw),
            .hold       (hold_reg),
            .stop       (arr_stop),
            .beta       (beta),
            .nz_tau     (nzf_reg[0]),
            .nz_share   (nzf_reg[1]),
            .nz_q       (nzq_reg),
            .busy       (arr_busy),
            .sweep_done (swd),
            .spins      (spins)
        );
    end else begin : gen_dense
        wire         a_spin_we, a_run, a_busy, a_swd;
        wire [N-1:0] a_spin_data, a_spins;
        wire [7:0]   a_beta;
        wire         upd_we, upd_old, upd_new;
        wire signed [AW-1:0] upd_field;

        pbit_array #(
            .N        (N),
            .IW       (IW),
            .WB       (8),
            .WFRAC    (WFRAC),
            .AW       (AW),
            .USE_PWL  (USE_PWL),
            .SEED     (SEED),
            .W_FILE   (W_FILE),
            .H_FILE   (H_FILE),
            .O_FILE   (O_FILE),
            .LUT_FILE (LUT_FILE)
        ) pbit_array_inst (
            .clk        (clk),
            .rst        (rst),
            .w_we       (w_we),
            .w_addr     (args_reg[2*IW+7:8]),
            .w_data     (args_reg[7:0]),
            .h_we       (h_we),
            .h_addr     (args_reg[IW+7:8]),
            .h_data     (args_reg[7:0]),
            .o_we       (o_we),
            .o_addr     (args_reg[IW+7:8]),
            .o_data     (args_reg[IW-1:0]),
            .spin_we    (a_spin_we),
            .spin_data  (a_spin_data),
            .seed_load  (seed_load),
            .seed_in    (args_reg[31:0]),
            .run        (a_run),
            .n_sweeps   ((TEMPER_K > 1) ? 32'd1 : arr_nsw),
            .hold       ((TEMPER_K > 1) ? 1'b0 : hold_reg),
            .stop       (arr_stop),
            .beta       (a_beta),
            .busy       (a_busy),
            .sweep_done (a_swd),
            .spins      (a_spins),
            .upd_we     (upd_we),
            .upd_old    (upd_old),
            .upd_new    (upd_new),
            .upd_field  (upd_field)
        );

        if (TEMPER_K > 1) begin : gen_pt
            temper_ctrl #(
                .N        (N),
                .AW       (AW),
                .WFRAC    (WFRAC),
                .K        (TEMPER_K),
                .LADDER   (TEMPER_LADDER),
                .SEED     (SEED),
                .LUT_FILE (LUT_FILE)
            ) temper_ctrl_inst (
                .clk         (clk),
                .rst         (rst),
                .ladder_we   (l_we),
                .ladder_addr (args_reg[9:8]),
                .ladder_data (args_reg[7:0]),
                .reset_all   (seed_load),
                .seed_load   (seed_load),
                .seed_in     (args_reg[31:0]),
                .resync      (pt_resync),
                .run         (arr_run),
                .n_rounds    (arr_nsw),
                .hold        (hold_reg),
                .busy        (arr_busy),
                .round_done  (swd),
                .spins_out   (spins),
                .a_spin_we   (a_spin_we),
                .a_spin_data (a_spin_data),
                .a_run       (a_run),
                .a_beta      (a_beta),
                .a_busy      (a_busy),
                .a_swd       (a_swd),
                .a_spins     (a_spins),
                .upd_we      (upd_we),
                .upd_old     (upd_old),
                .upd_new     (upd_new),
                .upd_field   (upd_field)
            );
        end else begin : gen_plain
            assign a_spin_we   = spin_we;
            assign a_spin_data = {N{1'b0}};
            assign a_run       = arr_run;
            assign a_beta      = beta;
            assign arr_busy    = a_busy;
            assign swd         = a_swd;
            assign spins       = a_spins;
        end
    end
endgenerate

// raw noise lane for 'N' (entropy study): lane-0 polynomial and seeding
noise_bank #(
    .LANES     (1),
    .SEED      (SEED),
    .RO_JITTER (RO_JITTER)
) raw_noise_inst (
    .clk       (clk),
    .rst       (rst),
    .en        (raw_en),
    .seed_load (seed_load),
    .seed_in   (args_reg[31:0]),
    .rnd       (raw_rnd)
);

readout_counter #(
    .N         (N),
    .IW        (IW),
    .HIST_BITS (HIST_BITS),
    .CW        (32),
    .UP_RAM    (UP_RAM)
) readout_counter_inst (
    .clk       (clk),
    .rst       (rst),
    .clear     (ro_clear),
    .sample    (swd),
    .spins     (spins),
    .hist_addr (hist_addr),
    .hist_q    (hist_q),
    .up_sel    (up_sel),
    .up_q      (up_q),
    .n_samples (n_samples),
    .busy      (ro_busy)
);

generate
    if (WW > 32) begin : gen_wide
        assign ns_w = {{(WW-32){1'b0}}, n_samples};
        assign hq_w = {{(WW-32){1'b0}}, hist_q};
        assign uq_w = {{(WW-32){1'b0}}, up_q};
    end else begin : gen_narrow
        assign ns_w = n_samples;
        assign hq_w = hist_q;
        assign uq_w = up_q;
    end
    if (N < WW) begin : gen_pad
        assign spins_w = {{(WW-N){1'b0}}, spins};
    end else begin : gen_nopad
        assign spins_w = spins;
    end
endgenerate

assign led = ~{arr_busy, spins[4:0]};

endmodule

`resetall
