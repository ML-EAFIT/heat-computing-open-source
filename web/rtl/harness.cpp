// -----------------------------------------------------------------------------
// File   : harness.cpp
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-31
// -----------------------------------------------------------------------------
// Description:
//   Verilator main that turns rtl/top.v into a byte pipe: bytes on stdin are
//   sent to uart_rxd (8N1, CPB clocks per bit), bytes decoded from uart_txd go
//   to stdout. The web server (web/server.py) talks to it exactly as it talks
//   to the board's USB-UART, so the page can drive the real Verilog.
//   Build: web/build_rtl.py <chip> (passes CLK_HZ / BAUD so that CPB = 8).
//   Buttons stay released. The clock keeps running while the array is busy
//   (led[5]) or a byte is in flight; when everything has been idle for
//   IDLE_CYCLES the harness sleeps in select() until input arrives.
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdint>
#include <deque>
#include <sys/select.h>
#include <unistd.h>
#include "Vtop.h"
#include "verilated.h"

#ifndef CPB
#define CPB 8
#endif
static const uint64_t IDLE_CYCLES = 4096;

static Vtop *top;
static uint64_t cycles = 0;

static void tick() {
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
    cycles++;
}

// host -> chip transmitter state
static std::deque<uint8_t> txq;
static int tx_bit = -1;        // -1 idle, 0 start, 1..8 data, 9 stop
static int tx_cnt = 0;
static uint8_t tx_byte = 0;

static void tx_step() {
    if (tx_bit < 0) {
        if (txq.empty()) { top->uart_rxd = 1; return; }
        tx_byte = txq.front();
        txq.pop_front();
        tx_bit = 0;
        tx_cnt = 0;
    }
    if (tx_bit == 0) top->uart_rxd = 0;
    else if (tx_bit <= 8) top->uart_rxd = (tx_byte >> (tx_bit - 1)) & 1;
    else top->uart_rxd = 1;
    if (++tx_cnt >= (tx_bit == 9 ? CPB + 2 : CPB)) {
        tx_cnt = 0;
        if (++tx_bit > 9) tx_bit = -1;
    }
}

// chip -> host receiver state
static int rx_bit = -1;
static int rx_cnt = 0;
static uint8_t rx_byte = 0;
static uint8_t outbuf[4096];
static size_t outlen = 0;

static void flush_out() {
    size_t off = 0;
    while (off < outlen) {
        ssize_t w = write(1, outbuf + off, outlen - off);
        if (w <= 0) _exit(0);
        off += (size_t)w;
    }
    outlen = 0;
}

static void rx_step() {
    int d = top->uart_txd;
    if (rx_bit < 0) {
        if (!d) { rx_bit = 0; rx_cnt = CPB / 2; }
        return;
    }
    if (--rx_cnt > 0) return;
    rx_cnt = CPB;
    if (rx_bit == 0) {
        if (d) { rx_bit = -1; return; }        // glitch, not a start bit
        rx_bit = 1;
        rx_byte = 0;
    } else if (rx_bit <= 8) {
        rx_byte |= (uint8_t)(d << (rx_bit - 1));
        rx_bit++;
    } else {
        outbuf[outlen++] = rx_byte;
        if (outlen == sizeof(outbuf)) flush_out();
        rx_bit = -1;
    }
}

static bool read_input(bool block) {
    fd_set fs;
    FD_ZERO(&fs);
    FD_SET(0, &fs);
    timeval tv = {0, block ? 20000 : 0};
    if (select(1, &fs, nullptr, nullptr, &tv) <= 0) return true;
    uint8_t buf[4096];
    ssize_t n = read(0, buf, sizeof(buf));
    if (n <= 0) return false;                  // stdin closed: exit
    for (ssize_t k = 0; k < n; k++) txq.push_back(buf[k]);
    return true;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    top = new Vtop;
    top->btn_rst_n = 1;
    top->btn_run_n = 1;
    top->uart_rxd = 1;
    for (int k = 0; k < 64; k++) tick();
    uint64_t idle = 0;
    while (!Verilated::gotFinish()) {
        for (int k = 0; k < 256; k++) {
            tx_step();
            tick();
            rx_step();
        }
        bool busy = !((top->led >> 5) & 1);     // led active low
        bool quiet = tx_bit < 0 && txq.empty() && rx_bit < 0 && !busy;
        idle = quiet ? idle + 256 : 0;
        if (outlen) flush_out();
        if (!read_input(idle >= IDLE_CYCLES)) break;
    }
    flush_out();
    delete top;
    return 0;
}
