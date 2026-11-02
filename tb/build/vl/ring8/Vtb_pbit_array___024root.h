// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb_pbit_array.h for the primary calling header

#ifndef VERILATED_VTB_PBIT_ARRAY___024ROOT_H_
#define VERILATED_VTB_PBIT_ARRAY___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb_pbit_array__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb_pbit_array___024root final {
  public:

    // TRIVIALLY CONSTRUCTIBLE STATE
    ::CData/*0:0*/ tb_pbit_array__DOT__clk;
    ::CData/*0:0*/ tb_pbit_array__DOT__rst;
    ::CData/*0:0*/ tb_pbit_array__DOT__run;
    ::CData/*7:0*/ tb_pbit_array__DOT__dut__DOT__w_q;
    ::CData/*7:0*/ tb_pbit_array__DOT__dut__DOT__h_q;
    ::CData/*2:0*/ tb_pbit_array__DOT__dut__DOT__o_q;
    ::CData/*2:0*/ tb_pbit_array__DOT__dut__DOT__state_reg;
    ::CData/*2:0*/ tb_pbit_array__DOT__dut__DOT__k_reg;
    ::CData/*2:0*/ tb_pbit_array__DOT__dut__DOT__node_reg;
    ::CData/*7:0*/ tb_pbit_array__DOT__dut__DOT__spins_reg;
    ::CData/*7:0*/ tb_pbit_array__DOT__dut__DOT__spins_next;
    ::CData/*0:0*/ tb_pbit_array__DOT__dut__DOT__swd_reg;
    ::CData/*0:0*/ tb_pbit_array__DOT__dut__DOT__mac_start;
    ::CData/*0:0*/ tb_pbit_array__DOT__dut__DOT__pb_start;
    ::CData/*2:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg;
    ::CData/*7:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg;
    ::CData/*2:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg;
    ::CData/*0:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bit_reg;
    ::CData/*0:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_reg;
    ::CData/*0:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_next;
    ::CData/*0:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__take_reg;
    ::CData/*0:0*/ tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__run_reg;
    ::CData/*3:0*/ tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg;
    ::CData/*0:0*/ tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__vld_reg;
    ::CData/*2:0*/ tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg;
    ::CData/*0:0*/ tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__done_reg;
    ::CData/*2:0*/ tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg;
    ::CData/*7:0*/ tb_pbit_array__DOT__sched_inst__DOT__beta_reg;
    ::CData/*0:0*/ __Vdfg_Regularize_986;
    ::CData/*0:0*/ __Vdfg_Regularize_993;
    ::CData/*0:0*/ __Vdfg_Regularize_994;
    ::CData/*0:0*/ __Vdfg_Regularize_997;
    ::CData/*0:0*/ __Vdfg_Regularize_998;
    ::CData/*0:0*/ __Vdly_Set__tb_pbit_array__DOT__rst_28;
    ::CData/*0:0*/ __Vdly_Set__tb_pbit_array__DOT__nsw_30;
    ::CData/*0:0*/ __Vdly_Set__tb_pbit_array__DOT__run_31;
    ::CData/*0:0*/ __Vdly_Set__tb_pbit_array__DOT__run_32;
    ::CData/*0:0*/ __Vtrigprevexpr___TOP__tb_pbit_array__DOT__clk__0;
    ::SData/*9:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg;
    ::SData/*15:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__gen_lut__DOT__sigmoid_lut_inst__DOT__p_reg;
    ::SData/*12:0*/ tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__acc_reg;
    ::SData/*12:0*/ tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__w_ext;
    ::IData/*31:0*/ tb_pbit_array__DOT__nsw;
    ::IData/*31:0*/ tb_pbit_array__DOT__sweeps;
    ::IData/*31:0*/ tb_pbit_array__DOT__cyc1;
    ::IData/*31:0*/ tb_pbit_array__DOT__cyc;
    ::VlWide<32>/*1023:0*/ tb_pbit_array__DOT__trace_file;
    ::IData/*31:0*/ tb_pbit_array__DOT__dut__DOT__left_reg;
    ::IData/*31:0*/ tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__state_reg;
    ::IData/*21:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg;
    ::IData/*21:0*/ tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg;
    ::IData/*31:0*/ __Vdly_Val__tb_pbit_array__DOT__nsw_29;
    ::IData/*31:0*/ __Vi;
    ::QData/*47:0*/ tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap_w;
    ::VlUnpacked<::CData/*7:0*/, 2000> tb_pbit_array__DOT__expect_mem;
    ::VlUnpacked<::CData/*7:0*/, 64> tb_pbit_array__DOT__dut__DOT__wmem;
    ::VlUnpacked<::CData/*7:0*/, 8> tb_pbit_array__DOT__dut__DOT__hmem;
    ::VlUnpacked<::CData/*2:0*/, 8> tb_pbit_array__DOT__dut__DOT__omem;
    ::VlUnpacked<::SData/*15:0*/, 1024> tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__gen_lut__DOT__sigmoid_lut_inst__DOT__rom;
    ::VlUnpacked<::QData/*63:0*/, 1> __VstlTriggered;
    ::VlUnpacked<::QData/*63:0*/, 1> __VicoTriggered;
    ::VlUnpacked<::QData/*63:0*/, 1> __VactTriggered;
    ::VlUnpacked<::QData/*63:0*/, 1> __VactTriggeredAcc;
    ::VlUnpacked<::QData/*63:0*/, 1> __VnbaTriggered;

    // NOT TRIVIALLY CONSTRUCTIBLE STATE
    union {
        struct {
            ::VlDelayScheduler __VdlySched;
            ::VlTriggerScheduler __VtrigSched_h852eb7a7__0;
        } __Vnontrivs;
    };

    // INTERNAL VARIABLES
    Vtb_pbit_array__Syms* vlSymsp;
    const char* vlNamep;

    // CONSTRUCTORS
    Vtb_pbit_array___024root(Vtb_pbit_array__Syms* symsp, const char* namep);
    ~Vtb_pbit_array___024root();
    VL_UNCOPYABLE(Vtb_pbit_array___024root);
    VL_UNMOVABLE(Vtb_pbit_array___024root);

    // INTERNAL METHODS
};


#endif  // guard
