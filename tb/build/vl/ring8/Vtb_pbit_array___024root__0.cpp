// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_pbit_array.h for the primary calling header

#include "Vtb_pbit_array__pch.h"

VL_ATTR_COLD void Vtb_pbit_array___024root___eval_initial__TOP(Vtb_pbit_array___024root* vlSelf);
VlCoroutine Vtb_pbit_array___024root___eval_initial__TOP__Vtiming__0(Vtb_pbit_array___024root* vlSelf);
VlCoroutine Vtb_pbit_array___024root___eval_initial__TOP__Vtiming__1(Vtb_pbit_array___024root* vlSelf);

void Vtb_pbit_array___024root___eval_initial(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_initial\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtb_pbit_array___024root___eval_initial__TOP(vlSelf);
    Vtb_pbit_array___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtb_pbit_array___024root___eval_initial__TOP__Vtiming__1(vlSelf);
}

void Vtb_pbit_array___024root___eval_sample(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_sample\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_pbit_array___024root___dump_triggers__ico(const ::VlUnpacked<::QData/*63:0*/, 1> &triggers, const ::std::string &tag);
#endif  // VL_DEBUG

bool Vtb_pbit_array___024root___eval_ico(Vtb_pbit_array___024root* vlSelf, ::CData/*0:0*/ firstIteration) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_ico\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VicoTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VicoTriggered[0U]) 
                                     | (::IData)((::IData)(firstIteration)));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_pbit_array___024root___dump_triggers__ico(vlSelfRef.__VicoTriggered, "ico"s);
    }
#endif
    return (0U);
}

void Vtb_pbit_array___024root___timing_ready(Vtb_pbit_array___024root* vlSelf);
void Vtb_pbit_array___024root___trigger_orInto__act_vec_vec(::VlUnpacked<::QData/*63:0*/, 1> &out, const ::VlUnpacked<::QData/*63:0*/, 1> &in);
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_pbit_array___024root___dump_triggers__act(const ::VlUnpacked<::QData/*63:0*/, 1> &triggers, const ::std::string &tag);
#endif  // VL_DEBUG
bool Vtb_pbit_array___024root___trigger_anySet__act(const ::VlUnpacked<::QData/*63:0*/, 1> &in);
void Vtb_pbit_array___024root___timing_resume(Vtb_pbit_array___024root* vlSelf);

bool Vtb_pbit_array___024root___eval_act(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_act\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    ::CData/*0:0*/ __VactExecute;
    // Body
    {
        // Inlined CFunc: _eval_triggers_vec__act
        vlSelfRef.__VactTriggered[0U] = (::QData)((::IData)(
                                                            ((vlSelfRef.__Vnontrivs.__VdlySched.awaitingCurrentTime() 
                                                              << 1U) 
                                                             | ((::IData)(vlSelfRef.tb_pbit_array__DOT__clk) 
                                                                & (~ (::IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_pbit_array__DOT__clk__0))))));
        vlSelfRef.__Vtrigprevexpr___TOP__tb_pbit_array__DOT__clk__0 
            = vlSelfRef.tb_pbit_array__DOT__clk;
    }
    Vtb_pbit_array___024root___timing_ready(vlSelf);
    Vtb_pbit_array___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VactTriggered, vlSelfRef.__VactTriggeredAcc);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_pbit_array___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
    Vtb_pbit_array___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    __VactExecute = Vtb_pbit_array___024root___trigger_anySet__act(vlSelfRef.__VactTriggered);
    if (__VactExecute) {
        vlSelfRef.__VactTriggeredAcc.fill(0ULL);
        Vtb_pbit_array___024root___timing_resume(vlSelf);
    }
    return (__VactExecute);
}

bool Vtb_pbit_array___024root___eval_inact(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_inact\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    ::CData/*0:0*/ __VinactExecute;
    // Body
    __VinactExecute = vlSelfRef.__Vnontrivs.__VdlySched.awaitingZeroDelay();
    if (__VinactExecute) {
        VL_FATAL_MT("tb_pbit_array.v", 23, "", "ZERODLY: Design Verilated with '--no-sched-zero-delay', but #0 delay executed at runtime");
    }
    return (__VinactExecute);
}

void Vtb_pbit_array___024root___nba_sequent__TOP__0(Vtb_pbit_array___024root* vlSelf);
void Vtb_pbit_array___024root___trigger_clear__act(::VlUnpacked<::QData/*63:0*/, 1> &out);

bool Vtb_pbit_array___024root___eval_nba(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_nba\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    ::CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vtb_pbit_array___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        {
            // Inlined CFunc: _eval_body__nba
            if ((((1ULL & vlSelfRef.__VnbaTriggered[0U])) != 0ULL)) {
                Vtb_pbit_array___024root___nba_sequent__TOP__0(vlSelf);
            }
            if ((((3ULL & vlSelfRef.__VnbaTriggered[0U])) != 0ULL)) {
                {
                    // Inlined CFunc: _nba_sequent__TOP__1
                    if (vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__run_31) {
                        vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__run_31 = 0U;
                        vlSelfRef.tb_pbit_array__DOT__run = 1U;
                    }
                    if (vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__run_32) {
                        vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__run_32 = 0U;
                        vlSelfRef.tb_pbit_array__DOT__run = 0U;
                    }
                    if (vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__nsw_30) {
                        vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__nsw_30 = 0U;
                        vlSelfRef.tb_pbit_array__DOT__nsw 
                            = vlSelfRef.__Vdly_Val__tb_pbit_array__DOT__nsw_29;
                    }
                    if (vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__rst_28) {
                        vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__rst_28 = 0U;
                        vlSelfRef.tb_pbit_array__DOT__rst = 0U;
                    }
                    vlSelfRef.__Vdfg_Regularize_986 
                        = ((::IData)(vlSelfRef.tb_pbit_array__DOT__run) 
                           & (0U != vlSelfRef.tb_pbit_array__DOT__nsw));
                }
            }
        }
        Vtb_pbit_array___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

bool Vtb_pbit_array___024root___eval_obs(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_obs\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    return (0U);
}

bool Vtb_pbit_array___024root___eval_react(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_react\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    return (0U);
}

void Vtb_pbit_array___024root___eval_postponed(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_postponed\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

void Vtb_pbit_array___024root____VbeforeTrig_h852eb7a7__0(Vtb_pbit_array___024root* vlSelf, const char* __VeventDescription);

VlCoroutine Vtb_pbit_array___024root___eval_initial__TOP__Vtiming__0(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    ::IData/*31:0*/ tb_pbit_array__DOT____VlemExpr_1;
    ::IData/*31:0*/ tb_pbit_array__DOT____VlemExpr_0;
    ::IData/*31:0*/ tb_pbit_array__DOT__t;
    tb_pbit_array__DOT__t = 0;
    ::IData/*31:0*/ tb_pbit_array__DOT__errors;
    tb_pbit_array__DOT__errors = 0;
    ::IData/*31:0*/ tb_pbit_array__DOT__fd;
    tb_pbit_array__DOT__fd = 0;
    ::IData/*31:0*/ tb_pbit_array__DOT__unnamedblk1_1__DOT____Vrepeat0;
    tb_pbit_array__DOT__unnamedblk1_1__DOT____Vrepeat0 = 0;
    // Body
    VL_READMEM_N(true, 8, 2000, 0, "../model/vectors/ring8/trace.hex"s
                 ,  &(vlSelfRef.tb_pbit_array__DOT__expect_mem)
                 , 0, ~0ULL);
    tb_pbit_array__DOT____VlemExpr_0 = VL_VALUEPLUSARGS_INI(32, "sweeps=%d"s, 
                                                            vlSelfRef.tb_pbit_array__DOT__sweeps);
    if ((1U & (~ (0U != tb_pbit_array__DOT____VlemExpr_0)))) {
        vlSelfRef.tb_pbit_array__DOT__sweeps = 0x000007d0U;
    }
    tb_pbit_array__DOT__fd = 0U;
    tb_pbit_array__DOT____VlemExpr_1 = VL_VALUEPLUSARGS_INW(1024, "trace=%s"s, 
                                                            vlSelfRef.tb_pbit_array__DOT__trace_file);
    if (VL_UNLIKELY(((0U != tb_pbit_array__DOT____VlemExpr_1)))) {
        tb_pbit_array__DOT__fd = VL_FOPEN_NN(VL_CVT_PACK_STR_NW(32, vlSelfRef.tb_pbit_array__DOT__trace_file)
                                             , "w"s);
        ;
    }
    tb_pbit_array__DOT__errors = 0U;
    tb_pbit_array__DOT__t = 0U;
    vlSelfRef.tb_pbit_array__DOT__cyc = 0U;
    tb_pbit_array__DOT__unnamedblk1_1__DOT____Vrepeat0 = 3U;
    while (VL_LTS_III(32, 0U, tb_pbit_array__DOT__unnamedblk1_1__DOT____Vrepeat0)) {
        Vtb_pbit_array___024root____VbeforeTrig_h852eb7a7__0(vlSelf, 
                                                             "@(posedge tb_pbit_array.clk)");
        co_await vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.trigger(0U, 
                                                                         nullptr, 
                                                                         "@(posedge tb_pbit_array.clk)", 
                                                                         "tb_pbit_array.v", 
                                                                         90);
        tb_pbit_array__DOT__unnamedblk1_1__DOT____Vrepeat0 
            = (tb_pbit_array__DOT__unnamedblk1_1__DOT____Vrepeat0 
               - (::IData)(1U));
    }
    vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__rst_28 = 1U;
    Vtb_pbit_array___024root____VbeforeTrig_h852eb7a7__0(vlSelf, 
                                                         "@(posedge tb_pbit_array.clk)");
    co_await vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.trigger(0U, 
                                                                     nullptr, 
                                                                     "@(posedge tb_pbit_array.clk)", 
                                                                     "tb_pbit_array.v", 
                                                                     92);
    vlSelfRef.__Vdly_Val__tb_pbit_array__DOT__nsw_29 
        = vlSelfRef.tb_pbit_array__DOT__sweeps;
    vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__nsw_30 = 1U;
    vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__run_31 = 1U;
    Vtb_pbit_array___024root____VbeforeTrig_h852eb7a7__0(vlSelf, 
                                                         "@(posedge tb_pbit_array.clk)");
    co_await vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.trigger(0U, 
                                                                     nullptr, 
                                                                     "@(posedge tb_pbit_array.clk)", 
                                                                     "tb_pbit_array.v", 
                                                                     95);
    vlSelfRef.__Vdly_Set__tb_pbit_array__DOT__run_32 = 1U;
    while (VL_LTS_III(32, tb_pbit_array__DOT__t, vlSelfRef.tb_pbit_array__DOT__sweeps)) {
        Vtb_pbit_array___024root____VbeforeTrig_h852eb7a7__0(vlSelf, 
                                                             "@(posedge tb_pbit_array.clk)");
        co_await vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.trigger(0U, 
                                                                         nullptr, 
                                                                         "@(posedge tb_pbit_array.clk)", 
                                                                         "tb_pbit_array.v", 
                                                                         99);
        co_await vlSelfRef.__Vnontrivs.__VdlySched.delay(0x00000000000003e8ULL, 
                                                         nullptr, 
                                                         "tb_pbit_array.v", 
                                                         100);
        if (vlSelfRef.tb_pbit_array__DOT__dut__DOT__swd_reg) {
            if ((VL_GTS_III(32, 0x000007d0U, tb_pbit_array__DOT__t) 
                 & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_reg) 
                    != ((0x07cfU >= (0x000007ffU & tb_pbit_array__DOT__t))
                         ? vlSelfRef.tb_pbit_array__DOT__expect_mem
                        [(0x000007ffU & tb_pbit_array__DOT__t)]
                         : 0U)))) {
                tb_pbit_array__DOT__errors = ((::IData)(1U) 
                                              + tb_pbit_array__DOT__errors);
                if (VL_UNLIKELY((VL_GTS_III(32, 5U, tb_pbit_array__DOT__errors)))) {
                    VL_WRITEF_NX("  sweep %0d: rtl %h expected %h\n",3
                                 , '~',32,tb_pbit_array__DOT__t
                                 , '#',8,(::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_reg)
                                 , '#',8,((0x07cfU 
                                           >= (0x000007ffU 
                                               & tb_pbit_array__DOT__t))
                                           ? vlSelfRef.tb_pbit_array__DOT__expect_mem
                                          [(0x000007ffU 
                                            & tb_pbit_array__DOT__t)]
                                           : 0U));
                }
            }
            if (VL_UNLIKELY(((0U != tb_pbit_array__DOT__fd)))) {
                VL_FWRITEF_NX(tb_pbit_array__DOT__fd,"%h\n",1
                              , '#',8,vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_reg);
            }
            if ((0U == tb_pbit_array__DOT__t)) {
                vlSelfRef.tb_pbit_array__DOT__cyc1 
                    = vlSelfRef.tb_pbit_array__DOT__cyc;
            }
            tb_pbit_array__DOT__t = ((::IData)(1U) 
                                     + tb_pbit_array__DOT__t);
        }
    }
    VL_WRITEF_NX("tb_pbit_array[../model/vectors/ring8/]: N=8, %0d sweeps, %0d cycles/sweep, bit-exact over first 2000 sweeps: %0d mismatches\n",3
                 , '~',32,vlSelfRef.tb_pbit_array__DOT__sweeps
                 , '~',32,VL_DIVS_III(32, (vlSelfRef.tb_pbit_array__DOT__cyc 
                                           - vlSelfRef.tb_pbit_array__DOT__cyc1), 
                                      (VL_LTS_III(32, 1U, vlSelfRef.tb_pbit_array__DOT__sweeps)
                                        ? (vlSelfRef.tb_pbit_array__DOT__sweeps 
                                           - (::IData)(1U))
                                        : 1U)), '~',32,tb_pbit_array__DOT__errors);
    if (VL_UNLIKELY(((0U != tb_pbit_array__DOT__fd)))) {
        VL_FCLOSE_I(tb_pbit_array__DOT__fd); }
    if ((0U == tb_pbit_array__DOT__errors)) {
        VL_WRITEF_NX("tb_pbit_array: PASS\n",0);
    } else {
        VL_WRITEF_NX("tb_pbit_array: FAIL\n",0);
    }
    VL_FINISH_MT("tb_pbit_array.v", 116, "");
    co_return;
}

VlCoroutine Vtb_pbit_array___024root___eval_initial__TOP__Vtiming__1(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    while (VL_LIKELY(!vlSymsp->_vm_contextp__->gotFinish())) {
        co_await vlSelfRef.__Vnontrivs.__VdlySched.delay(0x0000000000001388ULL, 
                                                         nullptr, 
                                                         "tb_pbit_array.v", 
                                                         57);
        vlSelfRef.tb_pbit_array__DOT__clk = (1U & (~ (::IData)(vlSelfRef.tb_pbit_array__DOT__clk)));
    }
    co_return;
}

bool Vtb_pbit_array___024root___trigger_anySet__ico(const ::VlUnpacked<::QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___trigger_anySet__ico\n"); );
    // Locals
    ::IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        if (in[n]) {
            return (1U);
        }
        n = ((::IData)(1U) + n);
    } while ((1U > n));
    return (0U);
}

bool Vtb_pbit_array___024root___trigger_anySet__act(const ::VlUnpacked<::QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___trigger_anySet__act\n"); );
    // Locals
    ::IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        if (in[n]) {
            return (1U);
        }
        n = ((::IData)(1U) + n);
    } while ((1U > n));
    return (0U);
}

void Vtb_pbit_array___024root___nba_sequent__TOP__0(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___nba_sequent__TOP__0\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    ::IData/*31:0*/ tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t;
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t = 0;
    ::SData/*15:0*/ tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r;
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r = 0;
    ::QData/*47:0*/ __Vfunc_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vfuncout_0;
    __Vfunc_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vfuncout_0 = 0;
    ::IData/*31:0*/ __Vfunc_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__s_1;
    __Vfunc_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__s_1 = 0;
    ::CData/*2:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__state_reg_3;
    __Vdly__tb_pbit_array__DOT__dut__DOT__state_reg_3 = 0;
    ::CData/*2:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__k_reg_4;
    __Vdly__tb_pbit_array__DOT__dut__DOT__k_reg_4 = 0;
    ::IData/*31:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__left_reg_6;
    __Vdly__tb_pbit_array__DOT__dut__DOT__left_reg_6 = 0;
    ::CData/*2:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg_10;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg_10 = 0;
    ::IData/*21:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg_12;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg_12 = 0;
    ::CData/*7:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg_13;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg_13 = 0;
    ::CData/*2:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg_14;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg_14 = 0;
    ::SData/*9:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg_15;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg_15 = 0;
    ::CData/*3:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg_21;
    __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg_21 = 0;
    ::CData/*2:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg_23;
    __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg_23 = 0;
    ::CData/*2:0*/ __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg_25;
    __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg_25 = 0;
    ::CData/*7:0*/ __Vdly__tb_pbit_array__DOT__sched_inst__DOT__beta_reg_27;
    __Vdly__tb_pbit_array__DOT__sched_inst__DOT__beta_reg_27 = 0;
    // Body
    __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg_25 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg_15 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg;
    __Vdly__tb_pbit_array__DOT__sched_inst__DOT__beta_reg_27 
        = vlSelfRef.tb_pbit_array__DOT__sched_inst__DOT__beta_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg_14 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg_12 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg_13 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__k_reg_4 = vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg_10 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__left_reg_6 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__left_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg_21 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg_23 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__state_reg_3 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg;
    vlSelfRef.tb_pbit_array__DOT__cyc = ((::IData)(1U) 
                                         + vlSelfRef.tb_pbit_array__DOT__cyc);
    if ((4U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))) {
        __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg_15 
            = (0x000003ffU & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg));
        __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg_14 
            = (7U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg));
        __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg_12 
            = (0x003fffffU & vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg);
    } else if ((2U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))) {
        __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg_15 
            = (0x000003ffU & ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))
                               ? (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg)
                               : (VL_LTS_III(32, 0x000001ffU, 
                                             VL_EXTENDS_II(32,22, 
                                                           (0x003fffffU 
                                                            & VL_SHIFTRS_III(22,32,32, 
                                                                             ((::IData)(2U) 
                                                                              + 
                                                                              VL_EXTENDS_II(32,22, vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg)), 2U))))
                                   ? 0x01ffU : (VL_GTS_III(32, 0xfffffe00U, 
                                                           VL_EXTENDS_II(32,22, 
                                                                         (0x003fffffU 
                                                                          & VL_SHIFTRS_III(22,32,32, 
                                                                                ((::IData)(2U) 
                                                                                + 
                                                                                VL_EXTENDS_II(32,22, vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg)), 2U))))
                                                 ? 0x0200U
                                                 : 
                                                VL_SHIFTRS_III(10,32,32, 
                                                               ((::IData)(2U) 
                                                                + 
                                                                VL_EXTENDS_II(32,22, vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg)), 2U)))));
        __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg_14 
            = (7U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg));
        __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg_12 
            = (0x003fffffU & vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg);
    } else {
        __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg_15 
            = (0x000003ffU & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg));
        if ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))) {
            __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg_14 
                = (7U & ((::IData)(1U) + (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg)));
            __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg_12 
                = (0x003fffffU & (vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg 
                                  << 1U));
        } else {
            __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg_14 
                = (7U & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg) 
                         & (- (::IData)((::IData)(vlSelfRef.__Vdfg_Regularize_997)))));
            __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg_12 
                = (0x003fffffU & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pb_start)
                                   ? ((0x003fe000U 
                                       & ((- (::IData)(
                                                       (1U 
                                                        & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__acc_reg) 
                                                           >> 0x0000000cU)))) 
                                          << 0x0000000dU)) 
                                      | (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__acc_reg))
                                   : vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg));
        }
    }
    if (vlSelfRef.tb_pbit_array__DOT__dut__DOT__mac_start) {
        __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg_25 
            = vlSelfRef.tb_pbit_array__DOT__dut__DOT__node_reg;
        __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg_23 
            = (7U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg));
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__acc_reg 
            = (0x00001fffU & ((0x00001f00U & ((- (::IData)(
                                                           (1U 
                                                            & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__h_q) 
                                                               >> 7U)))) 
                                              << 8U)) 
                              | (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__h_q)));
    } else {
        __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg_25 
            = vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg;
        __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg_23 
            = (7U & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__run_reg)
                      ? ((8U > (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg))
                          ? (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg)
                          : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg))
                      : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg)));
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__acc_reg 
            = (0x00001fffU & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__vld_reg)
                               ? ((1U & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_reg) 
                                         >> (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg)))
                                   ? ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__acc_reg) 
                                      + (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__w_ext))
                                   : ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__acc_reg) 
                                      - (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__w_ext)))
                               : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__acc_reg)));
    }
    if ((4U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))) {
        __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg_13 
            = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg 
            = (0x003fffffU & vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg);
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bit_reg 
            = ((2U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))
                ? (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bit_reg)
                : ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))
                    ? (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bit_reg)
                    : ((0x0000ffffU & (::IData)((vlSelfRef.tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap_w 
                                                 >> 0x00000020U))) 
                       < (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__gen_lut__DOT__sigmoid_lut_inst__DOT__p_reg))));
    } else {
        if ((2U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))) {
            __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg_13 
                = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg;
            vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg 
                = (0x003fffffU & vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg);
        } else if ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))) {
            __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg_13 
                = ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg) 
                   >> 1U);
            vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg 
                = (0x003fffffU & ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg))
                                   ? (vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg 
                                      + vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg)
                                   : vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg));
        } else {
            __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg_13 
                = ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pb_start)
                    ? (::IData)(vlSelfRef.tb_pbit_array__DOT__sched_inst__DOT__beta_reg)
                    : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg));
            vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg 
                = (0x003fffffU & (vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg 
                                  & (- (::IData)((::IData)(vlSelfRef.__Vdfg_Regularize_997)))));
        }
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bit_reg 
            = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bit_reg;
    }
    __Vdly__tb_pbit_array__DOT__sched_inst__DOT__beta_reg_27 
        = vlSelfRef.tb_pbit_array__DOT__sched_inst__DOT__beta_reg;
    __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg_10 
        = (((2U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))
             ? ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))
                 ? 4U : 3U) : ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))
                                ? ((7U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg))
                                    ? 2U : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg))
                                : ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pb_start)
                                    ? 1U : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg)))) 
           & (- (::IData)((::IData)(vlSelfRef.__Vdfg_Regularize_994))));
    __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg_21 
        = (0x0000000fU & (((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__run_reg)
                            ? ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg) 
                               + (8U > (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg)))
                            : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg)) 
                          & (- (::IData)((::IData)(vlSelfRef.__Vdfg_Regularize_998)))));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__state_reg 
        = ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__take_reg)
            ? (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap_w)
            : vlSelfRef.tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__state_reg);
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__w_q = vlSelfRef.tb_pbit_array__DOT__dut__DOT__wmem
        [(((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg) 
           << 3U) | (7U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg)))];
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__swd_reg 
        = ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_reg) 
           & ((7U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg)) 
              & (5U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))));
    if (vlSelfRef.tb_pbit_array__DOT__rst) {
        __Vdly__tb_pbit_array__DOT__sched_inst__DOT__beta_reg_27 = 0x18U;
    }
    vlSelfRef.tb_pbit_array__DOT__sched_inst__DOT__beta_reg 
        = __Vdly__tb_pbit_array__DOT__sched_inst__DOT__beta_reg_27;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg 
        = __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg_13;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg 
        = __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg_12;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg 
        = __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg_14;
    __Vdly__tb_pbit_array__DOT__dut__DOT__k_reg_4 = 
        (7U & ((4U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                ? ((2U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                    ? ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                        ? (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg)
                        : (((::IData)(1U) + (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg)) 
                           & (- (::IData)((7U != (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg))))))
                    : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg))
                : ((2U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                    ? (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg)
                    : ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                        ? (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg)
                        : ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg) 
                           & (- (::IData)((1U & (~ (::IData)(vlSelfRef.__Vdfg_Regularize_986))))))))));
    if (vlSelfRef.tb_pbit_array__DOT__rst) {
        __Vdly__tb_pbit_array__DOT__dut__DOT__k_reg_4 = 0U;
        __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg_10 = 0U;
    }
    if ((4U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))) {
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__node_reg 
            = vlSelfRef.tb_pbit_array__DOT__dut__DOT__node_reg;
        __Vdly__tb_pbit_array__DOT__dut__DOT__left_reg_6 
            = ((2U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                ? ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                    ? vlSelfRef.tb_pbit_array__DOT__dut__DOT__left_reg
                    : (vlSelfRef.tb_pbit_array__DOT__dut__DOT__left_reg 
                       - (7U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg))))
                : vlSelfRef.tb_pbit_array__DOT__dut__DOT__left_reg);
    } else if ((2U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))) {
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__node_reg 
            = ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                ? (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__node_reg)
                : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__o_q));
        __Vdly__tb_pbit_array__DOT__dut__DOT__left_reg_6 
            = vlSelfRef.tb_pbit_array__DOT__dut__DOT__left_reg;
    } else {
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__node_reg 
            = vlSelfRef.tb_pbit_array__DOT__dut__DOT__node_reg;
        __Vdly__tb_pbit_array__DOT__dut__DOT__left_reg_6 
            = ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                ? vlSelfRef.tb_pbit_array__DOT__dut__DOT__left_reg
                : ((::IData)(vlSelfRef.__Vdfg_Regularize_986)
                    ? vlSelfRef.tb_pbit_array__DOT__nsw
                    : vlSelfRef.tb_pbit_array__DOT__dut__DOT__left_reg));
    }
    if (vlSelfRef.tb_pbit_array__DOT__rst) {
        __Vdly__tb_pbit_array__DOT__dut__DOT__left_reg_6 = 0U;
    }
    __Vdly__tb_pbit_array__DOT__dut__DOT__state_reg_3 
        = ((4U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
            ? ((2U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                ? ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                    ? 1U : ((7U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg))
                             ? (1U & (- (::IData)((1U 
                                                   != vlSelfRef.tb_pbit_array__DOT__dut__DOT__left_reg))))
                             : 1U)) : ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                                        ? ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_reg)
                                            ? 6U : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                                        : ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__done_reg)
                                            ? 5U : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))))
            : ((2U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                ? ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                    ? 4U : 3U) : ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))
                                   ? 2U : ((::IData)(vlSelfRef.__Vdfg_Regularize_986)
                                            ? 1U : (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg)))));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__left_reg 
        = __Vdly__tb_pbit_array__DOT__dut__DOT__left_reg_6;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg 
        = __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg_25;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg 
        = __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg_21;
    if (vlSelfRef.tb_pbit_array__DOT__rst) {
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__state_reg = 0x32d689b4U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__acc_reg = 0U;
        __Vdly__tb_pbit_array__DOT__dut__DOT__state_reg_3 = 0U;
    }
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg 
        = __Vdly__tb_pbit_array__DOT__dut__DOT__state_reg_3;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__run_reg 
        = ((::IData)(vlSelfRef.__Vdfg_Regularize_993) 
           | (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__mac_start));
    if ((::IData)((3U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg)))) {
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__gen_lut__DOT__sigmoid_lut_inst__DOT__p_reg 
            = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__gen_lut__DOT__sigmoid_lut_inst__DOT__rom
            [((0x00000200U & ((~ ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg) 
                                  >> 9U)) << 9U)) | 
              (0x000001ffU & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg)))];
    }
    __Vfunc_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__s_1 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__state_reg;
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = __Vfunc_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__s_1;
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r = 0U;
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xfffeU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xfffdU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (2U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                    << 1U)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xfffbU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (4U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                    << 2U)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xfff7U & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (8U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                    << 3U)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xffefU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00000010U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 4U)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xffdfU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00000020U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 5U)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xffbfU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00000040U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 6U)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xff7fU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00000080U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 7U)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xfeffU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00000100U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 8U)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xfdffU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00000200U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 9U)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xfbffU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00000400U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 0x0000000aU)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xf7ffU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00000800U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 0x0000000bU)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xefffU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00001000U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 0x0000000cU)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xdfffU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00002000U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 0x0000000dU)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0xbfffU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00004000U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 0x0000000eU)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r 
        = ((0x7fffU & (::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
           | (0x00008000U & (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                             << 0x0000000fU)));
    tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
        = ((1U & tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)
            ? (0xb4bcd35cU ^ (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                              >> 1U)) : (tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t 
                                         >> 1U));
    __Vfunc_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vfuncout_0 
        = (((::QData)((::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r)) 
            << 0x00000020U) | (::QData)((::IData)(tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t)));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap_w 
        = __Vfunc_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vfuncout_0;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__take_reg 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_next;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__h_q = vlSelfRef.tb_pbit_array__DOT__dut__DOT__hmem
        [vlSelfRef.tb_pbit_array__DOT__dut__DOT__o_q];
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_reg 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_next;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__done_reg 
        = ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__vld_reg) 
           & (7U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg)));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__w_ext 
        = ((0x00001f00U & ((- (::IData)((1U & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__w_q) 
                                               >> 7U)))) 
                           << 8U)) | (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__w_q));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_reg 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_next;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__mac_start 
        = (::IData)((3U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg)));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg 
        = __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg_15;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg 
        = __Vdly__tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg_10;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg 
        = __Vdly__tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg_23;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_next 
        = (::IData)((4U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg)));
    vlSelfRef.__Vdfg_Regularize_994 = (1U & (~ ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg) 
                                                >> 2U)));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__o_q = vlSelfRef.tb_pbit_array__DOT__dut__DOT__omem
        [vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg];
    if (vlSelfRef.tb_pbit_array__DOT__rst) {
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__swd_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__run_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__take_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__done_reg = 0U;
    }
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pb_start 
        = (::IData)(((4U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg)) 
                     & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__done_reg)));
    vlSelfRef.__Vdfg_Regularize_997 = (1U & ((4U != (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg)) 
                                             | (~ (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__done_reg))));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__vld_reg 
        = ((::IData)(vlSelfRef.__Vdfg_Regularize_998) 
           & (::IData)(vlSelfRef.__Vdfg_Regularize_993));
    if (vlSelfRef.tb_pbit_array__DOT__rst) {
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__vld_reg = 0U;
    }
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_next 
        = vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_reg;
    if ((4U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))) {
        if ((1U & (~ ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg) 
                      >> 1U)))) {
            if ((1U & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg))) {
                if (vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_reg) {
                    vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_next 
                        = (((~ ((::IData)(1U) << (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__node_reg))) 
                            & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_next)) 
                           | (0x00ffU & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bit_reg) 
                                         << (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__node_reg))));
                }
            }
        }
    }
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg = __Vdly__tb_pbit_array__DOT__dut__DOT__k_reg_4;
    vlSelfRef.__Vdfg_Regularize_998 = (::IData)((3U 
                                                 != (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg)));
    vlSelfRef.__Vdfg_Regularize_993 = ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__run_reg) 
                                       & (8U > (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg)));
}

void Vtb_pbit_array___024root___timing_ready(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___timing_ready\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((((1ULL & vlSelfRef.__VactTriggered[0U])) != 0ULL)) {
        vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.ready(
                                                              "@(posedge tb_pbit_array.clk)");
    }
}

void Vtb_pbit_array___024root___timing_resume(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___timing_resume\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.moveToResumeQueue(
                                                                      "@(posedge tb_pbit_array.clk)");
    vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.resume(
                                                           "@(posedge tb_pbit_array.clk)");
    if ((((2ULL & vlSelfRef.__VactTriggered[0U])) != 0ULL)) {
        vlSelfRef.__Vnontrivs.__VdlySched.resume();
    }
}

void Vtb_pbit_array___024root___trigger_orInto__act_vec_vec(::VlUnpacked<::QData/*63:0*/, 1> &out, const ::VlUnpacked<::QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___trigger_orInto__act_vec_vec\n"); );
    // Locals
    ::IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = (out[n] | in[n]);
        n = ((::IData)(1U) + n);
    } while ((0U >= n));
}

void Vtb_pbit_array___024root___trigger_clear__act(::VlUnpacked<::QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___trigger_clear__act\n"); );
    // Locals
    ::IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((::IData)(1U) + n);
    } while ((1U > n));
}

void Vtb_pbit_array___024root____VbeforeTrig_h852eb7a7__0(Vtb_pbit_array___024root* vlSelf, const char* __VeventDescription) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root____VbeforeTrig_h852eb7a7__0\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    ::VlUnpacked<::QData/*63:0*/, 1> __VTmp;
    // Body
    __VTmp[0U] = (::QData)((::IData)(((::IData)(vlSelfRef.tb_pbit_array__DOT__clk) 
                                      & (~ (::IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_pbit_array__DOT__clk__0)))));
    vlSelfRef.__Vtrigprevexpr___TOP__tb_pbit_array__DOT__clk__0 
        = vlSelfRef.tb_pbit_array__DOT__clk;
    if ((1ULL & __VTmp[0U])) {
        vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.ready(__VeventDescription);
        vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.ready(__VeventDescription);
        vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.ready(__VeventDescription);
        vlSelfRef.__Vnontrivs.__VtrigSched_h852eb7a7__0.ready(__VeventDescription);
    }
    vlSelfRef.__VactTriggeredAcc[0U] = (vlSelfRef.__VactTriggeredAcc[0U] 
                                        | __VTmp[0U]);
}

#ifdef VL_DEBUG
void Vtb_pbit_array___024root___eval_debug_assertions(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_debug_assertions\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
