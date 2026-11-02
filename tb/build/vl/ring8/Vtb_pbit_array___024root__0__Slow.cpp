// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_pbit_array.h for the primary calling header

#include "Vtb_pbit_array__pch.h"

void Vtb_pbit_array___024root___timing_ready(Vtb_pbit_array___024root* vlSelf);

VL_ATTR_COLD void Vtb_pbit_array___024root___eval_static(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_static\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    {
        // Inlined CFunc: _eval_static__TOP
        ::IData/*31:0*/ __Vinline_0__eval_static__TOP_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t;
        __Vinline_0__eval_static__TOP_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t = 0;
        ::SData/*15:0*/ __Vinline_0__eval_static__TOP_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r;
        __Vinline_0__eval_static__TOP_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r = 0;
        vlSelfRef.tb_pbit_array__DOT__clk = 0U;
        vlSelfRef.tb_pbit_array__DOT__rst = 1U;
        vlSelfRef.tb_pbit_array__DOT__run = 0U;
        vlSelfRef.tb_pbit_array__DOT__nsw = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__w_q = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__h_q = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__o_q = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__k_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__node_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__left_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__spins_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__swd_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__state_reg = 0x32d689b4U;
        __Vinline_0__eval_static__TOP_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__t = 0;
        __Vinline_0__eval_static__TOP_tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap__Vstatic__r = 0;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__prod_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__mcand_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bsh_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__cnt_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__logit_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__bit_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__take_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__gen_lut__DOT__sigmoid_lut_inst__DOT__p_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__run_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__vld_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__jd_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__done_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__row_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__acc_reg = 0U;
        vlSelfRef.tb_pbit_array__DOT__sched_inst__DOT__beta_reg = 0U;
    }
    vlSelfRef.__Vtrigprevexpr___TOP__tb_pbit_array__DOT__clk__0 = 0U;
    Vtb_pbit_array___024root___timing_ready(vlSelf);
    do {
        vlSelfRef.__VactTriggeredAcc[vlSelfRef.__Vi] 
            = vlSelfRef.__VactTriggered[vlSelfRef.__Vi];
        vlSelfRef.__Vi = ((::IData)(1U) + vlSelfRef.__Vi);
    } while ((0U >= vlSelfRef.__Vi));
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_pbit_array___024root___dump_triggers__stl(const ::VlUnpacked<::QData/*63:0*/, 1> &triggers, const ::std::string &tag);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtb_pbit_array___024root___trigger_anySet__stl(const ::VlUnpacked<::QData/*63:0*/, 1> &in);
VL_ATTR_COLD void Vtb_pbit_array___024root___stl_comb__TOP__0(Vtb_pbit_array___024root* vlSelf);

VL_ATTR_COLD bool Vtb_pbit_array___024root___eval_stl(Vtb_pbit_array___024root* vlSelf, ::CData/*0:0*/ firstIteration) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_stl\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    ::CData/*0:0*/ __VstlExecute;
    // Body
    vlSelfRef.__VstlTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VstlTriggered[0U]) 
                                     | (::IData)((::IData)(firstIteration)));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_pbit_array___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
    }
#endif
    __VstlExecute = Vtb_pbit_array___024root___trigger_anySet__stl(vlSelfRef.__VstlTriggered);
    if (__VstlExecute) {
        {
            // Inlined CFunc: _eval_body__stl
            if ((((1ULL & vlSelfRef.__VstlTriggered[0U])) != 0ULL)) {
                Vtb_pbit_array___024root___stl_comb__TOP__0(vlSelf);
            }
        }
    }
    return (__VstlExecute);
}

VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__stl(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_dump_triggers__stl\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
#ifdef VL_DEBUG
    Vtb_pbit_array___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
#endif
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_pbit_array___024root___dump_triggers__ico(const ::VlUnpacked<::QData/*63:0*/, 1> &triggers, const ::std::string &tag);
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__ico(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_dump_triggers__ico\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
#ifdef VL_DEBUG
    Vtb_pbit_array___024root___dump_triggers__ico(vlSelfRef.__VicoTriggered, "ico"s);
#endif
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_pbit_array___024root___dump_triggers__act(const ::VlUnpacked<::QData/*63:0*/, 1> &triggers, const ::std::string &tag);
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__act(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_dump_triggers__act\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
#ifdef VL_DEBUG
    Vtb_pbit_array___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
}

VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__nba(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_dump_triggers__nba\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
#ifdef VL_DEBUG
    Vtb_pbit_array___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
}

VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__obs(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_dump_triggers__obs\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__react(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_dump_triggers__react\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vtb_pbit_array___024root___eval_final(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_final\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vtb_pbit_array___024root___eval_initial__TOP(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___eval_initial__TOP\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    ::VlUnpacked<::CData/*7:0*/, 16> tb_pbit_array__DOT__sched_inst__DOT__table_mem;
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        tb_pbit_array__DOT__sched_inst__DOT__table_mem[__Vi0] = 0;
    }
    ::IData/*31:0*/ __Vilp1;
    // Body
    __Vilp1 = 0U;
    while ((__Vilp1 <= 0x0000003fU)) {
        vlSelfRef.tb_pbit_array__DOT__dut__DOT__wmem[__Vilp1] = 0U;
        __Vilp1 = ((::IData)(1U) + __Vilp1);
    }
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__hmem[0U] = 0U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__omem[0U] = 0U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__hmem[1U] = 0U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__omem[1U] = 1U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__hmem[2U] = 0U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__omem[2U] = 2U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__hmem[3U] = 0U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__omem[3U] = 3U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__hmem[4U] = 0U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__omem[4U] = 4U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__hmem[5U] = 0U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__omem[5U] = 5U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__hmem[6U] = 0U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__omem[6U] = 6U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__hmem[7U] = 0U;
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__omem[7U] = 7U;
    VL_READMEM_N(true, 8, 64, 0, "../model/vectors/ring8/weights.hex"s
                 ,  &(vlSelfRef.tb_pbit_array__DOT__dut__DOT__wmem)
                 , 0U, 0x0000003fU);
    VL_READMEM_N(true, 8, 8, 0, "../model/vectors/ring8/bias.hex"s
                 ,  &(vlSelfRef.tb_pbit_array__DOT__dut__DOT__hmem)
                 , 0U, 7U);
    VL_READMEM_N(true, 3, 8, 0, "../model/vectors/ring8/order.hex"s
                 ,  &(vlSelfRef.tb_pbit_array__DOT__dut__DOT__omem)
                 , 0U, 7U);
    VL_READMEM_N(true, 16, 1024, 0, "../model/vectors/sigmoid_lut.hex"s
                 ,  &(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__gen_lut__DOT__sigmoid_lut_inst__DOT__rom)
                 , 0U, 0x000003ffU);
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[0U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[1U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[2U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[3U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[4U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[5U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[6U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[7U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[8U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[9U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[10U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[11U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[12U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[13U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[14U] = 0U;
    tb_pbit_array__DOT__sched_inst__DOT__table_mem[15U] = 0U;
    VL_READMEM_N(true, 8, 16, 0, "../model/vectors/ring8/beta_table.hex"s
                 ,  &(tb_pbit_array__DOT__sched_inst__DOT__table_mem)
                 , 0U, 0x0000000fU);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_pbit_array___024root___dump_triggers__stl(const ::VlUnpacked<::QData/*63:0*/, 1> &triggers, const ::std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (::IData)(Vtb_pbit_array___024root___trigger_anySet__stl(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (::IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD bool Vtb_pbit_array___024root___trigger_anySet__stl(const ::VlUnpacked<::QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___trigger_anySet__stl\n"); );
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

VL_ATTR_COLD void Vtb_pbit_array___024root___stl_comb__TOP__0(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___stl_comb__TOP__0\n"); );
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
    // Body
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_next 
        = (::IData)((4U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg)));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__mac_start 
        = (::IData)((3U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg)));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__w_ext 
        = ((0x00001f00U & ((- (::IData)((1U & ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__w_q) 
                                               >> 7U)))) 
                           << 8U)) | (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__w_q));
    vlSelfRef.__Vdfg_Regularize_994 = (1U & (~ ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__state_reg) 
                                                >> 2U)));
    vlSelfRef.__Vdfg_Regularize_998 = (::IData)((3U 
                                                 != (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg)));
    vlSelfRef.tb_pbit_array__DOT__dut__DOT__pb_start 
        = (::IData)(((4U == (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg)) 
                     & (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__done_reg)));
    vlSelfRef.__Vdfg_Regularize_986 = ((::IData)(vlSelfRef.tb_pbit_array__DOT__run) 
                                       & (0U != vlSelfRef.tb_pbit_array__DOT__nsw));
    vlSelfRef.__Vdfg_Regularize_993 = ((::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__run_reg) 
                                       & (8U > (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__j_reg)));
    vlSelfRef.__Vdfg_Regularize_997 = (1U & ((4U != (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__state_reg)) 
                                             | (~ (::IData)(vlSelfRef.tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__done_reg))));
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
}

bool Vtb_pbit_array___024root___trigger_anySet__ico(const ::VlUnpacked<::QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_pbit_array___024root___dump_triggers__ico(const ::VlUnpacked<::QData/*63:0*/, 1> &triggers, const ::std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (::IData)(Vtb_pbit_array___024root___trigger_anySet__ico(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (::IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

bool Vtb_pbit_array___024root___trigger_anySet__act(const ::VlUnpacked<::QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_pbit_array___024root___dump_triggers__act(const ::VlUnpacked<::QData/*63:0*/, 1> &triggers, const ::std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (::IData)(Vtb_pbit_array___024root___trigger_anySet__act(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (::IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: @(posedge tb_pbit_array.clk)\n");
    }
    if ((1U & (::IData)((triggers[0U] >> 1U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 1 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb_pbit_array___024root___ctor_var_reset(Vtb_pbit_array___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_pbit_array___024root___ctor_var_reset\n"); );
    Vtb_pbit_array__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->vlNamep);
    vlSelf->tb_pbit_array__DOT__expect_mem[0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 5091178619117726220ull);
    VL_UNPACKED_FILL(vlSelf->tb_pbit_array__DOT__expect_mem.data(), sizeof(vlSelf->tb_pbit_array__DOT__expect_mem[0]), 2000);
    vlSelf->tb_pbit_array__DOT__sweeps = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 64605424483113713ull);
    vlSelf->tb_pbit_array__DOT__cyc1 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 2526993210881625151ull);
    vlSelf->tb_pbit_array__DOT__cyc = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 14832038971986632180ull);
    VL_SCOPED_RAND_RESET_W(1024, vlSelf->tb_pbit_array__DOT__trace_file, __VscopeHash, 14662826230031340181ull);
    vlSelf->tb_pbit_array__DOT__dut__DOT__wmem[0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 4030057322065042444ull);
    VL_UNPACKED_FILL(vlSelf->tb_pbit_array__DOT__dut__DOT__wmem.data(), sizeof(vlSelf->tb_pbit_array__DOT__dut__DOT__wmem[0]), 64);
    vlSelf->tb_pbit_array__DOT__dut__DOT__hmem[0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 18181019502136210862ull);
    VL_UNPACKED_FILL(vlSelf->tb_pbit_array__DOT__dut__DOT__hmem.data(), sizeof(vlSelf->tb_pbit_array__DOT__dut__DOT__hmem[0]), 8);
    vlSelf->tb_pbit_array__DOT__dut__DOT__omem[0] = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 9704700854167611664ull);
    VL_UNPACKED_FILL(vlSelf->tb_pbit_array__DOT__dut__DOT__omem.data(), sizeof(vlSelf->tb_pbit_array__DOT__dut__DOT__omem[0]), 8);
    vlSelf->tb_pbit_array__DOT__dut__DOT__spins_next = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 2349969254737680037ull);
    vlSelf->tb_pbit_array__DOT__dut__DOT__mac_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12598139196655841179ull);
    vlSelf->tb_pbit_array__DOT__dut__DOT__pb_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17625189615767959082ull);
    vlSelf->tb_pbit_array__DOT__dut__DOT__noise_bank_inst__DOT__gen_lane__BRA__0__KET____DOT__lfsr_inst__DOT__leap_w = VL_SCOPED_RAND_RESET_Q(48, __VscopeHash, 151304263316412542ull);
    vlSelf->tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__done_next = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2242331973851369362ull);
    vlSelf->tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__gen_lut__DOT__sigmoid_lut_inst__DOT__rom[0] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17447032192484474689ull);
    VL_UNPACKED_FILL(vlSelf->tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__gen_lut__DOT__sigmoid_lut_inst__DOT__rom.data(), sizeof(vlSelf->tb_pbit_array__DOT__dut__DOT__pbit_inst__DOT__gen_lut__DOT__sigmoid_lut_inst__DOT__rom[0]), 1024);
    vlSelf->tb_pbit_array__DOT__dut__DOT__coupling_row_inst__DOT__weight_mac_inst__DOT__w_ext = VL_SCOPED_RAND_RESET_I(13, __VscopeHash, 5236245426629607113ull);
    vlSelf->__Vdfg_Regularize_986 = 0;
    vlSelf->__Vdfg_Regularize_993 = 0;
    vlSelf->__Vdfg_Regularize_994 = 0;
    vlSelf->__Vdfg_Regularize_997 = 0;
    vlSelf->__Vdfg_Regularize_998 = 0;
    vlSelf->__Vdly_Set__tb_pbit_array__DOT__rst_28 = 0;
    vlSelf->__Vdly_Val__tb_pbit_array__DOT__nsw_29 = 0;
    vlSelf->__Vdly_Set__tb_pbit_array__DOT__nsw_30 = 0;
    vlSelf->__Vdly_Set__tb_pbit_array__DOT__run_31 = 0;
    vlSelf->__Vdly_Set__tb_pbit_array__DOT__run_32 = 0;
    vlSelf->__VstlTriggered[0] = 0;
    VL_UNPACKED_FILL(vlSelf->__VstlTriggered.data(), sizeof(vlSelf->__VstlTriggered[0]), 1);
    vlSelf->__VicoTriggered[0] = 0;
    VL_UNPACKED_FILL(vlSelf->__VicoTriggered.data(), sizeof(vlSelf->__VicoTriggered[0]), 1);
    vlSelf->__VactTriggered[0] = 0;
    VL_UNPACKED_FILL(vlSelf->__VactTriggered.data(), sizeof(vlSelf->__VactTriggered[0]), 1);
    vlSelf->__VactTriggeredAcc[0] = 0;
    VL_UNPACKED_FILL(vlSelf->__VactTriggeredAcc.data(), sizeof(vlSelf->__VactTriggeredAcc[0]), 1);
    vlSelf->__Vtrigprevexpr___TOP__tb_pbit_array__DOT__clk__0 = 0;
    vlSelf->__VnbaTriggered[0] = 0;
    VL_UNPACKED_FILL(vlSelf->__VnbaTriggered.data(), sizeof(vlSelf->__VnbaTriggered[0]), 1);
    vlSelf->__Vi = 0;
}
