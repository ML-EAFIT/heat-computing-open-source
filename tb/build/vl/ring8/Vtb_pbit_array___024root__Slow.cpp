// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_pbit_array.h for the primary calling header

#include "Vtb_pbit_array__pch.h"

void Vtb_pbit_array___024root___ctor_var_reset(Vtb_pbit_array___024root* vlSelf);

void Vtb_pbit_array___024root__ctor(Vtb_pbit_array___024root* vlSelf, Vtb_pbit_array__Syms* symsp, const char* namep) {
    vlSelf->vlSymsp = symsp;
    vlSelf->vlNamep = strdup(namep);
    // Construct members
    VL_INPLACE_CONSTRUCT(vlSelf->__Vnontrivs.__VdlySched, *vlSelf->vlSymsp->_vm_contextp__);
    VL_INPLACE_CONSTRUCT(vlSelf->__Vnontrivs.__VtrigSched_h852eb7a7__0);
    // Reset structure values
    Vtb_pbit_array___024root___ctor_var_reset(vlSelf);
}

Vtb_pbit_array___024root::Vtb_pbit_array___024root(Vtb_pbit_array__Syms* symsp, const char* namep) {
    Vtb_pbit_array___024root__ctor(this, symsp, namep);
}

void Vtb_pbit_array___024root__dtor(Vtb_pbit_array___024root* vlSelf) {
    // Destroy members
    VL_INPLACE_DESTRUCT(vlSelf->__Vnontrivs.__VtrigSched_h852eb7a7__0);
    VL_INPLACE_DESTRUCT(vlSelf->__Vnontrivs.__VdlySched);
    VL_DO_DANGLING(std::free(const_cast<char*>(vlSelf->vlNamep)), vlSelf->vlNamep);
}

Vtb_pbit_array___024root::~Vtb_pbit_array___024root() {
    Vtb_pbit_array___024root__dtor(this);
}
