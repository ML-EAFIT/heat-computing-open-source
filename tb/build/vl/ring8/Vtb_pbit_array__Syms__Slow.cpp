// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vtb_pbit_array__pch.h"


void Vtb_pbit_array__Syms__ctor(Vtb_pbit_array__Syms* vlSymsp, const char* namep) {
    // Setup top module instance
    VL_INPLACE_CONSTRUCT(vlSymsp->vlInst.TOP, vlSymsp, namep);
    // Check resources
    Verilated::stackCheck(38);
    // Setup sub module instances
    // Configure time unit / time precision
    vlSymsp->_vm_contextp__->timeunit(-9);
    vlSymsp->_vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
}

Vtb_pbit_array__Syms::Vtb_pbit_array__Syms(VerilatedContext* contextp, const char* namep, Vtb_pbit_array* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    , __Vm_didInit{modelp->m_didInit}
{
    Vtb_pbit_array__Syms__ctor(this, namep);
}

void Vtb_pbit_array__Syms__dtor(Vtb_pbit_array__Syms* vlSymsp) {
    // Tear down scopes
    // Tear down sub module instances
    // Tear down top module instance
    VL_INPLACE_DESTRUCT(vlSymsp->vlInst.TOP);
}

Vtb_pbit_array__Syms::~Vtb_pbit_array__Syms() {
    Vtb_pbit_array__Syms__dtor(this);
}
