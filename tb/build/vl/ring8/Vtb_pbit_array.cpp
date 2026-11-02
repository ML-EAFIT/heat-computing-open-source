// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vtb_pbit_array__pch.h"

//============================================================
// Constructors

Vtb_pbit_array::Vtb_pbit_array(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vtb_pbit_array__Syms(contextp(), _vcname__, this)}
    , m_evalLoop{*this, /*convergeLimit:*/ 10000}
    , rootp{&(vlSymsp->vlInst.TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vtb_pbit_array::Vtb_pbit_array(const char* _vcname__)
    : Vtb_pbit_array(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vtb_pbit_array::~Vtb_pbit_array() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vtb_pbit_array___024root___eval_debug_assertions(Vtb_pbit_array___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD void Vtb_pbit_array___024root___eval_static(Vtb_pbit_array___024root* vlSelf);
void Vtb_pbit_array___024root___eval_initial(Vtb_pbit_array___024root* vlSelf);
VL_ATTR_COLD bool Vtb_pbit_array___024root___eval_stl(Vtb_pbit_array___024root* vlSelf, ::CData/*0:0*/ firstIteration);
void Vtb_pbit_array___024root___eval_sample(Vtb_pbit_array___024root* vlSelf);
bool Vtb_pbit_array___024root___eval_ico(Vtb_pbit_array___024root* vlSelf, ::CData/*0:0*/ firstIteration);
bool Vtb_pbit_array___024root___eval_act(Vtb_pbit_array___024root* vlSelf);
bool Vtb_pbit_array___024root___eval_inact(Vtb_pbit_array___024root* vlSelf);
bool Vtb_pbit_array___024root___eval_nba(Vtb_pbit_array___024root* vlSelf);
bool Vtb_pbit_array___024root___eval_obs(Vtb_pbit_array___024root* vlSelf);
bool Vtb_pbit_array___024root___eval_react(Vtb_pbit_array___024root* vlSelf);
void Vtb_pbit_array___024root___eval_postponed(Vtb_pbit_array___024root* vlSelf);
VL_ATTR_COLD void Vtb_pbit_array___024root___eval_final(Vtb_pbit_array___024root* vlSelf);
VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__stl(Vtb_pbit_array___024root* vlSelf);
VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__ico(Vtb_pbit_array___024root* vlSelf);
VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__act(Vtb_pbit_array___024root* vlSelf);
VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__nba(Vtb_pbit_array___024root* vlSelf);
VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__obs(Vtb_pbit_array___024root* vlSelf);
VL_ATTR_COLD void Vtb_pbit_array___024root___eval_dump_triggers__react(Vtb_pbit_array___024root* vlSelf);

void Vtb_pbit_array::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vtb_pbit_array::eval_step\n"); );
    m_evalLoop.eval();
}

void Vtb_pbit_array::evalBegin() {
#ifdef VL_DEBUG
    // Debug assertions
    Vtb_pbit_array___024root___eval_debug_assertions(&(vlSymsp->vlInst.TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
}

void Vtb_pbit_array::evalEnd() {
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
    vlSymsp->vlInst.TOP.__Vnontrivs.__VdlySched.cleanupForevered();
}

void Vtb_pbit_array::evalStatic() {
    Vtb_pbit_array___024root___eval_static(&(vlSymsp->vlInst.TOP));
}

void Vtb_pbit_array::evalInitial() {
    Vtb_pbit_array___024root___eval_initial(&(vlSymsp->vlInst.TOP));
}

bool Vtb_pbit_array::evalStl(bool firstIteration) {
    return Vtb_pbit_array___024root___eval_stl(&(vlSymsp->vlInst.TOP), firstIteration);
}

void Vtb_pbit_array::evalSample() {
    Vtb_pbit_array___024root___eval_sample(&(vlSymsp->vlInst.TOP));
}

bool Vtb_pbit_array::evalIco(bool firstIteration) {
    return Vtb_pbit_array___024root___eval_ico(&(vlSymsp->vlInst.TOP), firstIteration);
}

bool Vtb_pbit_array::evalAct() {
    return Vtb_pbit_array___024root___eval_act(&(vlSymsp->vlInst.TOP));
}

bool Vtb_pbit_array::evalInact() {
    return Vtb_pbit_array___024root___eval_inact(&(vlSymsp->vlInst.TOP));
}

bool Vtb_pbit_array::evalNba() {
    return Vtb_pbit_array___024root___eval_nba(&(vlSymsp->vlInst.TOP));
}

bool Vtb_pbit_array::evalObs() {
    return Vtb_pbit_array___024root___eval_obs(&(vlSymsp->vlInst.TOP));
}

bool Vtb_pbit_array::evalReact() {
    return Vtb_pbit_array___024root___eval_react(&(vlSymsp->vlInst.TOP));
}

void Vtb_pbit_array::evalPostponed() {
    Vtb_pbit_array___024root___eval_postponed(&(vlSymsp->vlInst.TOP));
}

void Vtb_pbit_array::evalFinal() {
    Vtb_pbit_array___024root___eval_final(&(vlSymsp->vlInst.TOP));
}

VL_ATTR_COLD void Vtb_pbit_array::dumpTriggersStl() {
    Vtb_pbit_array___024root___eval_dump_triggers__stl(&(vlSymsp->vlInst.TOP));
}

VL_ATTR_COLD void Vtb_pbit_array::dumpTriggersIco() {
    Vtb_pbit_array___024root___eval_dump_triggers__ico(&(vlSymsp->vlInst.TOP));
}

VL_ATTR_COLD void Vtb_pbit_array::dumpTriggersAct() {
    Vtb_pbit_array___024root___eval_dump_triggers__act(&(vlSymsp->vlInst.TOP));
}

VL_ATTR_COLD void Vtb_pbit_array::dumpTriggersNba() {
    Vtb_pbit_array___024root___eval_dump_triggers__nba(&(vlSymsp->vlInst.TOP));
}

VL_ATTR_COLD void Vtb_pbit_array::dumpTriggersObs() {
    Vtb_pbit_array___024root___eval_dump_triggers__obs(&(vlSymsp->vlInst.TOP));
}

VL_ATTR_COLD void Vtb_pbit_array::dumpTriggersReact() {
    Vtb_pbit_array___024root___eval_dump_triggers__react(&(vlSymsp->vlInst.TOP));
}

//============================================================
// Events and timing
bool Vtb_pbit_array::eventsPending() { return !vlSymsp->vlInst.TOP.__Vnontrivs.__VdlySched.empty() && !contextp()->gotFinish(); }

uint64_t Vtb_pbit_array::nextTimeSlot() { return vlSymsp->vlInst.TOP.__Vnontrivs.__VdlySched.nextTimeSlot(); }

//============================================================
// Utilities

const char* Vtb_pbit_array::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

VL_ATTR_COLD void Vtb_pbit_array::final() {
    contextp()->executingFinal(true);
    evalFinal();
    contextp()->executingFinal(false);
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vtb_pbit_array::hierName() const { return vlSymsp->name(); }
const char* Vtb_pbit_array::modelName() const { return "Vtb_pbit_array"; }
unsigned Vtb_pbit_array::threads() const { return 1; }
void Vtb_pbit_array::prepareClone() const { contextp()->prepareClone(); }
void Vtb_pbit_array::atClone() const {
    contextp()->threadPoolpOnClone();
}
