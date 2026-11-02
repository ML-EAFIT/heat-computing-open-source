# Tang Nano 9K build for rtl/top.v (open-source flow, OSS CAD Suite).
#   make            synth + place & route + bitstream  -> build/top.fs
#   make report     print utilisation and timing summary
#   make sweep      synthesise N = 8/16/32/64 p-bit variants (fit study)
#   make sweep-sparse   fit study of the SPARSE top (N x P, D = 12)
#   make sparse     SPARSE bitstream, power-on program sp_$(SCASE)_p$(SP)
#                   -> build/sparse_$(SN)_p$(SP)_$(SCASE).fs (flash with make flash-sparse)
#   make case CASE=factor6   bitstream whose power-on program is a model case
#                   -> build/case_<CASE>.fs (flash with make flash-case)
#   make case CASE=factor6_pt   same with parallel tempering (TEMPER_K = 4)
#   make sparse SNOISE=1        SPARSE build with the 'Y' noise options (NOISE_CTL = 1)
#   make pmode PMCASE=chain4    continuous p-mode board image -> build/pmode_<PMCASE>.fs
#   make pmode-fit              p-mode fit study (M = 4, 8, 16 sites)
#   make flash      program SRAM      (openFPGALoader -b tangnano9k)
#   make flash-fl   program the flash (persists over power cycles)
# NOTE: nextpnr-gowin is no longer shipped in OSS CAD Suite; the Gowin
# backend now lives in nextpnr-himbaechel (--vopt family=GW1N-9C).

DEVICE  := GW1NR-LV9QN88PC6/I5
FAMILY  := GW1N-9C
BUILD   := build
TOPMOD  := top
N       ?= 16
HIST    ?= 12
RTL := ../rtl/top.v ../rtl/pbit_array.v ../rtl/coupling_row.v ../rtl/weight_mac.v \
       ../rtl/pbit.v ../rtl/sigmoid_lut.v ../rtl/sigmoid_pwl.v ../rtl/noise_bank.v \
       ../rtl/lfsr.v ../rtl/schedule_ctrl.v ../rtl/readout_counter.v \
       ../rtl/uart_rx.v ../rtl/uart_tx.v ../rtl/sparse_array.v ../rtl/temper_ctrl.v
PMODE_RTL := ../rtl/pmode_top.v ../rtl/pmode_array.v ../rtl/sigmoid_lut.v ../rtl/noise_bank.v ../rtl/lfsr.v

.PHONY: all report sweep sweep-sparse sparse case flash flash-fl flash-case flash-sparse clean pmode pmode-fit flash-pmode
CASE    ?= factor6
PY      ?= /usr/bin/python3

all: $(BUILD)/top.fs

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/top.json: $(RTL) | $(BUILD)
	yosys -q -l $(BUILD)/yosys.log -p "read_verilog -defer $(RTL); \
	  hierarchy -top $(TOPMOD) -chparam N $(N) -chparam HIST_BITS $(HIST); \
	  synth_gowin -top $(TOPMOD) -json $@; tee -o $(BUILD)/yosys_stat.txt stat"

$(BUILD)/pnr.json: $(BUILD)/top.json tangnano9k.cst
	nextpnr-himbaechel --json $< --write $@ --device $(DEVICE) \
	  --vopt family=$(FAMILY) --vopt cst=tangnano9k.cst \
	  --freq 27 --report $(BUILD)/pnr_report.json -l $(BUILD)/nextpnr.log

$(BUILD)/top.fs: $(BUILD)/pnr.json
	gowin_pack -d $(FAMILY) -o $@ $<

report:
	@grep -A40 "Device utilisation" $(BUILD)/nextpnr.log | head -40
	@grep -E "Max frequency for clock" $(BUILD)/nextpnr.log | tail -2

# fit study: N p-bits with zero-initialised RAMs (programmed over UART later)
SWEEP_N ?= 8 16 32 64 128
sweep: | $(BUILD)
	@for n in $(SWEEP_N); do \
	  $(PY) sweep_wrapper.py $$n; \
	  yosys -q -l $(BUILD)/sweep_yosys_$$n.log -p "read_verilog -defer $(RTL) $(BUILD)/sweep_top_$$n.v; \
	    hierarchy -top sweep_top; synth_gowin -top sweep_top -json $(BUILD)/sweep_$$n.json" \
	    > /dev/null 2>&1 || { echo "N=$$n: synthesis FAILED (see $(BUILD)/sweep_yosys_$$n.log)"; continue; }; \
	  nextpnr-himbaechel --json $(BUILD)/sweep_$$n.json --write $(BUILD)/sweep_pnr_$$n.json \
	    --device $(DEVICE) --vopt family=$(FAMILY) --vopt cst=tangnano9k.cst --freq 27 \
	    -l $(BUILD)/sweep_$$n.log > /dev/null 2>&1 || { echo "N=$$n: place & route FAILED (see $(BUILD)/sweep_$$n.log)"; continue; }; \
	  printf "N=%-4s %s | %s | %s | %s\n" $$n \
	    "$$(grep -E 'LUT4:' $(BUILD)/sweep_$$n.log | head -1 | sed 's/Info: *//;s/  */ /g')" \
	    "$$(grep -E ' DFF:' $(BUILD)/sweep_$$n.log | head -1 | sed 's/Info: *//;s/  */ /g')" \
	    "$$(grep -E 'BSRAM:' $(BUILD)/sweep_$$n.log | head -1 | sed 's/Info: *//;s/  */ /g')" \
	    "$$(grep -E 'Max frequency for clock' $(BUILD)/sweep_$$n.log | tail -1 | sed 's/Info: Max frequency for clock//')"; \
	done

# SPARSE fit study: N p-bits on degree <= SD graphs, SP lanes, empty RAMs
SWEEP_SN ?= 64 128 256
SWEEP_SP ?= 1 2 4
SD       ?= 12
sweep-sparse: | $(BUILD)
	@for n in $(SWEEP_SN); do for p in $(SWEEP_SP); do t=$${n}_p$${p}; \
	  $(PY) sparse_wrapper.py $$n $$p $(SD) > /dev/null; \
	  yosys -q -l $(BUILD)/sparse_yosys_$$t.log -p "read_verilog -defer $(RTL) $(BUILD)/sparse_top_$$t.v; \
	    hierarchy -top sparse_top; synth_gowin -top sparse_top -json $(BUILD)/sparse_$$t.json" \
	    > /dev/null 2>&1 || { echo "N=$$n P=$$p: synthesis FAILED (see $(BUILD)/sparse_yosys_$$t.log)"; continue; }; \
	  nextpnr-himbaechel --json $(BUILD)/sparse_$$t.json --write $(BUILD)/sparse_pnr_$$t.json \
	    --device $(DEVICE) --vopt family=$(FAMILY) --vopt cst=tangnano9k.cst --freq 27 \
	    -l $(BUILD)/sparse_$$t.log > /dev/null 2>&1 || { echo "N=$$n P=$$p: place & route FAILED (see $(BUILD)/sparse_$$t.log)"; continue; }; \
	  printf "N=%-4s P=%s %s | %s | %s | %s\n" $$n $$p \
	    "$$(grep -E 'LUT4:' $(BUILD)/sparse_$$t.log | head -1 | sed 's/Info: *//;s/  */ /g')" \
	    "$$(grep -E ' DFF:' $(BUILD)/sparse_$$t.log | head -1 | sed 's/Info: *//;s/  */ /g')" \
	    "$$(grep -E 'BSRAM:' $(BUILD)/sparse_$$t.log | head -1 | sed 's/Info: *//;s/  */ /g')" \
	    "$$(grep -E 'Max frequency for clock' $(BUILD)/sparse_$$t.log | tail -1 | sed 's/Info: Max frequency for clock//')"; \
	done; done

# SPARSE bitstream with a power-on case (default: 16x16 G12 ferromagnet, P = 4)
SN    ?= 256
SP    ?= 4
SCASE ?= g12_ferro16
SNOISE ?= 0
STAG  := $(SN)_p$(SP)_$(SCASE)$(if $(filter 1,$(SNOISE)),_noise)
sparse: | $(BUILD)
	$(PY) sparse_wrapper.py $(SN) $(SP) $(SD) $(SCASE) $(if $(filter 1,$(SNOISE)),noise)
	yosys -q -l $(BUILD)/sparse_yosys_$(STAG).log -p "read_verilog -defer $(RTL) $(BUILD)/sparse_top_$(STAG).v; \
	  hierarchy -top sparse_top; synth_gowin -top sparse_top -json $(BUILD)/sparse_$(STAG).json"
	nextpnr-himbaechel --json $(BUILD)/sparse_$(STAG).json --write $(BUILD)/sparse_pnr_$(STAG).json \
	  --device $(DEVICE) --vopt family=$(FAMILY) --vopt cst=tangnano9k.cst --freq 27 \
	  -l $(BUILD)/sparse_$(STAG).log > /dev/null 2>&1
	gowin_pack -d $(FAMILY) -o $(BUILD)/sparse_$(STAG).fs $(BUILD)/sparse_pnr_$(STAG).json
	@grep -E "LUT4:|ALU:| DFF:|BSRAM:" $(BUILD)/sparse_$(STAG).log | head -4
	@grep -E "Max frequency for clock" $(BUILD)/sparse_$(STAG).log | tail -1

flash-sparse:
	openFPGALoader -b tangnano9k $(BUILD)/sparse_$(STAG).fs

# case build: top.v with the RAMs initialised from model/vectors/$(CASE)/
case: | $(BUILD)
	$(PY) case_wrapper.py $(CASE)
	yosys -q -l $(BUILD)/case_yosys_$(CASE).log -p "read_verilog -defer $(RTL) $(BUILD)/case_top_$(CASE).v; \
	  hierarchy -top case_top; synth_gowin -top case_top -json $(BUILD)/case_$(CASE).json"
	nextpnr-himbaechel --json $(BUILD)/case_$(CASE).json --write $(BUILD)/case_pnr_$(CASE).json \
	  --device $(DEVICE) --vopt family=$(FAMILY) --vopt cst=tangnano9k.cst --freq 27 \
	  -l $(BUILD)/case_$(CASE).log > /dev/null 2>&1
	gowin_pack -d $(FAMILY) -o $(BUILD)/case_$(CASE).fs $(BUILD)/case_pnr_$(CASE).json
	@grep -E "LUT4:|ALU:| DFF:|BSRAM:" $(BUILD)/case_$(CASE).log | head -4
	@grep -E "Max frequency for clock" $(BUILD)/case_$(CASE).log | tail -1

flash-case:
	openFPGALoader -b tangnano9k $(BUILD)/case_$(CASE).fs

# continuous p-modes (pmode_top.v): power-on case, run button toggles stepping, LEDs = sign(x)
PMCASE ?= chain4
pmode: | $(BUILD)
	$(PY) pmode_wrapper.py $(PMCASE)
	yosys -q -l $(BUILD)/pmode_yosys_$(PMCASE).log -p "read_verilog -defer $(PMODE_RTL) $(BUILD)/pmode_top_$(PMCASE).v; \
	  hierarchy -top pmode_wrap; synth_gowin -top pmode_wrap -json $(BUILD)/pmode_$(PMCASE).json"
	nextpnr-himbaechel --json $(BUILD)/pmode_$(PMCASE).json --write $(BUILD)/pmode_pnr_$(PMCASE).json \
	  --device $(DEVICE) --vopt family=$(FAMILY) --vopt cst=tangnano9k.cst --freq 27 \
	  -l $(BUILD)/pmode_$(PMCASE).log > /dev/null 2>&1
	gowin_pack -d $(FAMILY) -o $(BUILD)/pmode_$(PMCASE).fs $(BUILD)/pmode_pnr_$(PMCASE).json
	@grep -E "LUT4:|ALU:| DFF:|BSRAM:|MULT" $(BUILD)/pmode_$(PMCASE).log | head -8
	@grep -E "Max frequency for clock" $(BUILD)/pmode_$(PMCASE).log | tail -1

PM_FIT ?= 4 8 16
pmode-fit: | $(BUILD)
	@for m in $(PM_FIT); do \
	  $(PY) pmode_wrapper.py fit $$m > /dev/null; \
	  yosys -q -l $(BUILD)/pmode_yosys_fit$$m.log -p "read_verilog -defer $(PMODE_RTL) $(BUILD)/pmode_top_fit$$m.v; \
	    hierarchy -top pmode_wrap; synth_gowin -top pmode_wrap -json $(BUILD)/pmode_fit$$m.json" \
	    > /dev/null 2>&1 || { echo "M=$$m: synthesis FAILED"; continue; }; \
	  nextpnr-himbaechel --json $(BUILD)/pmode_fit$$m.json --write $(BUILD)/pmode_pnr_fit$$m.json \
	    --device $(DEVICE) --vopt family=$(FAMILY) --vopt cst=tangnano9k.cst --freq 27 \
	    -l $(BUILD)/pmode_fit$$m.log > /dev/null 2>&1 || { echo "M=$$m: place & route FAILED"; continue; }; \
	  printf "M=%-3s %s | %s | %s | %s\n" $$m \
	    "$$(grep -E 'LUT4:' $(BUILD)/pmode_fit$$m.log | head -1 | sed 's/Info: *//;s/  */ /g')" \
	    "$$(grep -E ' DFF:' $(BUILD)/pmode_fit$$m.log | head -1 | sed 's/Info: *//;s/  */ /g')" \
	    "$$(grep -E 'MULT|DSP' $(BUILD)/pmode_fit$$m.log | grep -v ' 0/' | head -3 | sed 's/Info: *//;s/  */ /g' | tr '\n' ' ')" \
	    "$$(grep -E 'Max frequency for clock' $(BUILD)/pmode_fit$$m.log | tail -1 | sed 's/Info: Max frequency for clock//')"; \
	done

flash-pmode:
	openFPGALoader -b tangnano9k $(BUILD)/pmode_$(PMCASE).fs

flash: $(BUILD)/top.fs
	openFPGALoader -b tangnano9k $<

flash-fl: $(BUILD)/top.fs
	openFPGALoader -b tangnano9k -f $<

clean:
	rm -rf $(BUILD)
