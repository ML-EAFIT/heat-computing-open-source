# Verification flow for the p-bit FPGA emulator.
#   make test     : unit tests + bit-exact array tests (iverilog) + lint
#   make stats    : long Verilator runs + histogram comparison (TV/KL) + plots
#   make stats-sparse : long sparse_array runs, TV vs exact (N<=16), energy/|m| vs float Gibbs (N=256)
#   make top      : UART-driven top-level test (Verilator)
#   make top-sparse : SPARSE top (N = 256, P = 4) over UART vs Python replay
#   make denoise  : denoising-chain demo test
#   make pgates   : p-gate library (AND/OR/NOT/XOR, full adder, factor6) forward + backward
#   make sparse   : sparse_array (P parallel lanes, neighbour lists) bit-exact, all sp_* cases
#   make noise    : RNG persistence (tau0) / shared-noise study, raw-noise entropy
#   make pmode    : continuous p-mode (Langevin) cell: bit-exact + histograms vs exact densities
#   make train    : Boltzmann-machine learning with the chip as sampler (+ RTL replay of 3 epochs)
#   make script   : generic UART sessions (tb_top_script.v) vs model/uart_script.TopModel
#   make all      : everything
# Requires the OSS CAD Suite on PATH (source ~/tools/oss-cad-suite/environment).

SHELL    := /bin/bash
RTL_DIR  := ../rtl
VEC      := ../model/vectors
BUILD    := build
# Verilator's generated makefiles cannot live in a path containing spaces
VL_BUILD ?= /tmp/pbit_vl_$(USER)
# the OSS CAD Suite environment shadows python3 (no numpy); use the system one
PY       ?= /usr/bin/python3

CORE := $(RTL_DIR)/pbit_array.v $(RTL_DIR)/coupling_row.v $(RTL_DIR)/weight_mac.v \
        $(RTL_DIR)/pbit.v $(RTL_DIR)/sigmoid_lut.v $(RTL_DIR)/sigmoid_pwl.v \
        $(RTL_DIR)/noise_bank.v $(RTL_DIR)/lfsr.v $(RTL_DIR)/schedule_ctrl.v
SPARSE_CORE := $(RTL_DIR)/sparse_array.v $(RTL_DIR)/weight_mac.v $(RTL_DIR)/pbit.v \
        $(RTL_DIR)/sigmoid_lut.v $(RTL_DIR)/sigmoid_pwl.v $(RTL_DIR)/noise_bank.v \
        $(RTL_DIR)/lfsr.v $(RTL_DIR)/schedule_ctrl.v
TOP  := $(CORE) $(RTL_DIR)/readout_counter.v $(RTL_DIR)/uart_rx.v $(RTL_DIR)/uart_tx.v \
        $(RTL_DIR)/top.v $(RTL_DIR)/sparse_array.v $(RTL_DIR)/temper_ctrl.v
PGATES := pgate_not pgate_and pgate_or pgate_xor fulladder factor6 factor6_pt
CASES := ring8 ring8_q6 maxcut12 grid16 maxcut16 $(PGATES)
STAT_CASES := ring8 ring8_q6 maxcut12 grid16
STAT_SWEEPS ?= 200000
SPARSE_CASES := sp_ring8_p1 sp_ring8_p2 sp_ring8_p4 sp_maxcut12_p1 sp_maxcut12_p2 sp_maxcut12_p4 \
        sp_grid16_p1 sp_grid16_p2 sp_grid16_p4 sp_g12_ferro16_p1 sp_g12_ferro16_p4 \
        sp_g12_glass16_p1 sp_g12_glass16_p4 \
        sp_ring8_p4_tau sp_grid16_p4_tau sp_grid16_p4_share sp_grid16_p4_tsh

IV   := iverilog -g2005 -Wall
VLB  := verilator --binary --timing -O3 -Wno-fatal -Wno-lint -Wno-style

.PHONY: all test units arrays sparse stats stats-sparse top top-sparse denoise pgates lint vectors clean script noise train pmode

all: test stats stats-sparse top top-sparse denoise pgates script noise train pmode

test: vectors units arrays sparse lint

vectors:
	cd ../model && $(PY) export_vectors.py

$(BUILD):
	mkdir -p $(BUILD)

units: | $(BUILD)
	$(IV) -o $(BUILD)/tb_noise_bank.vvp tb_noise_bank.v $(RTL_DIR)/noise_bank.v $(RTL_DIR)/lfsr.v
	vvp -n $(BUILD)/tb_noise_bank.vvp
	$(IV) -o $(BUILD)/tb_sigmoid.vvp tb_sigmoid.v $(RTL_DIR)/sigmoid_lut.v $(RTL_DIR)/sigmoid_pwl.v
	vvp -n $(BUILD)/tb_sigmoid.vvp
	$(IV) -o $(BUILD)/tb_weight_mac.vvp tb_weight_mac.v $(RTL_DIR)/coupling_row.v $(RTL_DIR)/weight_mac.v
	vvp -n $(BUILD)/tb_weight_mac.vvp
	$(IV) -o $(BUILD)/tb_pbit.vvp tb_pbit.v $(RTL_DIR)/pbit.v $(RTL_DIR)/sigmoid_lut.v $(RTL_DIR)/sigmoid_pwl.v
	vvp -n $(BUILD)/tb_pbit.vvp
	$(IV) -o $(BUILD)/tb_schedule_ctrl.vvp tb_schedule_ctrl.v $(RTL_DIR)/schedule_ctrl.v
	vvp -n $(BUILD)/tb_schedule_ctrl.vvp
	$(IV) -o $(BUILD)/tb_readout_counter.vvp tb_readout_counter.v $(RTL_DIR)/readout_counter.v
	vvp -n $(BUILD)/tb_readout_counter.vvp
	$(IV) -P tb_readout_counter.UP_RAM=1 -o $(BUILD)/tb_readout_counter_ram.vvp tb_readout_counter.v $(RTL_DIR)/readout_counter.v
	vvp -n $(BUILD)/tb_readout_counter_ram.vvp
	$(IV) -o $(BUILD)/tb_uart.vvp tb_uart.v $(RTL_DIR)/uart_rx.v $(RTL_DIR)/uart_tx.v
	vvp -n $(BUILD)/tb_uart.vvp

arrays: | $(BUILD)
	@for c in $(CASES); do \
	  $(IV) -I $(VEC)/$$c -P "tb_pbit_array.CASE_DIR=\"$(VEC)/$$c/\"" \
	    -o $(BUILD)/tb_pbit_array_$$c.vvp tb_pbit_array.v $(CORE) || exit 1; \
	  vvp -n $(BUILD)/tb_pbit_array_$$c.vvp | grep -v finish || exit 1; \
	done
	$(IV) -I $(VEC)/ring8 -P "tb_pbit_array.CASE_DIR=\"$(VEC)/ring8/\"" -P tb_pbit_array.USE_PWL=1 -P 'tb_pbit_array.TRACE="trace_pwl.hex"' \
	  -o $(BUILD)/tb_pbit_array_pwl.vvp tb_pbit_array.v $(CORE)
	vvp -n $(BUILD)/tb_pbit_array_pwl.vvp | grep -v finish

sparse: | $(BUILD)
	@for c in $(SPARSE_CASES); do \
	  $(IV) -I $(VEC)/$$c -P "tb_sparse_array.CASE_DIR=\"$(VEC)/$$c/\"" \
	    -o $(BUILD)/tb_sparse_$$c.vvp tb_sparse_array.v $(SPARSE_CORE) || exit 1; \
	  vvp -n $(BUILD)/tb_sparse_$$c.vvp | grep -v finish || exit 1; \
	done

lint:
	verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-PROCASSINIT \
	  -I$(RTL_DIR) --top-module top $(TOP)
	verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-PROCASSINIT \
	  -I$(RTL_DIR) --top-module top -GSPARSE=1 -GN=256 -GIW=8 -GAW=13 -GP=4 -GD=12 -GDW=4 -GLW=6 -GROUNDS=64 -GUP_RAM=1 $(TOP)
	verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-PROCASSINIT \
	  -I$(RTL_DIR) --top-module top -GSPARSE=1 -GN=256 -GIW=8 -GAW=13 -GP=4 -GD=12 -GDW=4 -GLW=6 -GROUNDS=64 -GUP_RAM=1 -GNOISE_CTL=1 $(TOP)
	verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-PROCASSINIT \
	  -I$(RTL_DIR) --top-module top -GTEMPER_K=4 $(TOP)
	verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-PROCASSINIT \
	  -I$(RTL_DIR) --top-module pmode_top -GM=16 -GMW=4 $(RTL_DIR)/pmode_top.v $(RTL_DIR)/pmode_array.v \
	  $(RTL_DIR)/sigmoid_lut.v $(RTL_DIR)/noise_bank.v $(RTL_DIR)/lfsr.v
	@echo "lint: clean (verilator -Wall): top dense, dense TEMPER_K=4, SPARSE N=256 (+NOISE_CTL); pmode_top M=16"

# long runs: Verilator binary per case, then Python comparison
stats: | $(BUILD)
	@for c in $(STAT_CASES); do \
	  mkdir -p $(VL_BUILD)/$$c; \
	  $(VLB) -I$(VEC)/$$c -GCASE_DIR='"$(VEC)/'$$c'/"' --top-module tb_pbit_array \
	    -Mdir $(VL_BUILD)/$$c -o tb tb_pbit_array.v $(CORE) > $(BUILD)/vl_$$c.log 2>&1 || exit 1; \
	  $(VL_BUILD)/$$c/tb +sweeps=$(STAT_SWEEPS) +trace=$(BUILD)/$${c}_rtl_trace.hex | grep -v -E "finish|S i m|Verilator: cpu" || exit 1; \
	done
	$(PY) compare.py --sweeps $(STAT_SWEEPS) --cases $(STAT_CASES) --out $(BUILD)/stats.json
	$(PY) plot_hist.py --stats $(BUILD)/stats.json --out $(BUILD)

SPARSE_STAT_SMALL := sp_ring8_p4 sp_maxcut12_p4 sp_grid16_p4
SPARSE_STAT_BIG   := sp_g12_ferro16_p4 sp_g12_glass16_p4
SPARSE_BIG_SWEEPS ?= 20000
stats-sparse: | $(BUILD)
	@for c in $(SPARSE_STAT_SMALL) $(SPARSE_STAT_BIG); do \
	  case " $(SPARSE_STAT_BIG) " in *" $$c "*) sw=$(SPARSE_BIG_SWEEPS);; *) sw=$(STAT_SWEEPS);; esac; \
	  mkdir -p $(VL_BUILD)/$$c; \
	  $(VLB) -I$(VEC)/$$c -GCASE_DIR='"$(VEC)/'$$c'/"' --top-module tb_sparse_array \
	    -Mdir $(VL_BUILD)/$$c -o tb tb_sparse_array.v $(SPARSE_CORE) > $(BUILD)/vl_$$c.log 2>&1 || exit 1; \
	  $(VL_BUILD)/$$c/tb +sweeps=$$sw +trace=$(BUILD)/$${c}_rtl_trace.hex | grep -v -E "finish|S i m|Verilator: cpu" || exit 1; \
	done
	$(PY) compare.py --sparse-stats $(SPARSE_STAT_SMALL) $(SPARSE_STAT_BIG) --out $(BUILD)/sparse_stats.json

top: | $(BUILD)
	mkdir -p $(VL_BUILD)/top
	$(VLB) --top-module tb_top -Mdir $(VL_BUILD)/top -o tb tb_top.v $(TOP) > $(BUILD)/vl_top.log 2>&1
	$(VL_BUILD)/top/tb +dump=$(BUILD)/top_uart_dump.txt | grep -v -E "S i m|Verilator: cpu"
	$(PY) compare.py --top-dump $(BUILD)/top_uart_dump.txt

top-sparse: | $(BUILD)
	mkdir -p $(VL_BUILD)/top_sparse
	$(VLB) -I$(VEC)/sp_g12_glass16_p4 --top-module tb_top_sparse -Mdir $(VL_BUILD)/top_sparse -o tb \
	  tb_top_sparse.v $(TOP) > $(BUILD)/vl_top_sparse.log 2>&1
	$(VL_BUILD)/top_sparse/tb +dump=$(BUILD)/top_sparse_uart_dump.txt | grep -v -E "S i m|Verilator: cpu"
	$(PY) compare.py --top-sparse-dump $(BUILD)/top_sparse_uart_dump.txt

denoise: | $(BUILD)
	cd ../model && $(PY) denoise.py --export
	$(IV) -o $(BUILD)/tb_denoise.vvp tb_denoise.v $(RTL_DIR)/denoise_demo.v $(CORE)
	vvp -n $(BUILD)/tb_denoise.vvp | grep -v finish

pgates: | $(BUILD)
	@for c in $(PGATES); do \
	  $(IV) -I $(VEC)/$$c -P "tb_pgates.CASE_DIR=\"$(VEC)/$$c/\"" \
	    -o $(BUILD)/tb_pgates_$$c.vvp tb_pgates.v $(RTL_DIR)/pgates.v $(RTL_DIR)/temper_ctrl.v $(CORE) || exit 1; \
	  vvp -n $(BUILD)/tb_pgates_$$c.vvp +dump=$(BUILD)/pgate_$${c}_dump.hex | grep -v finish || exit 1; \
	done
	$(PY) compare.py --pgates $(PGATES) --out $(BUILD)/pgates.json
	$(PY) plot_pgates.py --stats $(BUILD)/pgates.json --out $(BUILD)

# noise study: long RTL runs with RNG persistence / shared noise + model q-scan + raw-noise entropy
NOISE_RTL := sp_grid16_p4_tau sp_grid16_p4_share
NOISE_SWEEPS ?= 200000
noise: | $(BUILD)
	@for c in $(NOISE_RTL); do \
	  mkdir -p $(VL_BUILD)/$$c; \
	  $(VLB) -I$(VEC)/$$c -GCASE_DIR='"$(VEC)/'$$c'/"' --top-module tb_sparse_array \
	    -Mdir $(VL_BUILD)/$$c -o tb tb_sparse_array.v $(SPARSE_CORE) > $(BUILD)/vl_$$c.log 2>&1 || exit 1; \
	  $(VL_BUILD)/$$c/tb +sweeps=$(NOISE_SWEEPS) +trace=$(BUILD)/$${c}_rtl_trace.hex | grep -v -E "finish|S i m|Verilator: cpu" || exit 1; \
	done
	$(PY) noise_study.py --sweeps $(NOISE_SWEEPS) --out $(BUILD)/noise.json --rtl \
	  $(foreach c,$(NOISE_RTL),$(c)=$(BUILD)/$(c)_rtl_trace.hex)
	$(MAKE) script SCRIPTS="noise_dense noise_sparse"
	$(PY) ../fpga/entropy.py --dump $(BUILD)/script_noise_dense/dump.hex --skip 3 --count 300 --lfsr-check

# learning loop: demos on the bit-exact model chip, then 3 recorded training epochs replayed on the RTL
TRAIN_RTL := $(BUILD)/script_train_xor_h/
train: | $(BUILD)
	$(PY) ../model/train.py --out $(BUILD)/train.json
	$(PY) ../model/train.py --rtl-script $(TRAIN_RTL) --rtl-epochs 3
	mkdir -p $(VL_BUILD)/script_train
	$(VLB) -I$(TRAIN_RTL) -GSCRIPT_DIR='"$(TRAIN_RTL)"' --top-module tb_top_script -Mdir $(VL_BUILD)/script_train -o tb \
	  tb_top_script.v $(TOP) > $(BUILD)/vl_script_train.log 2>&1
	$(VL_BUILD)/script_train/tb | grep -v -E "S i m|Verilator: cpu|finish"
	$(PY) ../model/uart_script.py check train_xor_h --dir $(TRAIN_RTL)

# continuous p-modes (Langevin soft spins): bit-exact, then long runs vs exact densities
PMODE_CORE  := $(RTL_DIR)/pmode_array.v $(RTL_DIR)/sigmoid_lut.v $(RTL_DIR)/noise_bank.v $(RTL_DIR)/lfsr.v
PM_CASES    := well tilt pair chain4
PM_STAT     := well tilt pair
PM_STEPS    ?= 400000
pmode: | $(BUILD)
	@for c in $(PM_CASES); do \
	  $(IV) -I $(VEC)/pm_$$c -P "tb_pmode.CASE_DIR=\"$(VEC)/pm_$$c/\"" \
	    -o $(BUILD)/tb_pmode_$$c.vvp tb_pmode.v $(PMODE_CORE) || exit 1; \
	  vvp -n $(BUILD)/tb_pmode_$$c.vvp | grep -v finish || exit 1; \
	done
	@for c in $(PM_STAT); do \
	  mkdir -p $(VL_BUILD)/pm_$$c; \
	  $(VLB) -I$(VEC)/pm_$$c -GCASE_DIR='"$(VEC)/pm_'$$c'/"' --top-module tb_pmode \
	    -Mdir $(VL_BUILD)/pm_$$c -o tb tb_pmode.v $(PMODE_CORE) > $(BUILD)/vl_pm_$$c.log 2>&1 || exit 1; \
	  $(VL_BUILD)/pm_$$c/tb +steps=$(PM_STEPS) +trace=$(BUILD)/pm_$${c}_rtl_trace.hex | grep -v -E "finish|S i m|Verilator: cpu" || exit 1; \
	done
	$(PY) pmode_study.py --out $(BUILD)/pmode.json --rtl $(foreach c,$(PM_STAT),pm_$(c)=$(BUILD)/pm_$(c)_rtl_trace.hex)

# generic UART sessions: model/uart_script.py writes script + expected bytes, tb_top_script plays them
SCRIPTS ?= noise_dense noise_sparse temper_dense
script: | $(BUILD)
	@for s in $(SCRIPTS); do \
	  d=$(BUILD)/script_$$s/; \
	  $(PY) ../model/uart_script.py write $$s --dir $$d || exit 1; \
	  mkdir -p $(VL_BUILD)/script_$$s; \
	  $(VLB) -I$$d -GSCRIPT_DIR='"'$$d'"' --top-module tb_top_script -Mdir $(VL_BUILD)/script_$$s -o tb \
	    tb_top_script.v $(TOP) > $(BUILD)/vl_script_$$s.log 2>&1 || { tail -20 $(BUILD)/vl_script_$$s.log; exit 1; }; \
	  $(VL_BUILD)/script_$$s/tb | grep -v -E "S i m|Verilator: cpu|finish" || exit 1; \
	  $(PY) ../model/uart_script.py check $$s --dir $$d || exit 1; \
	done

clean:
	rm -rf $(BUILD) $(VL_BUILD)
