#!/usr/bin/env python3
"""Build the Verilator byte-pipe simulator of one chip (rtl/top.v + web/rtl/harness.cpp).

  python3 web/build_rtl.py dense16        -> /tmp/pbit_vl_$USER/web_dense16/Vtop

Verilator cannot build in a path containing spaces, so the object directory
lives in /tmp and the harness is copied there. The binary must run with cwd =
web/ so that the $readmemh paths ("../model/vectors/...") resolve.
"""
import getpass
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chips  # noqa: E402

RTL = ["top.v", "pbit_array.v", "coupling_row.v", "weight_mac.v", "pbit.v", "sigmoid_lut.v", "sigmoid_pwl.v",
       "noise_bank.v", "lfsr.v", "schedule_ctrl.v", "readout_counter.v", "uart_rx.v", "uart_tx.v",
       "sparse_array.v", "temper_ctrl.v"]
OSS = os.path.expanduser("~/tools/oss-cad-suite")


def mdir(chip_id):
    return f"/tmp/pbit_vl_{getpass.getuser()}/web_{chip_id}"


def binary(chip_id):
    return os.path.join(mdir(chip_id), "Vtop")


def env():
    e = dict(os.environ)
    if shutil.which("verilator") is None and os.path.isdir(OSS):
        e["PATH"] = f"{OSS}/bin:{OSS}/py3bin:" + e.get("PATH", "")
        e["VERILATOR_ROOT"] = e.get("VERILATOR_ROOT", f"{OSS}/share/verilator")
    return e


def gflag(k, v):
    return f'-G{k}="{v}"' if isinstance(v, str) else f"-G{k}={int(v)}"


def build(chip_id, log=print):
    chip = chips.BY_ID[chip_id]
    if "rtl" not in chip["backends"]:
        raise ValueError(f"{chip_id}: {chip.get('note', 'no RTL backend')}")
    d = mdir(chip_id)
    # always a clean build: Verilator's dependency files hold the RTL paths relative
    # to web/ (the build runs make inside /tmp), so an incremental rebuild cannot find them
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    harness = os.path.join(d, "harness.cpp")
    shutil.copy(os.path.join(HERE, "rtl", "harness.cpp"), harness)
    params = chips.verilator_params(chip)
    cmd = ["verilator", "--cc", "--exe", "--build", "-O3", "-j", "0", "-Wno-fatal", "-Wno-lint", "-Wno-style",
           "--top-module", "top", "--Mdir", d, "-o", "Vtop", "-CFLAGS", "-O2 -DCPB=8",
           *[gflag(k, v) for k, v in params.items()], harness, *[os.path.join("..", "rtl", f) for f in RTL]]
    log(f"verilator: building {chip_id} into {d} (about 30-60 s)")
    r = subprocess.run(cmd, cwd=HERE, env=env(), capture_output=True, text=True)
    with open(os.path.join(d, "build.log"), "w") as f:
        f.write(" ".join(cmd) + "\n\n" + r.stdout + r.stderr)
    if r.returncode != 0 or not os.path.exists(binary(chip_id)):
        tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
        raise RuntimeError(f"verilator build of {chip_id} failed (see {d}/build.log):\n" + "\n".join(tail))
    with open(os.path.join(d, "params.txt"), "w") as f:
        f.write(repr(sorted(params.items())))
    return binary(chip_id)


def ensure(chip_id, log=print):
    """Build if missing or if the parameters / sources changed."""
    b = binary(chip_id)
    stamp = os.path.join(mdir(chip_id), "params.txt")
    params = repr(sorted(chips.verilator_params(chips.BY_ID[chip_id]).items()))
    if os.path.exists(b) and os.path.exists(stamp) and open(stamp).read() == params:
        t = os.path.getmtime(b)
        srcs = [os.path.join(HERE, "..", "rtl", f) for f in RTL] + [os.path.join(HERE, "rtl", "harness.cpp")]
        if all(os.path.getmtime(s) <= t for s in srcs):
            return b
    return build(chip_id, log)


if __name__ == "__main__":
    for cid in sys.argv[1:] or [c["id"] for c in chips.CHIPS if "rtl" in c["backends"]]:
        print(ensure(cid))
