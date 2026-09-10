"""ncu metrics CSVs -> profile/roofline.json: per kernel (per step) time, DRAM bytes, FP64 flops."""

import csv, collections, glob, json, os, re, sys

D = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/vogtha/claude/gt4py_autodiff/tmp/santis_results/roofline"
)
CALLS = 3  # 1 warm-up + 2 timed calls of one step


def num(v):
    return float(v.replace(",", "").replace("'", ""))


out = {}
for f in sorted(glob.glob(os.path.join(D, "ncu_*.csv"))):
    m = re.match(r"ncu_(.+)_(\d+)\.csv", os.path.basename(f))
    mode, size = m.group(1), int(m.group(2))
    rows = [
        r for r in csv.reader(open(f)) if r and r[0] not in ("ID",) and not r[0].startswith("==")
    ]
    hdr = [r for r in csv.reader(open(f)) if r and r[0] == "ID"][0]
    i = {h: k for k, h in enumerate(hdr)}
    per = collections.defaultdict(dict)
    for r in rows:
        if len(r) <= i["Metric Value"]:
            continue
        per[(r[i["ID"]], r[i["Kernel Name"]])][r[i["Metric Name"]]] = (
            num(r[i["Metric Value"]]),
            r[i["Metric Unit"]],
        )

    def val(
        d,
        k,
        scale={
            "ns": 1e-9,
            "us": 1e-6,
            "ms": 1e-3,
            "s": 1,
            "nsecond": 1e-9,
            "usecond": 1e-6,
            "msecond": 1e-3,
            "second": 1,
            "byte": 1,
            "Kbyte": 1e3,
            "Mbyte": 1e6,
            "Gbyte": 1e9,
            "inst": 1,
            "Kinst": 1e3,
            "Minst": 1e6,
            "Ginst": 1e9,
        },
    ):
        v, u = d.get(k, (0.0, ""))
        return v * scale.get(u, 1)

    kern = collections.defaultdict(
        lambda: {"n": 0, "s": 0.0, "bytes": 0.0, "flops": 0.0, "f32": 0.0, "l2": 0.0}
    )
    for (kid, name), d in per.items():
        k = kern[name.split("(")[0][:48]]
        k["n"] += 1
        k["s"] += val(d, "gpu__time_duration.sum")
        k["bytes"] += val(d, "dram__bytes.sum")
        k["l2"] += val(d, "lts__t_bytes.sum")
        k["flops"] += (
            val(d, "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum")
            + val(d, "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum")
            + 2 * val(d, "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum")
        )
        k["f32"] += (
            val(d, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
            + val(d, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
            + 2 * val(d, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
        )
    ks = []
    for name, k in kern.items():
        ks.append(
            {
                "kernel": name,
                "launches_per_step": k["n"] / CALLS,
                "s_per_step": k["s"] / CALLS,
                "bytes_per_step": k["bytes"] / CALLS,
                "l2_bytes_per_step": k["l2"] / CALLS,
                "flops_per_step": k["flops"] / CALLS,
                "f32_per_step": k["f32"] / CALLS,
            }
        )
    tot = {
        "s": sum(k["s_per_step"] for k in ks),
        "bytes": sum(k["bytes_per_step"] for k in ks),
        "flops": sum(k["flops_per_step"] for k in ks),
    }
    out[f"{mode}|{size}"] = {"kernels": sorted(ks, key=lambda k: -k["s_per_step"]), "total": tot}
    print(
        f"{mode:16s} {size:6d}: {len(ks):2d} kernels, {1e3 * tot['s']:.2f} ms/step, {tot['bytes'] / 1e9:.1f} GB/step, {tot['flops'] / 1e9:.1f} GFLOP/step, AI {tot['flops'] / max(tot['bytes'], 1):.2f} F/B, {tot['flops'] / max(tot['s'], 1e-9) / 1e12:.2f} TFLOP/s, {tot['bytes'] / max(tot['s'], 1e-9) / 1e12:.2f} TB/s"
    )
os.makedirs("/home/vogtha/claude/gt4py_autodiff/tmp/santis_results/profile", exist_ok=True)
json.dump(
    out,
    open("/home/vogtha/claude/gt4py_autodiff/tmp/santis_results/profile/roofline.json", "w"),
    indent=1,
)
