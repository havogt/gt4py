"""Extract the series and tables the report needs from the Santis JSONL rows -> report_data.json."""

import glob, json, os, re, statistics as st, collections, sys

RES = sys.argv[1] if len(sys.argv) > 1 else "."
FILE_RE = re.compile(r"^(?P<kind>[a-z0-9]+)_(?P<run>single|multi\d+)_(?P<job>\d+)\.jsonl$")


def load_all():
    # merge every file of a (kind, run) in job-id order; the last row of a case wins
    files = collections.defaultdict(list)
    for p in glob.glob(os.path.join(RES, "*.jsonl")):
        m = FILE_RE.match(os.path.basename(p))
        if m:
            files[(m["kind"], m["run"])].append((int(m["job"]), p))
    rows = []
    for (kind, run), fl in sorted(files.items()):
        dedup = {}
        for job, p in sorted(fl):
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                r["kind"], r["run"], r["proc"] = (
                    kind,
                    run,
                    ("single" if run == "single" else "multi"),
                )
                key = (
                    r.get("layout"),
                    r.get("transport"),
                    r.get("mode"),
                    r.get("size"),
                    r.get("weak"),
                    r.get("n_steps"),
                )
                dedup[key] = r
        rows.extend(dedup.values())
    return rows


def stats(r):
    s = r.get("samples_step_ms")
    if s:
        s = sorted(s)
        n = len(s)
        q = lambda f: s[min(n - 1, int(round(f * (n - 1))))]
        return {"med": st.median(s), "q25": q(0.25), "q75": q(0.75), "min": s[0], "n": n}
    return {
        "med": r["per_step_ms"],
        "q25": r["per_step_ms"],
        "q75": r["per_step_ms"],
        "min": r["per_step_ms"],
        "n": 1,
    }


def P_of(layout):
    a, b = layout.split("x")
    return int(a) * int(b)


rows = load_all()
ok = [r for r in rows if r.get("status") == "ok"]
# merge remat kinds into strong/weak
for r in ok:
    if r["kind"] == "rstrong":
        r["kind"] = "strong"
    if r["kind"] == "rweak":
        r["kind"] = "weak"


def sel(kind, proc, layout, transport, mode, weak=None):
    out = {}
    for r in ok:
        if (
            r["kind"] != kind
            or r["proc"] != proc
            or r["layout"] != layout
            or r["transport"] != transport
            or r["mode"] != mode
        ):
            continue
        if weak is not None and bool(r.get("weak")) != weak:
            continue
        out[r["size"]] = stats(r)
    return out


TR = ["padded", "coloured8", "coloured2ph", "allgather", "ragged"]
RUNS = [
    ("single", "1x1", 1),
    ("multi", "2x1", 2),
    ("multi", "2x2", 4),
    ("multi", "4x1", 4),
    ("single", "2x2", 4),
    ("single", "4x1", 4),
    ("multi", "8x1", 8),
    ("multi", "4x2", 8),
    ("multi", "16x1", 16),
    ("multi", "4x4", 16),
    ("multi", "32x1", 32),
    ("multi", "8x4", 32),
]
data = {"runs": [f"{p}:{l}" for p, l, _ in RUNS], "P": {f"{p}:{l}": P for p, l, P in RUNS}}

data["strong"] = {
    m: {t: {f"{p}:{l}": sel("strong", p, l, t, m) for p, l, _ in RUNS} for t in TR}
    for m in ["fwd", "grad", "grad_remat"]
}
data["ref"] = {
    m: sel("strong", "single", "1x1", None, m) for m in ["ref", "ref_grad", "ref_grad_remat"]
}
data["weak"] = {
    m: {t: {f"{p}:{l}": sel("weak", p, l, t, m) for p, l, _ in RUNS} for t in TR}
    for m in ["fwd", "grad", "grad_remat"]
}
data["weak_ref"] = {
    m: sel("weak", "single", "1x1", None, m) for m in ["ref", "ref_grad", "ref_grad_remat"]
}
data["exch"] = {
    m: {t: {f"{p}:{l}": sel("exch", p, l, t, m) for p, l, _ in RUNS} for t in TR}
    for m in ["exch", "exch_grad"]
}

# OOM boundaries (first oom size per (kind, proc, layout, transport, mode)), from all rows incl. non-ok
oom = collections.defaultdict(list)
for r in rows:
    if r.get("status") == "oom":
        k = r["kind"].replace("rstrong", "strong").replace("rweak", "weak")
        oom[f"{k}|{r['proc']}|{r['layout']}|{r['transport']}|{r['mode']}"].append(r["size"])
data["oom"] = {k: min(v) for k, v in oom.items()}

# memory: per-case peak when the cumulative peak rose
mem = {}
for r in ok:
    if r.get("peak_mem_bytes_delta") and r["peak_mem_bytes_delta"] > 0 and r["kind"] in ("strong",):
        mem[f"{r['proc']}|{r['layout']}|{r['transport']}|{r['mode']}|{r['size']}"] = (
            r["peak_mem_cumulative_bytes"] / 2**30
        )
data["mem_gb"] = mem

# roofline rows (xla estimate) and achieved
roof = {}
for r in ok:
    if r["kind"] == "roof" and r.get("xla_bytes_accessed"):
        roof[f"{r['layout']}|{r['transport']}|{r['mode']}|{r['size']}"] = {
            "bytes_per_cell_step": r["xla_bytes_accessed"] / (r["MLOC"] * r["NLOC"]) / r["n_steps"],
            "kernels": r["xla_kernels"],
            "gbps": r["achieved_GBps"],
            "med": stats(r)["med"],
        }
data["roof"] = roof

# aux: lin, prealloc, battery, repeat, scanstep, pgrad
data["lin"] = {
    f"{r['n_steps']}|{r['transport']}|{r['mode']}|{r['size']}": stats(r)["med"]
    for r in ok
    if r["kind"] in ("lin10", "lin40")
}
data["prealloc"] = {
    f"{r['transport']}|{r['mode']}|{r['size']}": r.get("status")
    for r in rows
    if r["kind"] == "prealloc"
}
data["pgrad"] = {
    f"{r['proc']}|{r['layout']}|{r['transport']}|{r['mode']}|{r['size']}": (
        stats(r)["med"] if r.get("status") == "ok" else r.get("status")
    )
    for r in rows
    if r["kind"] == "pgrad"
}
data["scanstep"] = {
    f"{r['n_steps']}|{r['transport']}|{r['mode']}|{r['size']}": stats(r)
    for r in ok
    if r["kind"] == "scanstep"
}
data["repeat32"] = {
    f"{r['layout']}|{r['transport']}|{r['mode']}|{r['size']}": stats(r)["med"]
    for r in ok
    if r["kind"] == "repeat"
}
batt = {}
for r in rows:
    if r["kind"] == "battery2048":
        batt[r["transport"]] = {
            k: r[k]
            for k in r
            if k.startswith("T")
            and (
                "pass" in k
                or "status" in k
                or k
                in ("T5_max_rel_diff_common", "T6_rate2_last", "T4_max_abs_diff", "T1_max_abs_diff")
            )
        }
        batt[r["transport"]]["all_pass"] = r.get("all_pass")
data["battery2048"] = batt
env = next((r for r in ok if r["kind"] == "strong" and r["proc"] == "multi"), None)
data["env"] = {
    k: env[k]
    for k in ("jax", "python", "device_kind", "shard_map_api", "gt4py_tracer_dispatch")
    if env
}
data["n_rows_ok"] = len(ok)
data["n_rows"] = len(rows)
json.dump(data, open("report_data.json", "w"))
print("rows", len(rows), "ok", len(ok), "files", len({(r["kind"], r["run"]) for r in rows}))
print("padded fwd 1x1 sizes:", sorted(data["strong"]["fwd"]["padded"]["single:1x1"]))
print("padded fwd multi:32x1 16384:", data["strong"]["fwd"]["padded"]["multi:32x1"].get(16384))
print("oom sample:", [k for k in data["oom"] if "padded|grad" in k][:6])
print(
    "mem sample:",
    [
        (k, round(v, 2))
        for k, v in data["mem_gb"].items()
        if k.startswith("single|1x1|padded|grad_remat")
    ],
)
