import json, datetime
from build_report import *

N = json.load(open("prose_numbers.json"))
today = "10 September 2026"
ncu_ref, ncu_fwd = ncu_bars()
ROOF_SVG, ROOF = roofline_chart()
def _roof_block():
    if not ROOF: return ""
    r, f, g = ROOF.get("ref|16384"), ROOF.get("fwd|16384"), ROOF.get("ref_grad_remat|16384")
    small = ROOF.get("ref|2048")
    cap = ("Roofline of one GPU (Nsight Compute, one model step per call). Small marks are individual kernels, large marks the whole step: flop count over DRAM bytes on the x axis, attained double-precision rate on the y axis. "
           "The dashed roof is 4 TB/s of HBM3 up to the ridge at 8.5 flop per byte, then the 34 TFLOP/s FP64 peak. Every kernel sits on the bandwidth slope at 0.1–1 flop per byte; nothing is near the compute roof.")
    txt = f"""<p>The roofline places every kernel where the per-kernel throughput numbers said it would be: on the memory slope, one to two orders of magnitude in intensity below the ridge point. The reference step at 16384² performs {r['GF']:.1f} GFLOP over {r['GB']:.1f} GB of DRAM traffic, {r['ai']:.2f} flop per byte, and attains {r['tflops']:.2f} TFLOP/s, which is {100*r['tbps']/HBM_TBPS:.0f}% of HBM peak and {100*r['tflops']/FP64_TFLOPS:.1f}% of FP64 peak. The sharded padded step moves {f['GB']:.1f} GB for the same {f['GF']:.1f} GFLOP ({f['ai']:.2f} flop per byte), the pad and select kernels being pure traffic, and the reference gradient step {g['GB']:.1f} GB for {g['GF']:.1f} GFLOP ({g['ai']:.2f} flop per byte). Per cell and step the reference step moves {r['GB']*1e9/16384**2:.0f} bytes, {r['GB']*1e9/16384**2/8:.0f} double-precision accesses, for {r['GF']*1e9/16384**2:.0f} flops.""" + (f""" At 2048² the same reference step reaches only {100*small['tbps']/HBM_TBPS:.0f}% of HBM peak: the kernels are too short to fill the machine, which is the launch-floor regime of the scaling tables.</p>""" if small else "</p>")
    return fig(ROOF_SVG, cap) + txt
ROOF_BLOCK = _roof_block()

def sec(id_, title, body, eyebrow=None):
    eb = f'<p class="eyebrow">{eyebrow}</p>' if eyebrow else ""
    return f'<section id="{id_}">{eb}<h2>{title}</h2>{body}</section>'

CSS = """
:root{--paper:#f5f6f3;--surface:#ffffff;--ink:#171d1c;--ink-2:#4a5553;--ink-3:#7d8886;--rule:#d9ddd9;--accent:#0f5c5a;--accent-soft:#e3efee;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;
--o1:#86b6ef;--o2:#5598e7;--o3:#2a78d6;--o4:#1c5cab;--o5:#104281;--o6:#0d366b;--w1:#f2a97f;--w2:#eb6834;--w3:#c24a1b;--w4:#8a3311;--bad:#d03b3b;--good:#0ca30c;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--paper:#141817;--surface:#1c2120;--ink:#eef1ee;--ink-2:#b8c0bd;--ink-3:#869190;--rule:#2e3634;--accent:#5ec2bd;--accent-soft:#1d2f2e;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--o1:#b7d3f6;--o2:#86b6ef;--o3:#5598e7;--o4:#3987e5;--o5:#256abf;--o6:#184f95;--w1:#f7c9ad;--w2:#f0975f;--w3:#d95926;--w4:#a8431a;color-scheme:dark}}
:root[data-theme="dark"]{--paper:#141817;--surface:#1c2120;--ink:#eef1ee;--ink-2:#b8c0bd;--ink-3:#869190;--rule:#2e3634;--accent:#5ec2bd;--accent-soft:#1d2f2e;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--o1:#b7d3f6;--o2:#86b6ef;--o3:#5598e7;--o4:#3987e5;--o5:#256abf;--o6:#184f95;--w1:#f7c9ad;--w2:#f0975f;--w3:#d95926;--w4:#a8431a;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:17px;line-height:1.55;font-variant-numeric:tabular-nums}
main{max-width:940px;margin:0 auto;padding:40px 24px 80px}
h1,h2,h3{font-family:"Archivo","Helvetica Neue",Arial,sans-serif;text-wrap:balance;letter-spacing:-0.01em}
h1{font-size:40px;line-height:1.1;margin:8px 0 12px;font-weight:700}
h2{font-size:26px;margin:56px 0 12px;font-weight:600;color:var(--ink)}
h3{font-size:19px;margin:28px 0 8px;font-weight:600}
p,li{max-width:70ch}
.eyebrow{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:var(--accent);margin:0}
.lede{font-size:20px;color:var(--ink-2);max-width:66ch}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px 24px;margin:24px 0 8px;padding:16px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);font-size:14px;color:var(--ink-2)}
.meta b{display:block;color:var(--ink);font-weight:600}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:28px 0}
.stat{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 16px}
.stat .n{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:28px;font-weight:600;color:var(--accent);line-height:1.1}
.stat .l{font-size:14px;color:var(--ink-2);margin-top:6px}
.key{background:var(--accent-soft);border-left:3px solid var(--accent);padding:14px 18px;border-radius:0 6px 6px 0;margin:20px 0}
.key ul{margin:6px 0 0;padding-left:20px}
.key li{margin:6px 0}
figure{margin:28px 0;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 12px 10px}
figure svg{width:100%;height:auto;display:block}
figcaption{font-size:14px;color:var(--ink-2);margin:8px 6px 0;max-width:none}
.chart .ct{font:600 14px "Archivo",sans-serif;fill:var(--ink)}
.chart .tk{font:12px "JetBrains Mono",monospace;fill:var(--ink-2)}
.chart .al{font:12px "Source Sans 3",sans-serif;fill:var(--ink-2)}
.chart .dl{font:12px "Source Sans 3",sans-serif}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .ax{stroke:var(--ink-3);stroke-width:1}
.chart .pt:hover circle,.chart .pt:hover rect{stroke:var(--ink);stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2);margin:8px 6px 0}
.legend i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px;border-top:3px solid transparent;box-sizing:content-box;height:0}
.tw{overflow-x:auto;margin:16px 0}
table.data{border-collapse:collapse;font-size:14px;min-width:100%}
table.data th{text-align:left;font-weight:600;color:var(--ink-2);border-bottom:2px solid var(--rule);padding:6px 10px;white-space:nowrap}
table.data td{padding:6px 10px;border-bottom:1px solid var(--rule);font-family:"JetBrains Mono",ui-monospace,monospace;font-size:13px;white-space:nowrap}
table.data td:first-child{font-family:"Source Sans 3",sans-serif;font-size:14px}
.mut{color:var(--ink-3)}
code{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:0.9em;background:var(--surface);border:1px solid var(--rule);border-radius:3px;padding:0 4px}
.note{font-size:15px;color:var(--ink-2);border-top:1px solid var(--rule);padding-top:8px;max-width:70ch}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:760px){.two{grid-template-columns:1fr}h1{font-size:32px}}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
"""

HEAD = f"""<title>Halo-Exchange AD on Santis</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
"""

# ---------- prose ----------
intro = f"""
<main>
<p class="eyebrow">GT4Py · JAX automatic differentiation · CSCS Alps</p>
<h1>Halo-Exchange AD on Santis</h1>
<p class="lede">How a GT4Py shallow-water model, sharded with <code>jax.shard_map</code> and differentiated with reverse-mode JAX, scales from one GH200 to 32 across eight nodes, for the forward step, its gradient, and five halo-exchange transports.</p>
<div class="meta">
<div><b>Machine</b>Santis (CSCS Alps), NVIDIA GH200, 96 GB HBM3 per GPU, 4 GPUs per node, Slingshot-11</div>
<div><b>Software</b>jax {env.get('jax')} (CUDA 13 wheels), Python {env.get('python')}, uenv icon/26.7:v1, gt4py branch <code>ad_halo</code></div>
<div><b>Model</b>Sadourny shallow water, float64, periodic, 20 time steps per measurement, halo width 1</div>
<div><b>Measurement</b>20 timed calls per case after one compile call; medians with interquartile range</div>
<div><b>Date</b>{today}</div>
</div>

<div class="stats">
<div class="stat"><div class="n">{N['sp_c32']:.0f}×</div><div class="l">forward speed-up on 32 GPUs at 16384², coloured 2-phase transport ({100*N['sp_c32']/32:.0f}% efficiency)</div></div>
<div class="stat"><div class="n">{N['remat_vs_grad_2048']:.2f}×</div><div class="l">rematerialised gradient relative to the stored-residual gradient at 2048² on one GPU: recompute is cheaper than the tape</div></div>
<div class="stat"><div class="n">73–92%</div><div class="l">of HBM peak bandwidth reached by every kernel of the reference step at 16384² (ncu); the step is memory-bound</div></div>
<div class="stat"><div class="n">1.4–1.7×</div><div class="l">per-step cost of the 20-step <code>lax.scan</code> relative to one unrolled step at ≥4096² (reference 1.6–1.7×, sharded forward 1.5–1.6×, gradient 1.4×)</div></div>
</div>

<div class="key"><strong>What the study established</strong>
<ul>
<li><b>The sharded model is correct at scale.</b> At 2048² per GPU on 32 GPUs the padded, coloured and all-gather transports reproduce the single-GPU forward bit for bit, the gradient agrees with the reference to 1.4×10⁻¹³ relative, and the Taylor remainder converges at second order.</li>
<li><b>Forward strong scaling is near-ideal to 8 GPUs and holds to 32 for the right transport.</b> At 16384² the coloured 2-phase transport reaches {N['sp_c32']:.1f}× on 32 GPUs; padded reaches {N['sp_p32']:.1f}× on the 32×1 strip and {base16/med('fwd','padded','multi:8x4',16384):.1f}× on the 8×4 block, because its dense all-to-all sends a full-size slot to every peer: its wire volume grows with the GPU count and with the row length, while the coloured transports move exactly the halo.</li>
<li><b>The gradient scales worse than the forward.</b> With per-step rematerialisation the padded gradient at 8192² reaches {N['gr1']/N['gr32']:.1f}× on 32 GPUs ({100*N['gr1']/N['gr32']/32:.0f}% efficiency); the coloured 2-phase transport {N['grc1']/N['grc32']:.1f}× ({100*N['grc1']/N['grc32']/32:.0f}%). The reverse pass doubles the exchanges per step, and each transpose carries the same padded volume as its forward.</li>
<li><b>Rematerialisation is the right default for the time loop.</b> Storing 20 steps of residuals needs about 35 GiB at 4096² per GPU and does not fit at 8192²; recomputing the step costs nothing at large sizes because the recompute is cheaper than streaming the tape back from HBM.</li>
<li><b>Communication is a minor share until the blocks get thin.</b> One exchange is {100*share('multi:2x2', 8192, 'exch', 'fwd'):.0f}% of a forward step and one exchange plus adjoint {100*share('multi:2x2', 8192, 'exch_grad', 'grad_remat'):.0f}% of a gradient step at 4096² per GPU on 4 GPUs; at 512 rows per GPU on 32 GPUs the shares are {100*share('multi:32x1', 16384, 'exch', 'fwd'):.0f}% and {100*share('multi:32x1', 16384, 'exch_grad', 'grad_remat'):.0f}%. The rest is the stencil's own memory traffic.</li>
<li><b>Two defects found on the way.</b> The native <code>ragged_all_to_all</code> transport has a bit-exact forward and a wrong gradient on every GPU count; and the single-process, four-device gradient hangs instead of reporting out-of-memory at 4096² per GPU, where the four-process run completes.</li>
</ul></div>
"""

setup = f"""
<h3>The model</h3>
<p>The code under test is the GT4Py shallow-water example on the <code>ad_halo</code> branch: Sadourny's energy-conserving scheme on an Arakawa C grid, the same test case as the classic Fortran <em>swm</em> benchmark. Three prognostic fields, the velocity components <em>u</em>, <em>v</em> and the height <em>p</em>, are advanced with leapfrog time stepping and a Robert–Asselin filter (α = 0.001); the first step is a forward Euler step of Δt, every later step a leapfrog step of 2Δt. Grid spacing is 100 km in both directions, Δt = 90 s, and the initial state is the analytic stream function ψ = a·sin·sin with a = 10⁶ m² s⁻¹ and a height field of 5×10⁴ plus a cosine perturbation, evaluated on whatever global grid M×N a run uses. Every field is float64. The step is written in GT4Py's field view (<code>operators.timestep</code>: staggered averages and differences; four intermediate fields per step: the mass fluxes <em>cu</em>, <em>cv</em>, the potential vorticity <em>z</em> and the Bernoulli head <em>h</em>) and executed in embedded mode, meaning the field-view program runs operator by operator as JAX array operations under <code>jax.jit</code>; there is no code generation and no fusion beyond what XLA does with the resulting array program. Periodic boundaries are the halo: the reference model fills them with <code>make_periodic</code>, a chain of <code>concat_where</code> copies, and the sharded model fills them by exchange.</p>
<h3>The sharded formulation</h3>
<p>The global M×N grid is cut into R<sub>x</sub>×R<sub>y</sub> equal blocks, one per device, each stored as a local array of (M/R<sub>x</sub>+2)×(N/R<sub>y</sub>+2) cells with a one-cell halo. The blocks of one field are stacked rank-major into a (P·M/R<sub>x</sub>)×(N/R<sub>y</sub>) array sharded along its first axis over a one-dimensional device mesh, and the whole time loop runs inside one <code>jax.shard_map</code> body: pad the three input blocks, refresh their halos, take the Euler step, then a <code>lax.scan</code> of n−1 leapfrog steps in which each step first refreshes the halos of <em>u</em>, <em>v</em> and <em>p</em> (three exchanges per step) and then applies the same GT4Py step function to the local halo-extended arrays. Corners are filled by the exchange, so the stencil never sees a periodic wrap. The gradient is <code>jax.grad</code> of the scalar cost Σp² over the global interior after the last step with respect to the three initial fields; it is produced by JAX transposing the whole shard_map body, exchanges included, with no hand-written adjoint anywhere. The rematerialised variant wraps the scan step in <code>jax.checkpoint</code>, so the reverse pass recomputes each step's intermediates instead of reading them back from a tape of n stored steps. The single-GPU reference runs the identical step function and the identical scan structure on the global grid with <code>make_periodic</code> halos and no shard_map.</p>
<h3>The five transports</h3>
<p>A transport is a function from a local halo-extended array to the same array with its halo refreshed, built only from <code>jax.lax</code> collectives and gathers so that <code>jax.vjp</code> can transpose it. All five are exact in the forward (bit-identical to the all-gather oracle) and differ in what crosses the wire and in how their adjoint comes out:</p>
{table(["transport", "mechanism per exchange", "cells on the wire per GPU and field", "adjoint"], [
 ["padded all-to-all", "one dense <code>lax.all_to_all</code> with a fixed slot per peer pair, sized to the largest chunk any pair needs; receiver gathers its rim from the slots", "P × largest chunk (32×1 at 16384²: 32 × 16386)", "all_to_all transposes to all_to_all; gather to scatter-add"],
 ["coloured ppermute, 8 rounds", "eight <code>lax.ppermute</code> rounds, one per direction incl. corners, packed into one buffer", "exactly the rim: 2(M/R<sub>x</sub> + N/R<sub>y</sub>) + 4", "each ppermute transposes to the inverse permutation"],
 ["coloured ppermute, 2 phases", "east–west round pair, then north–south on the x-refreshed array (corners ride along), four rounds", "exactly the rim", "as above, phases in reverse order"],
 ["all-gather", "<code>lax.all_gather</code> of every block, then a static gather of the rim", "P × M/R<sub>x</sub> × N/R<sub>y</sub> received (the whole field)", "all_gather transposes to psum_scatter"],
 ["ragged all-to-all", "<code>lax.ragged_all_to_all</code> with exact per-pair sizes", "exactly the rim", "the built-in transpose rule; wrong in jax {env.get('jax')}"],
])}
<p>The padded transport as inherited from the FESOM2-JAX design carried a <code>where(pad_valid)</code> mask on the send buffer that its authors describe as necessary for the transpose; in this formulation the receive side never reads the pad slots, the mask is provably a no-op for both forward and gradient, and it was removed before the sweeps. The coloured transports' cost is independent of P by construction; padded's grows with P because every peer gets a slot; all-gather's grows with the global size.</p>
<h3>Hardware and software</h3>
<p>Santis is the CSCS Alps Grace–Hopper system: each node has four NVIDIA GH200 superchips (72-core Grace CPU, H100-class GPU with 96 GB of HBM3 at 4 TB/s, NVLink-C2C between them), the four GPUs of a node are connected by NVLink, and nodes by Slingshot-11 with four 200 Gb/s NICs per node. The software stack was the <code>icon/26.7:v1</code> uenv (Python {env.get('python')}, CUDA 13.1, NCCL 2.29 with the aws-ofi-nccl 1.17 plugin), a <code>uv</code> virtual environment created from the uenv's interpreter with <code>jax[cuda13]=={env.get('jax')}</code> from PyPI, and gt4py installed editable from the <code>ad_halo</code> branch. GT4Py needed one fix to run under <code>jax.grad</code> on this JAX version: since jax 0.11 a tracer is no longer a subclass of <code>jax.Array</code>, so gt4py's <code>singledispatch</code> field constructor did not recognise traced arrays; the fix is registered explicitly at import (upstream PR GridTools/gt4py#2867).</p>
<p>Two process models were used. <em>Single process</em>: one Python process per node drives all four GPUs through a local four-device mesh; JAX dispatches to the GPUs from one host thread and collectives run over NCCL within the node. <em>Multi-process</em>: one process per GPU, launched by <code>srun</code> with four tasks per node, initialised with <code>jax.distributed.initialize()</code> from the Slurm environment (the coordinator is the first task; each process picks the GPU with its local rank); collectives run over NCCL, within a node over NVLink and across nodes over Slingshot through the libfabric plugin (<code>NCCL_NET="AWS Libfabric"</code>, <code>NCCL_NET_PLUGIN=ofi</code>, <code>NCCL_CROSS_NIC=1</code>, <code>FI_CXI_DISABLE_HOST_REGISTER=1</code>; the plugin load is confirmed in the NCCL log of the 8-node sanity job). All sweeps ran with <code>XLA_PYTHON_CLIENT_PREALLOCATE=false</code>; one dedicated job repeated the gradient cases at the memory boundary with preallocation on (§7, §9). Nothing else was tuned: default XLA flags, no autotuning cache, no CUDA graphs forced.</p>
<h3>Measurement protocol</h3>
<p>One <em>case</em> is a (mode, transport, layout, global size) combination. For each case the program is jitted and called once (compile plus first run, recorded separately), then called 20 more times with <code>jax.block_until_ready</code> around each call; every call advances the same initial state by 20 time steps, and the wall time of one call divided by 20 is one <em>per-step sample</em>. Each case therefore yields 20 samples; the report shows their median and interquartile range, and every speed-up or efficiency is a ratio of medians whose uncertainty, where it is visible in the tables, is half the spread of the quartile ratios. Compile time is excluded from every number. In multi-process runs every process times its own calls and only the first process writes the row; the processes' samples agree because every collective synchronises them. For the exchange-only modes the program is a single exchange call (or its gradient) with one step, so the reported number is per call and includes the fixed dispatch cost of a call, about 0.15–0.4 ms, which is why those numbers are upper bounds and are read only from 4096² upward. Peak device memory comes from JAX's per-device allocator statistics; because that peak is a process-lifetime maximum, a case is credited with the peak only when the peak rose during that case (§7 gives the cases where it did). A case whose gradient exhausts device memory is recorded as out-of-memory and its larger sizes are skipped. The correctness battery (§2) uses the same programs and the same transports at 16×16 (as a gate for every code change) and at 2048² per GPU on 32 GPUs (once, for this report).</p>
<h3>The experiment matrix</h3>
<p>Global grid sizes are M = N = 2ⁿ for n = 5…14 in the strong-scaling sweeps (32² to 16384²) and per-GPU block edges 2⁵…2¹² in the weak-scaling sweeps (32² to 4096² per GPU, so a 32×1 layout at block edge 4096 is a 131072×4096 global grid). Every transport, every mode and every size was run on every layout in the table below; the P = 1 layout supplies the baselines, the single-process and multi-process P = 4 runs cross-check each other, and everything above four GPUs is multi-process.</p>
{table(["GPUs", "nodes", "layouts", "process model", "modes measured"], [
 ["1", "1", "1×1", "single", "fwd, grad, grad remat, reference fwd/grad/grad remat, exchange, exchange+adjoint"],
 ["2", "1", "2×1", "single and multi", "fwd, grad, grad remat, exchange, exchange+adjoint"],
 ["4", "1", "2×2, 4×1", "single and multi", "fwd, grad, grad remat, exchange, exchange+adjoint"],
 ["8", "2", "8×1, 4×2", "multi", "fwd, grad, grad remat, exchange, exchange+adjoint"],
 ["16", "4", "16×1, 4×4", "multi", "fwd, grad, grad remat, exchange, exchange+adjoint"],
 ["32", "8", "32×1, 8×4", "multi", "fwd, grad, grad remat, exchange, exchange+adjoint"],
])}
<p>Beyond the sweeps, dedicated jobs supplied the sanity checks reported in §9: the correctness battery at 2048² per GPU on 4 and 32 GPUs, step-count linearity (10- and 40-step calls), a repeat of 128 single-GPU cases and of the 16 largest 32-GPU cases in other jobs on other nodes, the gradient at the memory boundary with preallocation, one-step versus 20-step timings of the same programs, and two Nsight profiles of single steps (speed-of-light per kernel, and DRAM bytes with FP64 instruction counts for the roofline). In total {D['n_rows_ok']} measured cases entered the analysis, from Slurm jobs 855906–856172 between 9 and 10 September 2026; the first-generation sweeps with five repeats and the smoke tests were superseded by the 20-repeat sweeps and are not used. Three single-process jobs hit their time limit in a hang (§9) and their missing cases were re-run with size caps; the first multi-node pass ran only the strip layouts because of a job-script error, and the block layouts were run afterwards in separate jobs, so several result files per configuration were merged with the later row winning.</p>
"""

correctness = f"""
<p>Scaling numbers of a wrong result are worthless, so the correctness battery that gates the transports at 16×16 was made size-parametrised and run at 2048² per GPU on 32 GPUs (four processes per node, eight nodes). It checks the exchange against the all-gather oracle, the dot-product identity of the exchange and its transpose, bit-identity of the sharded forward against the single-GPU reference, the gradient against the reference gradient, and the Taylor remainder of the cost.</p>
{battery_table()}
<p>The ragged transport's forward is bit-exact but its gradient is wrong, by orders of magnitude, on 4 GPUs and on 32 alike: the transpose rule of <code>lax.ragged_all_to_all</code> in jax {env.get('jax')} does not implement the adjoint. Its forward timings are reported below; its gradient timings are omitted because they time a wrong computation. Running the battery at this size also exposed that the gradient gate was calibrated for 16×16: the per-field relative error grew with grid size while the absolute error stayed at 10⁻¹³ of the pressure adjoint, because the velocity gradients shrink with resolution. The gate now uses the error relative to the largest gradient component.</p>
"""

single = f"""
<p>On one GPU the reference model takes {fmt(ref['ref']['16384']['med'])} ms per step at 16384² and {fmt(ref['ref']['4096']['med'])} ms at 4096², a clean quadrupling per size doubling from 2048² upward. Below about 512² every configuration sits on a floor of 0.03–0.07 ms per forward step and about 0.1 ms per gradient step: a step is a fixed number of kernels, and the fit of one-call time against step count gives a fixed cost of about 0.16 ms per call plus 0.03 ms per step at 64², so the small-size points measure launch latency, not compute.</p>
{baseline_table()}
<figure>{ncu_ref}{ncu_fwd}<figcaption>Per-kernel profile of one step at 16384² (Nsight Compute, one GPU). Every fusion of the reference step moves data at 73–92% of the 4 TB/s HBM3 peak; the sharded padded step adds three pad and three select fusions for the halo bookkeeping and three scatter launches that are negligible. Kernel time per step: reference 17.4 ms, sharded forward 25.9 ms.</figcaption></figure>
{ROOF_BLOCK}
<p><b>The step is memory-bound and close to the roofline per kernel, but it moves several times more data than a fused stencil would.</b> The kernel time of a reference step at 16384² is 17.4 ms; at 85% of 4 TB/s that is about 59 GB, roughly 220 bytes or 28 double-precision accesses per cell per step, against an estimated 100 bytes (read six fields, write six) for a fully fused update, so about twice the traffic of the ideal. That is the price of executing the field-view program operator by operator: every <code>concat_where</code> halo fill and every intermediate is materialised. The sharded version pays a further {N['overhead16']:.2f}× at P=1 for its pad and select kernels ({fmt(base16)} versus {fmt(ref16)} ms per step), which is the cost of formulating the exchange on a halo-extended array rather than on the periodic field. XLA's own cost analysis is not usable for this comparison: it counts the scan body once per compiled module, so its byte totals understate per-step traffic by up to the step count.</p>
<h3>The time loop itself costs 1.4–1.7× per step</h3>
<p>All sweep timings run the 20 steps inside one <code>lax.scan</code>. Timing the same programs with one, two, five and twenty steps per call shows a step change, not a gradual one: one and two steps per call, which XLA unrolls, cost 25.6 and 28.1 ms per step for the padded forward at 16384²; five and twenty steps, which compile to a real while loop, cost 40.6 and 41.1 ms. For the reference model the unrolled step is 17.5 ms, exactly the kernel time the profiler sees, against 30.1 ms in the loop. The compiled HLO of the same program on CPU points at the mechanism: the while-loop body carries the six state fields through copies every iteration, and the unrolled program has none. Six extra read-and-write passes over 2 GB fields account for roughly half of the 13 ms gap at 16384² by a bandwidth estimate, so lost fusion across the step boundary likely contributes the rest; this attribution is an inference from the CPU HLO and the step-count experiment, not a GPU-side measurement. The sweep numbers describe the model as written; a time loop without this penalty would move every curve below by up to this factor without changing any ratio between configurations.</p>
{scan_table()}
"""

strong = f"""
<p>Strong scaling fixes the global grid and adds GPUs. The reference line is the plain single-GPU model. Points at and below 512² are in the launch-floor regime and are shown only for completeness.</p>
{fig(strong_chart('fwd', 'padded', 'Forward step, padded transport'), 'Forward step time versus global grid size for the padded transport, one GPU to 32 GPUs (four per node above P=4), with the single-GPU reference model. Error bars are the interquartile range of 20 calls; most are smaller than the marker.')}
{fig(strong_chart('grad_remat', 'padded', 'Gradient step (rematerialised), padded transport'), 'The same for the gradient with per-step rematerialisation. On one GPU the stored-residual gradient runs out of memory at 4096² with the allocator settings of the sweeps (it fits with preallocation, see §7), so the rematerialised variant is the one that reaches the large sizes everywhere.')}
{fig(eff_chart('fwd', 16384, 'Parallel efficiency, forward, 16384²'), 'Forward efficiency at 16384² per transport. The coloured transports hold 91–83% at 32 GPUs; padded drops to 66% because its slot-padded all-to-all volume grows with the GPU count; all-gather collapses because it moves the whole array.', C.legend([(TNAME[t], COL[t], False) for t in ['padded', 'coloured2ph', 'coloured8', 'allgather', 'ragged']]))}
{fig(eff_chart('grad_remat', 8192, 'Parallel efficiency, gradient (remat), 8192²'), 'Gradient efficiency at 8192². The reverse pass doubles the exchanges per step; the padded transport&#39;s transposes carry its padded volume twice. The ragged transport is omitted: its gradient is wrong.', C.legend([(TNAME[t], COL[t], False) for t in ['padded', 'coloured2ph', 'coloured8', 'allgather']]))}
<h3>Node scaling at the largest sizes</h3>
<p>Forward step at 16384² (speed-up over one GPU, parallel efficiency):</p>
{node_table('fwd', 16384)}
<p>Rematerialised gradient at 8192² (the largest size that runs on one GPU):</p>
{node_table('grad_remat', 8192)}
<p>One process driving four GPUs and four processes with one GPU each give the same numbers to within the run-to-run spread (2×2 rows above), so the single-process shortcut is a valid stand-in for the production layout on one node.</p>
<h3>1-D strips against 2-D blocks</h3>
<p>The first pass of the multi-node sweeps ran only the strip layouts (a job-script error truncated the layout list at the comma); the block layouts were run afterwards in dedicated jobs. Median ms per step, strip / block, ratio in brackets; forward at 16384², rematerialised gradient at 8192²:</p>
{table(["same P", "transport", "forward 16384²", "gradient (remat) 8192²"], strip_block_rows())}
<p>For the padded transport the block layout is faster wherever the strip has become thin, because its slot-padded all-to-all sizes every slot to the largest chunk: on 32×1 at 16384² that is a full 16384-cell row per peer, on 8×4 a 4096-cell one. The coloured transports move only the halo cells and are indifferent to the layout at these sizes.</p>
"""

weak = f"""
<p>Weak scaling fixes the block per GPU and adds GPUs, so ideal behaviour is a flat step time. Efficiency is the single-GPU time divided by the time at P GPUs for the same per-GPU block.</p>
{fig(weak_chart('fwd', 'Weak scaling, forward, padded'), 'Forward weak-scaling efficiency for four per-GPU block sizes. At 4096² per GPU the step time rises only 12% from one to 32 GPUs; at 256² per GPU the exchange is the step.', C.legend([('256² per GPU', 'var(--w1)', False), ('1024² per GPU', 'var(--w2)', False), ('2048² per GPU', 'var(--w3)', False), ('4096² per GPU', 'var(--w4)', False)]))}
{fig(weak_chart('grad_remat', 'Weak scaling, gradient (remat), padded'), 'Gradient weak-scaling efficiency. The same picture shifted down: two exchanges per step and their transposes.', C.legend([('256² per GPU', 'var(--w1)', False), ('1024² per GPU', 'var(--w2)', False), ('2048² per GPU', 'var(--w3)', False), ('4096² per GPU', 'var(--w4)', False)]))}
<p>At 4096² per GPU the padded forward keeps {pct(D['weak']['fwd']['padded']['single:1x1']['4096']['med'] / D['weak']['fwd']['padded']['multi:32x1']['4096']['med'])} efficiency on 32 GPUs and the rematerialised gradient {pct(D['weak']['grad_remat']['padded']['single:1x1']['4096']['med'] / D['weak']['grad_remat']['padded']['multi:32x1']['4096']['med'])}. At 1024² per GPU, the size at which the step itself is only 0.18 ms, the gradient efficiency is {pct(D['weak']['grad_remat']['padded']['single:1x1']['1024']['med'] / D['weak']['grad_remat']['padded']['multi:32x1']['1024']['med'])} on 32 GPUs: an exchange plus its adjoint costs 0.3–0.6 ms per call on 8–32 GPUs whenever the block is small (exchange-only timings at 1024²–2048² global), regardless of how little data it carries.</p>
"""

transports = f"""
{fig(transports_bar(), 'Forward step time at 8192² for every transport and GPU count (log scale). All-gather is the oracle, not a candidate. The ragged transport is shown for the forward only; its gradient is wrong.')}
{C.legend([(TNAME[t], COL[t], False) for t in ['padded', 'coloured2ph', 'coloured8', 'allgather', 'ragged']])}
<p>At 8192² padded is the fastest or tied forward transport up to 16 GPUs, the coloured 8-round transport is within 5% of it and the coloured 2-phase transport within 26%; the ragged transport, where it runs, matches padded in the forward. On the 1-D strips the forward ranking flips at 32 GPUs and the gradient's already at 16: the coloured transports keep their per-step cost while padded's dense all-to-all, whose slot is sized to the longest row, becomes the most expensive exchange. At 16384² on the 32×1 strip the coloured 2-phase forward step is {p32/c32:.2f}× faster than padded's, and its rematerialised gradient step {gr32/grc32:.1f}× faster at 8192² and {med('grad_remat','padded','multi:32x1',16384)/med('grad_remat','coloured2ph','multi:32x1',16384):.1f}× at 16384². On the 2-D blocks the picture is different: on 8×4 at 16384² padded's forward step ({fmt(med('fwd','padded','multi:8x4',16384))} ms) is the fastest of all transports (coloured 2-phase {fmt(med('fwd','coloured2ph','multi:8x4',16384))} ms), and its gradient is within 5% of the coloured 2-phase one, because the 4096-cell rows keep its slots small. All-gather is 0.5× the single-GPU speed on any multi-node layout because every GPU receives the whole array.</p>
<p>The padded transport's <code>where(pad_valid)</code> mask, which the FESOM2-JAX write-up describes as load-bearing for the transpose, is not needed in this formulation: forward and gradient are bit-identical without it, because the receive side never reads the pad slots. It was removed before the sweeps.</p>
"""

adcost = f"""
{fig(ad_cost_chart(), 'Gradient time divided by forward time. Squares: the plain reference model; circles: the padded sharded model at 1, 4 and 32 GPUs. Solid lines store the residuals of all 20 steps; dashed lines rematerialise each step. Blank regions are where the stored-residual gradient does not fit in memory.')}
<p>The gradient of the reference model costs {N['ad_ref']:.1f}× the forward at 2048² with stored residuals and {N['ad_ref_remat']:.1f}× at 4096² with rematerialisation. Both are above the textbook 2–3× because the forward is itself memory-bound: the reverse pass reads back every stored intermediate, and for this operator-by-operator step there are many of them. Rematerialisation is not a trade of compute for memory here but a net win once blocks exceed about 1024² per GPU:</p>
{remat_table()}
<p><em>Ratio of the rematerialised to the stored-residual gradient step time, padded transport. Below 1 the rematerialised gradient is faster. Blank cells: the stored-residual gradient did not fit.</em></p>
<h3>Memory</h3>
<p>The reverse pass of a 20-step scan with stored residuals needs about 35 GiB at 4096² per GPU and about four times that at 8192², which no longer fits into the 95.6 GiB of a GH200. The sweeps ran with <code>XLA_PYTHON_CLIENT_PREALLOCATE=false</code>, under which the allocator fragments and the 4096² case already reports out-of-memory; with preallocation on it runs. The table gives the per-GPU peak of the rematerialised gradient (which scales cleanly by 4× per size doubling and reaches 46.5 GiB at 8192² per GPU) and the step time of the stored-residual gradient with preallocation at the boundary. Memory figures are GiB (bytes/2³⁰).</p>
{mem_table()}
{fig(mem_chart(), 'Peak device memory per GPU for the rematerialised gradient, padded transport. The four-process 2×2 layout has one quarter of the cells per GPU and one quarter of the memory. At 16384² per GPU the rematerialised gradient no longer fits either.')}
"""

comm = f"""
<p>To separate communication from stencil work, one halo exchange and one exchange-plus-adjoint were timed on their own for every size and layout. A standalone exchange call carries a fixed dispatch cost of 0.15–0.4 ms, so the numbers are an upper bound on what the same exchange costs when it is fused into a step, and only meaningful from about 4096² upward where the exchange time clearly exceeds that floor.</p>
{fig(exch_chart(), 'Time of one exchange call (solid) and of one exchange with its adjoint (dashed), padded transport, per GPU count. Below 2048² the curves are flat at the dispatch floor.')}
{C.legend([('P=1', PCOL[1], False), ('P=4', PCOL[4], False), ('P=8', PCOL[8], False), ('P=32', PCOL[32], False)])}
<p>Exchange time and its share of the corresponding step (forward exchange over forward step, exchange plus adjoint over rematerialised gradient step):</p>
{exch_share_table()}
<p>At 4096² per GPU on four GPUs the exchange is {100*share('multi:2x2', 8192, 'exch', 'fwd'):.0f}% of the forward step and the exchange plus adjoint {100*share('multi:2x2', 8192, 'exch_grad', 'grad_remat'):.0f}% of the rematerialised gradient step; at 8192² per GPU {100*share('multi:2x2', 16384, 'exch', 'fwd'):.0f}% and {100*share('multi:2x2', 16384, 'exch_grad', 'grad_remat'):.0f}%; at 512 rows per GPU on 32 GPUs {100*share('multi:32x1', 16384, 'exch', 'fwd'):.0f}% and {100*share('multi:32x1', 16384, 'exch_grad', 'grad_remat'):.0f}%. That is the whole loss of scaling efficiency in the tables above, and for the padded transport it is volume, not latency: the useful halo at 16384² on 32×1 is two rows of 16384 cells, 0.26 MB per GPU per field, but the slot-padded all-to-all sends the largest chunk to every one of the 32 peers, 4.2 MB per GPU per field, and the exchange time grows with the row length (0.33, 0.40 and 0.57 ms at 4096², 8192² and 16384² on 32 GPUs). 4.1 MB off-device in 0.57 ms is 7 GB/s per GPU, a few times below one Slingshot NIC. The coloured transports move only the halo and stay flat. The adjoint of an exchange costs 1.5–3× the exchange, as expected for a transpose that must also accumulate into the owners.</p>
"""

checks = f"""
<h3>Step-count linearity</h3>
<p>The per-step numbers are wall time per 20-step call divided by 20, so a fixed per-call cost biases small sizes. Fitting one-call time against step count with 10- and 40-step calls gives:</p>
{lin_table()}
<p>At 4096² the 20-step per-step numbers are within 1–2% of the asymptote (the negative fixed costs there are two-point-fit noise); at 64² the fixed per-call cost is 10–40% of the reported per-step time. That is why the tables mark everything at and below 512² as launch-floor dominated.</p>
<h3>Repeatability</h3>
<p>128 single-process cases measured again in a later job on another node agree with the sweep at a median ratio of 0.995 with a worst case of 17% (a 2×2 gradient case); the sixteen largest 32-GPU cases repeated in a second 8-node job agree to within 2.3%. Of the {IQR[0]} strong-scaling points at 1024² and above, {100*IQR[1]:.0f}% have an interquartile range below 1% of the median and {100*IQR[2]:.0f}% below 5%; the widest spreads are padded on 32 GPUs at large sizes (3–7%) and a few multi-node small-size cases where the timing of a 0.1 ms step across 32 processes is bimodal.</p>
<h3>Allocator</h3>
<p>With <code>XLA_PYTHON_CLIENT_PREALLOCATE=true</code> and 92–95% of device memory reserved, the stored-residual gradient at 4096² on one GPU runs (34.9 GiB peak) where the default setting of the sweeps reports out-of-memory; 8192² fails either way. The out-of-memory boundaries quoted for the plain gradient are therefore one size conservative, and the memory table above reports the preallocated result.</p>
<h3>Single-process multi-GPU hang</h3>
<p>Three independent single-process jobs wrote all their rows within 15 minutes and then sat in the same case until their time limit: the 2×2 gradient at 8192² (4096² per GPU), and the rematerialised gradient at 16384². The identical case run as four processes either completes or reports out-of-memory cleanly. The single-process four-device gradient at that memory pressure deadlocks rather than failing, presumably inside a collective when one device's allocation fails. The affected single-process cells were therefore capped one size below the hang; the multi-process rows cover those sizes.</p>
"""

findings = f"""
<ul>
<li><b>Keep the padded transport on square-ish blocks, or switch to coloured 2-phase on strips.</b> Padded is simplest and fastest within a node and on 2-D block layouts at every GPU count tested; its slot is sized to the longest row, so on the 32×1 strip its volume is 16× the halo and coloured 2-phase, which moves only the halo, keeps 91% forward efficiency where padded drops to 66%. Both have exact adjoints by composition.</li>
<li><b>Rematerialise the time step.</b> <code>jax.checkpoint</code> on the scan step is faster than storing residuals from about 1024² per GPU and is the only way to differentiate 20 steps at 8192² per GPU on 96 GB.</li>
<li><b>Fix the time loop before optimising the exchange.</b> The 20-step <code>lax.scan</code> costs 1.4–1.7× per step relative to an unrolled step (most likely loop-carried copies), and the penalty appears as soon as XLA stops unrolling (at five steps); the exchange costs 5–30%. The stencil itself runs at 85% of HBM bandwidth per kernel but moves about twice the traffic of a fused update, which is the embedded-mode price and the larger optimisation target for GT4Py.</li>
<li><b>Do not use <code>lax.ragged_all_to_all</code> under <code>jax.grad</code></b> in jax {env.get('jax')}: forward exact, gradient wrong, on every layout tested.</li>
<li><b>Prefer one process per GPU for gradients near the memory limit.</b> The single-process four-device gradient hangs instead of failing at 4096² per GPU.</li>
<li><b>Set <code>XLA_PYTHON_CLIENT_PREALLOCATE=true</code></b> for memory-bound reverse passes; the non-preallocating allocator loses one size step to fragmentation.</li>
<li><b>The sharded formulation costs {N['overhead16']:.2f}× at P=1</b> ({fmt(base16)} versus {fmt(ref16)} ms per step at 16384²) for its pad and select kernels on the halo-extended array. Writing the exchange on the periodic field directly, or fusing the halo fill into the step, would recover it.</li>
</ul>
"""

limits = f"""
<ul>
<li>Timings are of the model as written on the <code>ad_halo</code> branch; the scan-loop copies and the pad/select kernels are properties of that code, not of JAX or GH200 in general.</li>
<li>The stored-residual gradient at 4096² per GPU on one GPU exists only with preallocation on; on the multi-GPU layouts it ran with the default allocator up to 4096² per GPU and is absent beyond. The rematerialised variant carries the large sizes.</li>
<li>Single-process 2×2 and 4×1 gradients are capped at 4096² (stored) and 8192² (rematerialised) because of the hang; the multi-process rows are complete.</li>
<li>Per-case peak memory is a process-lifetime maximum and is attributable only to the cases that raised it (the padded 1×1 and 2×2 columns); the plain-gradient peaks come from the separate preallocated runs.</li>
<li>Exchange-only timings include a dispatch floor and are upper bounds on the fused cost.</li>
<li>The roofline profiles one unrolled step per call. For the forward steps that program matches the scan body kernel for kernel; for the sharded gradient it does not: the one-step sharded gradient at 16384² spends 58 of its 100 ms of kernel time in a single select fusion that the 20-step scan version does not have (in the scan, the sharded and reference gradient steps cost the same per step), so the sharded gradient is left off the roofline figure. Why XLA compiles the unrolled sharded gradient this way is open.</li>
<li>The halo width is 1 throughout; wider halos would raise the exchange volume without changing its latency, and would favour the 2-D layouts.</li>
</ul>
<h3>Reproducibility</h3>
<p>Everything lives in <code>examples/next/swm/</code> of the <code>ad_halo</code> branch: <code>scaling_bench.py</code> (modes fwd, grad, grad_remat, ref, ref_grad, ref_grad_remat, exch, exch_grad; per-repeat samples; XLA cost and kernel capture), <code>swm_battery.py --size</code>, and under <code>santis/</code> the setup script, the job scripts for every run in this report (smoke, sweep, remat, exchange, sanity, profile, prealloc, scan-step, fill-in) and <code>analyze.py</code>, which produced every table here from the JSONL rows. Setup on Santis: uenv <code>icon/26.7:v1</code>, a <code>uv</code> venv from its Python 3.13 with <code>jax[cuda13]==0.11.1</code> and gt4py editable; NCCL over Slingshot via the uenv's aws-ofi-nccl plugin with <code>NCCL_NET="AWS Libfabric"</code>. Slurm jobs 855906–856098 on account csstaff, September 9–10, 2026.</p>
"""

html_out = HEAD + intro + \
    sec("setup", "What was measured and how", setup, "1 · Setup and method") + \
    sec("correctness", "Correctness at scale", correctness, "2 · Gate") + \
    sec("single", "One GPU: baseline, roofline, time loop", single, "3 · Baseline") + \
    sec("strong", "Strong scaling", strong, "4 · Fixed global grid") + \
    sec("weak", "Weak scaling", weak, "5 · Fixed block per GPU") + \
    sec("transports", "Transports", transports, "6 · Five halo exchanges") + \
    sec("ad", "Cost of the gradient and memory", adcost, "7 · Reverse mode") + \
    sec("comm", "Communication share", comm, "8 · Exchange-only timings") + \
    sec("checks", "Methodology checks", checks, "9 · Sanity") + \
    sec("findings", "Findings and recommendations", findings, "10 · Conclusions") + \
    sec("limits", "Limitations and reproducibility", limits, "11 · Scope") + "</main>"
open("santis_gpu_scaling_report.html", "w").write(html_out)
print("written", len(html_out) // 1024, "kB")
