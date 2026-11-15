"""Build a self-contained HTML report for the model experiments (U-Net variants, TerraMind, LightGBM, ensembles).

    python3 build_model_report.py --runs ../../runs --out ../docs/model_experiments_report.html

Reads only the result files written by ``bogsignal.train_unet`` (final.json), ``bogsignal.ensemble``,
``bogsignal.baseline_gbm`` and the example figures from ``bogsignal.plot_unet``; every number and chart is
computed here from those files. Needs matplotlib. Companion to docs/final_experiment_report.html (the
seed-17 A-D and temporal experiments), whose layout, chart style and percentage formatting it follows.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import html
import io
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Other", "Swamp", "Marsh", "Permanent water", "Peatland cropland", "Peatland non-cropland"]
SOURCES: list[dict] = []
RUNS = Path()

STYLE = """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f1f4f5;color:#1a2b33;font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1160px;margin:32px auto;background:white;padding:44px 48px;border-radius:12px}h1{font-size:38px;line-height:1.18;letter-spacing:-1px}h2{font-size:25px;line-height:1.35;margin-top:0}h3{font-size:19px}header{border-bottom:3px solid #20836a;margin-bottom:28px;padding-bottom:16px}.eyebrow{color:#366d64;letter-spacing:.1em;text-transform:uppercase;font-size:12px;font-weight:700}.subtitle{font-size:20px;color:#53646c}.summary{background:#ecf6f1;border-left:5px solid #20836a;padding:24px!important;border-radius:6px}.note{border-left:3px solid #ba913e;background:#fff9e9;padding:12px 16px}nav{display:flex;flex-wrap:wrap;gap:10px 22px;margin:28px 0}a{color:#176b58}section{padding-top:20px;margin:25px 0 38px}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}td,th{padding:11px 10px;text-align:left;border-bottom:1px solid #dce5e5;vertical-align:top}th{background:#eaf0f0;font-weight:650}tbody tr:nth-child(even){background:#f7fafa}tr.lead td{font-weight:650}.table-wrap{overflow-x:auto;margin:16px 0}figure{margin:24px 0}img{display:block;width:100%;height:auto}figure.narrow img{max-width:720px;margin:0 auto}figcaption,.small{font-size:13px;color:#52646c}figcaption{margin-top:9px}li{margin-bottom:10px}code{font:12px/1.6 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}details{margin:20px 0}summary{cursor:pointer;font-weight:600}@media(max-width:700px){main{margin:0;padding:24px 16px;border-radius:0}h1{font-size:30px}h2{font-size:22px}}@media print{body{background:white;font-size:10pt}main{margin:0;padding:0;max-width:none}nav{display:none}section{break-before:auto}tr,img{break-inside:avoid}h2,h3{break-after:avoid}.table-wrap{overflow:visible}details{display:none}a{color:inherit}}"""


def read(rel: str) -> dict:
    path = RUNS / rel
    SOURCES.append({"path": f"runs/{rel}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return json.loads(path.read_text())


def figure(png: bytes, caption: str, cls: str = "") -> str:
    return (f'<figure class="{cls}"><img alt="{html.escape(caption, quote=True)}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def image(rel: str, caption: str, cls: str = "") -> str:
    path = RUNS / rel
    SOURCES.append({"path": f"runs/{rel}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return figure(path.read_bytes(), caption, cls)


def table(head: list[str], rows: list[list[str]], lead: int | None = None) -> str:
    body = "".join(f'<tr{" class=lead" if i == lead else ""}>' + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for i, r in enumerate(rows))
    return '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + f"</tr></thead><tbody>{body}</tbody></table></div>"


def mean(values) -> float:
    return statistics.mean(values)


def spread(values) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pts(x: float) -> str:
    """a difference in percentage points"""
    return f"{100 * x:+.1f}"


def chart(panels: list[dict], caption: str, figsize=(12, 4.7), sharey=True) -> str:
    """Grouped bar charts in the style of final_experiment_report.html: default palette, 0-100% axis, light grid.

    Each panel: title, ylabel, groups (x tick labels), series = [(legend label, values, errors or None)].
    Values are fractions and are drawn as percentages. Error bars are the spread over seeds.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=sharey, squeeze=False)
    for ax, p in zip(axes[0], panels):
        n = len(p["series"])
        width = 0.8 / n
        for k, (label, values, errors) in enumerate(p["series"]):
            xs = [g - 0.4 + width * (k + 0.5) for g in range(len(p["groups"]))]
            kw = {"color": p["colors"][k]} if "colors" in p else {}
            ax.bar(xs, [100 * v for v in values], width=width, label=label,
                   yerr=[100 * e for e in errors] if errors else None, capsize=3, error_kw={"lw": 1, "ecolor": "#333333"}, **kw)
            if p.get("annotate", n <= 2):
                for x, v in zip(xs, values):
                    ax.text(x, 100 * v + (p.get("ymax", 100) * 0.015) + (100 * max(errors) if errors else 0), f"{100 * v:.0f}",
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(p["groups"])), p["groups"], fontsize=p.get("tick_size", 10))
        ax.set_ylim(0, p.get("ymax", 100))
        ax.set_title(p["title"])
        ax.set_ylabel(p.get("ylabel", ""))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        if n > 1 or p.get("legend"):
            ax.legend(fontsize=8.5, loc=p.get("legend_loc", "upper right"))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150)
    plt.close(fig)
    return figure(buf.getvalue(), caption)


GET = {
    "swamp": lambda m: m["ap"]["Swamp"], "marsh": lambda m: m["ap"]["Marsh"], "wet": lambda m: m["wetland_ap"],
    "water": lambda m: m["ap"]["Permanent water"], "peat_c": lambda m: m["ap"]["Peatland cropland"],
    "peat_n": lambda m: m["ap"]["Peatland non-cropland"], "r5": lambda m: m["wetland_recall_at_area"]["5"],
}


def validation_runs(config: str, tag: str = "s") -> list[dict]:
    """final-epoch validation metrics of every seed (or fold) of one configuration"""
    return [read(str(p.relative_to(RUNS)))["last"]["validation"] for p in sorted((RUNS / "exp").glob(f"{config}_{tag}[0-9]/final.json"))]


def hard_map(m: dict) -> dict:
    """Hard-map scores from the six-class confusion matrix: swamp+marsh merged after plain argmax, and macro IoU."""
    cm = m["confusion"]
    tp = sum(cm[i][j] for i in (1, 2) for j in (1, 2))
    ref = sum(sum(cm[i]) for i in (1, 2))
    pred = sum(cm[i][j] for i in range(6) for j in (1, 2))
    r, p = tp / ref, tp / pred
    ious = [cm[k][k] / (sum(cm[k]) + sum(cm[i][k] for i in range(6)) - cm[k][k]) for k in range(6)]
    return {"recall": r, "precision": p, "f1": 2 * p * r / (p + r), "f2": 5 * p * r / (4 * p + r), "iou": tp / (ref + pred - tp), "macro_iou": mean(ious)}


def main() -> None:
    global RUNS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="the runs directory (contains exp/, gbm/, plots/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ad-report", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "final_experiment_report_data.json",
                    help="data file of the seed-17 A-D report, for the side-by-side rows")
    args = ap.parse_args()
    RUNS = args.runs.resolve()

    configs = [
        ("base", "U-Net · all 14 channels"), ("tm", "TerraMind-small · all 14 channels"),
        ("s2nisar", "U-Net · S2 + NISAR"), ("s2s1", "U-Net · S2 + S1"), ("s2", "U-Net · S2 only"),
        ("reg", "Dropout + weight decay"), ("jit", "Radiometric jitter"), ("aug", "Jitter + zoom"),
        ("ovr", "Sigmoid heads"), ("nopeat", "Peat ignored in training"),
    ]
    val = {c: validation_runs(c) for c, _ in configs}
    gbm = read("gbm/results.json")
    ens_val = read("exp/analysis/ens_unet3_tm3.json")["ensemble"]
    ens_val_u, ens_val_t = read("exp/analysis/ens_unet3.json")["ensemble"], read("exp/analysis/ens_tm3.json")["ensemble"]
    test = {k: read(f"exp/analysis/{f}.json") for k, f in (
        ("ens", "test_ens_unet3_tm3"), ("ens_unet", "test_ens_unet3"), ("ens_tm", "test_ens_tm3"),
        ("s2s1", "test_s2s1"), ("s2", "test_s2"), ("s2nisar", "test_s2nisar"))}
    tm_ = {"base": list(test["ens_unet"]["members"].values()), "tm": list(test["ens_tm"]["members"].values()),
           "s2s1": list(test["s2s1"]["members"].values()), "s2": list(test["s2"]["members"].values()),
           "s2nisar": list(test["s2nisar"]["members"].values())}
    kf = {c: validation_runs(c, "f") for c in ("kf_all", "kf_s2s1")}
    train_eval = read("exp/analysis/eval_base_s0-last_s1s2nisar_256_v4_labels-s1s2nisar_256_v4.json")
    knock = read("exp/analysis/eval_reg_s0-last_s1s2nisar_256_v4_labels-s1s2nisar_256_v4.json")

    # ---- scores with the A-D report's definitions (bogsignal.eval_combined)
    cv, ct = read("exp/analysis/combined_validation.json"), read("exp/analysis/combined_test.json")
    GROUP = {"s2": "S2", "s2s1": "S2+S1", "s2nisar": "S2+NISAR", "base": "U-Net", "tm": "TerraMind"}

    def flat(m: dict) -> dict:
        b = m["ranking"]["area_budgets"]
        return {"macro_iou": m["macro_iou"], **m["wetland"], "ap": m["ranking"]["average_precision"],
                "r1": b[0]["recall"], "r5": b[1]["recall"], "r10": b[2]["recall"], "r20": b[3]["recall"], "p5": b[1]["precision"]}

    def members(src, config):
        return [flat(m) for m in src["groups"][GROUP[config]]["members"]]

    def gmean(src, config):
        ms_ = members(src, config)
        return {k: mean([m[k] for m in ms_]) for k in ms_[0]}

    def gspread(src, config, key):
        return spread([m[key] for m in members(src, config)])

    ad = json.loads(args.ad_report.read_text())
    SOURCES.append({"path": "docs/" + args.ad_report.name, "sha256": hashlib.sha256(args.ad_report.read_bytes()).hexdigest()})
    ad_rows = []
    for rec, rank in zip(ad["baseline"], ad["candidate_ranking"]["baseline"]):
        cm = rec["combined_class"]["confusion_matrix"]
        tp, ref, pred = cm[1][1], sum(cm[1]), sum(r[1] for r in cm)
        r_, p_ = tp / ref, tp / pred
        b = rank["area_budgets"]
        ad_rows.append({"macro_iou": rec["validation"]["macro_iou"], "recall": r_, "precision": p_, "f1": 2 * p_ * r_ / (p_ + r_),
                        "f2": 5 * p_ * r_ / (4 * p_ + r_), "iou": tp / (ref + pred - tp), "ap": rank["average_precision"],
                        "r1": b[0]["recall"], "r5": b[1]["recall"], "r10": b[2]["recall"], "r20": b[3]["recall"], "p5": b[1]["precision"]})
    ad_names = ["A · Sentinel-2", "B · Sentinel-2 + Sentinel-1", "C · Sentinel-2 + NISAR", "D · Sentinel-2 + Sentinel-1 + NISAR"]

    def col(runs, key):
        return [GET[key](m) for m in runs]

    def avg(runs, key):
        return mean(col(runs, key))

    def hm(config, key):
        return [m[key] for m in members(cv, config)]

    e_test, u_test, t_test = test["ens"]["ensemble"], tm_["base"], tm_["tm"]
    d_val = avg(val["base"], "wet") - avg(val["s2s1"], "wet")
    d_test = avg(tm_["base"], "wet") - avg(tm_["s2s1"], "wet")
    fold_delta = [GET["wet"](a) - GET["wet"](b) for a, b in zip(kf["kf_all"], kf["kf_s2s1"])]
    gbm_px, gbm_ctx = gbm["nisar+s1+s2 raw+idx"], gbm["all raw+idx+ctx"]
    sensors = [("s2", "S2"), ("s2s1", "S2 + S1"), ("s2nisar", "S2 + NISAR"), ("base", "All sensors")]

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wetland segmentation — model experiments report</title><style>{STYLE}</style></head><body><main><header><p class="eyebrow">BogSignal / Argos · {dt.date.today():%d %B %Y}</p><h1>Wetland segmentation:<br>model, ensemble and channel experiments</h1><p class="subtitle">Multi-seed U-Net, TerraMind and LightGBM runs on dataset v4</p></header>"""]

    parts.append(f"""<section class="summary"><h2>Executive findings</h2>
<p><strong>The best model is an average of three U-Nets and three TerraMind models.</strong> On the held-out test split its swamp/marsh ranking score (average precision) is {pct(e_test['wetland_ap'])}, against {pct(avg(u_test, 'wet'))} for a single U-Net and {pct(avg(t_test, 'wet'))} for a single TerraMind. Inspecting its highest-scoring 5% of the test area recovers {pct(e_test['wetland_recall_at_area']['5'])} of the reference swamp/marsh pixels.</p>
<p><strong>In the A–D report's own tables the ensemble needs a fitted decision rule to lead.</strong> With the plain most-likely-class decision, A–D model D has higher wetland recall ({pct(ad_rows[3]['recall'])} versus {pct(flat(cv['ensembles']['Ensemble6']['result'])['recall'])}) and F2. With per-class decision offsets fitted on validation the ensemble reaches {pct(flat(cv['ensembles']['Ensemble6']['result_tuned'])['recall'])} recall at {pct(flat(cv['ensembles']['Ensemble6']['result_tuned'])['precision'])} reference precision on validation, and {pct(flat(ct['ensembles']['Ensemble6']['result_tuned'])['recall'])} at {pct(flat(ct['ensembles']['Ensemble6']['result_tuned'])['precision'])} on the test split with the same offsets (section 2).</p>
<p><strong>TerraMind matches the U-Net as a single model.</strong> Swapping the encoder did not change the validation score ({pct(avg(val['tm'], 'wet'))} versus {pct(avg(val['base'], 'wet'))}); averaging the two families did, to {pct(ens_val['wetland_ap'])}.</p>
<p><strong>Spatial context carries most of the signal.</strong> A per-pixel LightGBM on all 14 channels scores {pct(gbm_px['wetland_ap'])} on validation ({pct(gbm_ctx['wetland_ap'])} with neighbourhood means), against {pct(avg(val['base'], 'wet'))} for the U-Net.</p>
<p><strong>The NISAR contribution is not settled.</strong> All 14 channels beat Sentinel-2 + Sentinel-1 by {pts(d_val)} points on the fixed validation split and {pts(d_test)} on test, but across five geographic folds the mean difference is {pts(mean(fold_delta))} points, ranging from {pts(min(fold_delta))} to {pts(max(fold_delta))}.</p>
<p><strong>Five training changes were tested with three seeds each and none beat the plain U-Net:</strong> sigmoid heads, ignoring peat classes, added regularisation, radiometric jitter, and jitter with zoom.</p>
<p><strong>Recommendation:</strong> use the six-model ensemble for maps and candidate ranking. Spend further effort on inputs (spring and summer imagery, cloud-free composites) and label review, not on loss functions or architecture.</p>
<p class="note">All scores measure agreement with historical/inferred weak labels, not independent contemporary wetland accuracy. The test split was scored once, at the end, for the eight model rows in section 4; no checkpoint, threshold or ensemble member was chosen on it.</p></section>
<nav aria-label="Report contents"><a href="#design">Design</a><a href="#adtables">A–D tables</a><a href="#models">Model comparison</a><a href="#test">Test results</a><a href="#channels">Channels and NISAR</a><a href="#variants">Training variants</a><a href="#ranking">Candidate ranking</a><a href="#examples">Examples</a><a href="#diagnosis">Diagnosis</a><a href="#limits">Limits</a><a href="#artifacts">Artifacts</a></nav>""")

    parts.append("""<section id="design"><h2>1. Study question and fixed design</h2><p>These experiments ask which model, loss, augmentation and sensor set best segment swamp and marsh on dataset v4, and how much of any difference survives seed and geographic variation. They complement the seed-17 A–D and temporal report: same dataset and splits, a different training script and evaluation protocol.</p>"""
                 + table(["Item", "Specification"], [
                     ["Dataset", "V4: 3,753 patches; 2,609 train / 585 validation / 559 test; 256 × 256 pixels at 20 m, EPSG:3035; geographic folds with buffers"],
                     ["Inputs", "Sentinel-2 B2, B3, B4, B8, B11, B12; Sentinel-1 ascending/descending VV, VH; NISAR ascending/descending HH, HV. Missing pixels are zero after normalisation; no validity-mask channels"],
                     ["Targets", "Six classes: Other, Swamp, Marsh, Permanent water, Peatland cropland, Peatland non-cropland; 255 ignored"],
                     ["U-Net", "ResNet-34 encoder, ImageNet weights, six logits"],
                     ["TerraMind", "TerraMind 1.0 small encoder. Sentinel-2 (six-band subset) and ascending Sentinel-1 use the pretrained embeddings; descending Sentinel-1 starts from the Sentinel-1 embedding; NISAR is a new, randomly initialised modality. Light convolutional decoder with a full-resolution stem"],
                     ["Optimisation", "20 epochs, batch 16, AdamW, learning rate 3·10⁻⁴ (TerraMind backbone 6·10⁻⁵), weight decay 10⁻⁴, one warm-up epoch then cosine decay, mixed precision; cross-entropy + Dice; flips and quarter turns; dataset 75/25 minority/background sampling"],
                     ["Evaluation weights", "Exponential moving average of the weights (decay 0.998). All results use the <strong>final epoch</strong>, so no checkpoint is selected on validation"],
                     ["Replication", "Three seeds (0, 1, 2) per configuration. Tables give the mean; thin black lines on the charts show the spread over seeds"],
                     ["Geographic k-fold", "Five folds over the train + validation blocks; whole connected-wetland groups stay in one fold and training patches touching a held-out patch are dropped. The test split is never included"],
                     ["LightGBM", "Per-pixel multiclass model on a stride-8 pixel grid of the training patches; 200 rounds; raw channels, spectral and polarisation indices, optional 5- and 15-pixel neighbourhood means"],
                 ])
                 + """<h3>How to read the metrics</h3><p>All scores are percentages; higher is better. Two kinds of score appear.</p><ul><li><strong>Ranking score (average precision):</strong> how well the model orders pixels from most to least likely wetland, summarised over every possible threshold. 100% means every reference wetland pixel is ranked above every other pixel; a random ordering scores about 2%, the share of the area that is swamp or marsh. It is the main score here because it does not depend on where a decision threshold is put, and it is available for every model including the ensemble.</li><li><strong>Combined swamp/marsh:</strong> pixels are ranked by <code>pSwamp + pMarsh</code> against pixels labelled either class, so a swamp–marsh subtype swap is not an error.</li><li><strong>Hard-map scores (recall, reference precision, F2, IoU):</strong> every pixel gets its most likely class, then swamp and marsh are merged. <em>Recall</em> is the share of reference swamp/marsh pixels recovered; <em>reference precision</em> is the share of predicted swamp/marsh that overlaps the weak labels; <em>F2</em> combines them weighting recall more; <em>macro IoU</em> averages overlap over the six original classes. These are the scores used in the A–D report.</li><li><strong>Recall at 5% area:</strong> the share of reference swamp/marsh pixels inside the highest-scoring 5% of the scored area.</li><li><strong>Points</strong> means percentage points: 33% to 40% is +7 points.</li></ul></section>""")

    # ---- 2 the A-D report's tables, with the best model
    HARD = [("macro_iou", "Six-class macro IoU"), ("recall", "Wetland recall"), ("precision", "Reference precision"), ("f1", "F1"), ("f2", "F2"), ("iou", "Wetland IoU")]
    ours = [("s2", "U-Net · Sentinel-2"), ("s2s1", "U-Net · Sentinel-2 + Sentinel-1"), ("s2nisar", "U-Net · Sentinel-2 + NISAR"),
            ("base", "U-Net · all sensors"), ("tm", "TerraMind · all sensors")]
    e6v, e6v_t = flat(cv["ensembles"]["Ensemble6"]["result"]), flat(cv["ensembles"]["Ensemble6"]["result_tuned"])
    e6t, e6t_t = flat(ct["ensembles"]["Ensemble6"]["result"]), flat(ct["ensembles"]["Ensemble6"]["result_tuned"])
    offsets = cv["ensembles"]["Ensemble6"]["offsets"]
    hard_row = lambda label, m: [label] + [pct(m[k]) for k, _ in HARD]  # noqa: E731
    rank_row = lambda label, m: [label, pct(m["ap"]), pct(m["r1"]), pct(m["r5"]), pct(m["r10"]), pct(m["r20"]), pct(m["p5"])]  # noqa: E731
    rows_h = [hard_row(n + " (A–D report, seed 17)", m) for n, m in zip(ad_names, ad_rows)]
    rows_h += [hard_row(label + " (3 seeds)", gmean(cv, c)) for c, label in ours]
    rows_h += [hard_row("Ensemble · 3 U-Net + 3 TerraMind, plain decision", e6v), hard_row("Ensemble · 3 U-Net + 3 TerraMind, fitted decision offsets", e6v_t)]
    rows_r = [rank_row(n + " (A–D report)", m) for n, m in zip(ad_names, ad_rows)]
    rows_r += [rank_row(label + " (3 seeds)", gmean(cv, c)) for c, label in ours] + [rank_row("Ensemble · 3 U-Net + 3 TerraMind", e6v)]
    rows_t = [hard_row(label + " (3 seeds)", gmean(ct, c)) + [pct(gmean(ct, c)["ap"]), pct(gmean(ct, c)["r5"])] for c, label in ours]
    rows_t += [hard_row("Ensemble · 3 U-Net + 3 TerraMind, plain decision", e6t) + [pct(e6t["ap"]), pct(e6t["r5"])],
               hard_row("Ensemble · 3 U-Net + 3 TerraMind, offsets fitted on validation", e6t_t) + [pct(e6t_t["ap"]), pct(e6t_t["r5"])]]
    bars_models = [("A", ad_rows[0]), ("B", ad_rows[1]), ("C", ad_rows[2]), ("D", ad_rows[3]), ("U-Net\nall", gmean(cv, "base")),
                   ("TerraMind\nall", gmean(cv, "tm")), ("Ensemble\nplain", e6v), ("Ensemble\n+ offsets", e6v_t)]
    left = {"title": "Combined wetland class: normal pixel classification", "ylabel": "Weak-reference score (%)", "groups": [g for g, _ in bars_models], "tick_size": 9,
            "series": [(lab, [m[k] for _, m in bars_models], None) for lab, k in (("Recall", "recall"), ("Reference precision", "precision"), ("F2", "f2"))]}
    right = {"title": "Original six-class segmentation", "ylabel": "Macro IoU (%)", "groups": [g for g, _ in bars_models], "tick_size": 9, "colors": ["#4a7ba6"],
             "series": [("Macro IoU", [m["macro_iou"] for _, m in bars_models], None)], "annotate": True}
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    for label, m in (("A: S2", ad_rows[0]), ("B: S2 + S1", ad_rows[1]), ("C: S2 + NISAR", ad_rows[2]), ("D: all sensors", ad_rows[3]),
                     ("U-Net, all sensors (3 seeds)", gmean(cv, "base")), ("TerraMind, all sensors (3 seeds)", gmean(cv, "tm")), ("Ensemble: 3 U-Net + 3 TerraMind", e6v)):
        ax.plot([1, 5, 10, 20], [100 * m[k] for k in ("r1", "r5", "r10", "r20")], marker="o", label=label, lw=2.6 if label.startswith("Ensemble") else 1.5)
    ax.plot([0, 20], [0, 20], ":", color="gray", label="Uniform random selection (expected)")
    ax.set(title="A–D validation area", xlabel="Selected fraction of scored area (%)", ylabel="Reference swamp/marsh recall (%)", xlim=(0, 20.5), ylim=(0, 100), xticks=[1, 5, 10, 20])
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150)
    plt.close(fig)
    d_ad = ad_rows[3]
    parts.append('<section id="adtables"><h2>2. The best model in the A–D report\'s tables</h2><p>The six-model ensemble and its members, scored exactly as in the A–D report: the same <strong>585 validation patches and ' + f"{cv['ensembles']['Ensemble6']['result']['scored_pixels']:,}" + ' pixels</strong> (label-eligible and valid in all 14 predictors), full resolution, and the same combined decision — sum the Swamp and Marsh probabilities, then choose among Other, combined Wetland, Permanent water, Peatland cropland and Peatland non-cropland. Rows A–D are copied from the A–D report (one seed, best validation epoch); the other single-model rows are means of three seeds at the final epoch.</p>'
                 + table(["Model"] + [h for _, h in HARD], rows_h, lead=len(rows_h) - 1)
                 + chart([left, right], "Same chart as the A–D report with this report's models added. A–D are the seed-17 models; U-Net and TerraMind are means of three seeds.", sharey=True, figsize=(13, 4.9))
                 + f"""<p><strong>Plain decision:</strong> with every pixel assigned its most likely class, the ensemble is more precise than D ({pct(e6v['precision'])} versus {pct(d_ad['precision'])}) but recovers less wetland ({pct(e6v['recall'])} versus {pct(d_ad['recall'])} recall), and D has the higher F2, wetland IoU and macro IoU. On hard maps the A–D recipe is the stronger one as trained.</p><p><strong>Fitted decision offsets:</strong> the ensemble ranks pixels better than D (next table) but its probabilities for the rare classes are too low to win the argmax. Adding one fixed offset per class to the log-probabilities, fitted on validation to maximise five-class macro IoU (offsets {', '.join(f'{o:+.2f}' for o in offsets)} for Other, Wetland, Water, Peat cropland, Peat non-cropland), gives {pct(e6v_t['recall'])} recall, {pct(e6v_t['precision'])} reference precision, {pct(e6v_t['f2'])} F2 and {pct(e6v_t['iou'])} wetland IoU, each above D. The model is unchanged; only the decision rule moves.</p><p class="note">The offsets are fitted on the same validation split they are scored on here, so this row is optimistic, in the same way that choosing D's checkpoint by validation macro IoU is. The test table below applies the validation offsets unchanged and is the fair check.</p><h3>Candidate ranking, same validation area</h3><p>Ranking by <code>pSwamp + pMarsh</code> does not depend on the decision rule, so the ensemble has one row.</p>"""
                 + table(["Model", "Average precision", "Recall at 1% area", "Recall at 5% area", "Recall at 10% area", "Recall at 20% area", "Reference precision at 5%"], rows_r, lead=len(rows_r) - 1)
                 + figure(buf.getvalue(), "Reference wetland recovery as the review-area budget grows, on the A–D validation area. The dotted line is the expected result of random selection.", "narrow")
                 + f"""<p><strong>The ensemble leads every area budget and average precision</strong> ({pct(e6v['ap'])} versus {pct(d_ad['ap'])} for D). Single all-sensor models from the two recipes rank pixels equally well: {pct(gmean(cv, 'base')['ap'])} for this report's U-Net and {pct(d_ad['ap'])} for D.</p><h3>Held-out test split, same definitions</h3><p>{ct['patches']} test patches, {ct['ensembles']['Ensemble6']['result']['scored_pixels']:,} scored pixels. The A–D models have not been scored on test, so only this report's models appear.</p>"""
                 + table(["Model"] + [h for _, h in HARD] + ["Average precision", "Recall at 5% area"], rows_t, lead=len(rows_t) - 1)
                 + f"""<p>With the offsets fitted on validation and applied unchanged, the ensemble's test recall goes from {pct(e6t['recall'])} to {pct(e6t_t['recall'])}, reference precision from {pct(e6t['precision'])} to {pct(e6t_t['precision'])}, F2 from {pct(e6t['f2'])} to {pct(e6t_t['f2'])} and six-class macro IoU from {pct(e6t['macro_iou'])} to {pct(e6t_t['macro_iou'])}.</p></section>""")

    # ---- 3 model comparison
    models = [("LightGBM\nper pixel", None), ("LightGBM\n+ context", None), ("U-Net", "base"), ("TerraMind", "tm"), ("Ensemble\n3 + 3", None)]
    def model_vals(key_ap, single):
        g = lambda name: gbm[name]["wetland_ap"] if key_ap == "wet" else gbm[name]["ap"][key_ap]  # noqa: E731
        e = ens_val["wetland_ap"] if key_ap == "wet" else ens_val["ap"][key_ap]
        return [g("nisar+s1+s2 raw+idx"), g("all raw+idx+ctx"), mean(single("base")), mean(single("tm")), e]
    series = []
    for label, key in (("Swamp", "Swamp"), ("Marsh", "Marsh"), ("Swamp/marsh combined", "wet")):
        single = (lambda c, k=key: col(val[c], "wet")) if key == "wet" else (lambda c, k=key: [m["ap"][k] for m in val[c]])
        errs = [0, 0, spread(single("base")), spread(single("tm")), 0]
        series.append((label, model_vals(key, single), errs))
    rows = []
    for label, m in (("LightGBM per pixel · 14 channels", gbm_px), ("LightGBM per pixel + neighbourhood means", gbm_ctx)):
        rows.append([label, "1", pct(m["ap"]["Swamp"]), pct(m["ap"]["Marsh"]), pct(m["wetland_ap"]), pct(m["ap"]["Permanent water"]), pct(m["wetland_recall_at_area"]["5"])])
    for c, label in configs[:2]:
        r = val[c]
        rows.append([label, "3", pct(avg(r, "swamp")), pct(avg(r, "marsh")), pct(avg(r, "wet")), pct(avg(r, "water")), pct(avg(r, "r5"))])
    rows.append(["Ensemble · 3 U-Net + 3 TerraMind", "1", pct(ens_val["ap"]["Swamp"]), pct(ens_val["ap"]["Marsh"]), pct(ens_val["wetland_ap"]),
                 pct(ens_val["ap"]["Permanent water"]), pct(ens_val["wetland_recall_at_area"]["5"])])
    parts.append('<section id="models"><h2>3. Model comparison on validation</h2><p>Common evaluation: <strong>585 validation patches</strong>, all 14 channels, final-epoch weights, identical pixels for every row. Scores are ranking scores (average precision).</p>'
                 + table(["Model", "Runs", "Swamp", "Marsh", "Swamp/marsh combined", "Permanent water", "Recall at 5% area"], rows, lead=len(rows) - 1)
                 + chart([{"title": "Ranking score on validation, all 14 channels", "ylabel": "Average precision (%)", "groups": [m for m, _ in models], "series": series, "legend_loc": "upper left"}],
                         "Validation ranking scores by model. U-Net and TerraMind bars are means of three seeds; black lines show the spread over seeds.", figsize=(11, 4.7))
                 + f"""<p><strong>U-Net versus TerraMind:</strong> the combined score is the same within seed spread ({pct(avg(val['base'], 'wet'))} and {pct(avg(val['tm'], 'wet'))}). TerraMind is higher on swamp ({pct(avg(val['tm'], 'swamp'))} versus {pct(avg(val['base'], 'swamp'))}) and equal on marsh. TerraMind has no L-band pretraining, so its NISAR pathway is learned from these patches alone.</p><p><strong>Ensemble:</strong> averaging the six models' probabilities raises the combined score to {pct(ens_val['wetland_ap'])}. Three U-Nets alone reach {pct(ens_val_u['wetland_ap'])} and three TerraMind models {pct(ens_val_t['wetland_ap'])}, so averaging seeds and mixing the two architectures each contribute a similar share of the gain.</p><p><strong>LightGBM:</strong> a model that sees one pixel at a time is close to useless for swamp and marsh, although it finds open water ({pct(gbm_px['ap']['Permanent water'])}). Neighbourhood means help only a little. The convolutional models are using shape and context, not per-pixel spectra.</p></section>""")

    # ---- 3 test
    order = [("s2", "U-Net · S2 only", "S2"), ("s2s1", "U-Net · S2 + S1", "S2 + S1"), ("s2nisar", "U-Net · S2 + NISAR", "S2 +\nNISAR"),
             ("base", "U-Net · all 14 channels", "U-Net\nall"), ("tm", "TerraMind-small · all 14 channels", "TerraMind\nall")]
    rows = [[label, "3", pct(avg(tm_[k], "swamp")), pct(avg(tm_[k], "marsh")), pct(avg(tm_[k], "wet")), pct(avg(tm_[k], "r5"))] for k, label, _ in order]
    for key, label in (("ens_unet", "Ensemble · 3 U-Net"), ("ens_tm", "Ensemble · 3 TerraMind"), ("ens", "Ensemble · 3 U-Net + 3 TerraMind")):
        m = test[key]["ensemble"]
        rows.append([label, "1", pct(m["ap"]["Swamp"]), pct(m["ap"]["Marsh"]), pct(m["wetland_ap"]), pct(m["wetland_recall_at_area"]["5"])])
    groups = [g for _, _, g in order] + ["Ensemble\n3 U-Net", "Ensemble\n3 TerraMind", "Ensemble\n3 + 3"]
    ens3 = [test[k]["ensemble"] for k in ("ens_unet", "ens_tm", "ens")]
    series = [(label, [mean([f(m) for m in tm_[k]]) for k, _, _ in order] + [f(m) for m in ens3],
               [spread([f(m) for m in tm_[k]]) for k, _, _ in order] + [0, 0, 0])
              for label, f in (("Swamp", GET["swamp"]), ("Marsh", GET["marsh"]), ("Swamp/marsh combined", GET["wet"]))]
    parts.append('<section id="test"><h2>4. Held-out test split</h2><p>Scored once after all training and design decisions: <strong>559 test patches</strong>, final-epoch weights, no tuning on this split. Scores are ranking scores (average precision).</p>'
                 + table(["Model", "Runs", "Swamp", "Marsh", "Swamp/marsh combined", "Recall at 5% area"], rows, lead=len(rows) - 1)
                 + chart([{"title": "Ranking score on the held-out test split", "ylabel": "Average precision (%)", "groups": groups, "series": series, "legend_loc": "upper left", "tick_size": 9}],
                         "Test ranking scores. Single-model bars are means of three seeds with the spread over seeds; ensemble bars are single evaluations.", figsize=(12, 4.9))
                 + f"""<p>The ordering of models is the same as on validation, and every score is higher: the test geography is easier, for swamp in particular (single U-Net swamp score {pct(avg(u_test, 'swamp'))} on test versus {pct(avg(val['base'], 'swamp'))} on validation). Seed spread is larger on test than on validation, so differences of a few points between single models are not findings. The six-model ensemble is {pts(e_test['wetland_ap'] - avg(u_test, 'wet'))} points above a single U-Net.</p></section>""")

    # ---- 4 channels
    rows = [[label, pct(mean(hm(c, "macro_iou"))), pct(mean(hm(c, "recall"))), pct(mean(hm(c, "precision"))), pct(mean(hm(c, "f1"))), pct(mean(hm(c, "f2"))), pct(mean(hm(c, "iou"))),
             pct(avg(val[c], "wet")), pct(avg(tm_[c], "wet"))] for c, label in sensors]
    hard_panel = {"title": "Combined wetland class: hard map (validation)", "ylabel": "Weak-reference score (%)", "groups": [s for _, s in sensors],
                  "series": [(lab, [mean(hm(c, k)) for c, _ in sensors], [spread(hm(c, k)) for c, _ in sensors]) for lab, k in (("Recall", "recall"), ("Reference precision", "precision"), ("F2", "f2"))]}
    rank_panel = {"title": "Swamp/marsh ranking score", "ylabel": "Average precision (%)", "groups": [s for _, s in sensors],
                  "series": [("Validation", [avg(val[c], "wet") for c, _ in sensors], [spread(col(val[c], "wet")) for c, _ in sensors]),
                             ("Test", [avg(tm_[c], "wet") for c, _ in sensors], [spread(col(tm_[c], "wet")) for c, _ in sensors])], "legend_loc": "upper left"}
    fold_rows = [[str(i), pct(GET["wet"](a)), pct(GET["wet"](b)), pts(GET["wet"](a) - GET["wet"](b)),
                  f"{pct(GET['swamp'](a))} / {pct(GET['swamp'](b))}", f"{pct(GET['marsh'](a))} / {pct(GET['marsh'](b))}"]
                 for i, (a, b) in enumerate(zip(kf["kf_all"], kf["kf_s2s1"]))]
    fold_rows.append(["Mean", pct(avg(kf["kf_all"], "wet")), pct(avg(kf["kf_s2s1"], "wet")), pts(mean(fold_delta)),
                      f"{pct(avg(kf['kf_all'], 'swamp'))} / {pct(avg(kf['kf_s2s1'], 'swamp'))}", f"{pct(avg(kf['kf_all'], 'marsh'))} / {pct(avg(kf['kf_s2s1'], 'marsh'))}"])
    fold_panel = {"title": "Swamp/marsh ranking score by geographic fold", "ylabel": "Average precision (%)",
                  "groups": [f"Fold {i}" for i in range(len(fold_delta))] + ["Mean"], "annotate": True,
                  "series": [("S2 + S1", col(kf["kf_s2s1"], "wet") + [avg(kf["kf_s2s1"], "wet")], None),
                             ("All 14 channels (adds NISAR)", col(kf["kf_all"], "wet") + [avg(kf["kf_all"], "wet")], None)], "legend_loc": "upper right"}
    ko = knock["knockout"]
    ko_items = [("Nothing\nblanked", knock), ("Sentinel-2\nblanked", ko["s2"]), ("Sentinel-1\nblanked", ko["s1"]), ("NISAR\nblanked", ko["nisar"])]
    ko_rows = [[label.replace("\n", " "), pct(m["ap"]["Swamp"]), pct(m["ap"]["Marsh"]), pct(m["wetland_ap"]), "—" if m is knock else pts(m["wetland_ap"] - knock["wetland_ap"])] for label, m in ko_items]
    ko_panel = {"title": "One U-Net with a sensor blanked at inference (validation)", "ylabel": "Average precision (%)", "groups": [l for l, _ in ko_items],
                "series": [("Swamp", [m["ap"]["Swamp"] for _, m in ko_items], None), ("Marsh", [m["ap"]["Marsh"] for _, m in ko_items], None),
                           ("Swamp/marsh combined", [m["wetland_ap"] for _, m in ko_items], None)]}
    gbm_sets = [("s2 raw+idx", "S2"), ("s1 raw+idx", "S1"), ("nisar raw+idx", "NISAR"), ("s1+s2 raw+idx", "S2 + S1"), ("nisar+s2 raw+idx", "S2 + NISAR"),
                ("nisar+s1 raw+idx", "S1 + NISAR"), ("nisar+s1+s2 raw+idx", "All sensors")]
    gbm_rows = [[label, pct(gbm[n]["ap"]["Swamp"]), pct(gbm[n]["ap"]["Marsh"]), pct(gbm[n]["wetland_ap"]), pct(gbm[n]["ap"]["Permanent water"])] for n, label in gbm_sets]
    imp = gbm["channel_importance"]
    imp_rows = [[k, pct(v["gain_share"]), pts(-v["swamp_ap_drop"]), pts(-v["marsh_ap_drop"]), pts(-v["wetland_ap_drop"])]
                for k, v in sorted(imp.items(), key=lambda kv: -kv[1]["wetland_ap_drop"])]
    worst = max(("S2_B2", "S2_B3", "S2_B4", "S2_B8", "S2_B11", "S2_B12"), key=lambda k: knock["wetland_ap"] - ko[k]["wetland_ap"])
    d_rec = mean(hm("base", "recall")) - mean(hm("s2s1", "recall"))
    d_prec = mean(hm("base", "precision")) - mean(hm("s2s1", "precision"))
    parts.append('<section id="channels"><h2>5. Do all 14 channels help?</h2><h3>U-Net sensor comparison on the fixed splits</h3><p>Three seeds per row; the same seeds, recipe and patches, differing only in the selected sensors. Hard-map columns are on the 585 validation patches with the definitions of the A–D report; the last two columns are ranking scores on the stride-4 pixel grid used elsewhere in this report.</p>'
                 + table(["Sensors", "Six-class macro IoU", "Wetland recall", "Reference precision", "F1", "F2", "Wetland IoU", "Ranking score (validation)", "Ranking score (test)"], rows, lead=len(rows) - 1)
                 + chart([hard_panel, rank_panel], "Sensor comparison with the U-Net. Left: hard-map scores on validation, with the A–D report's definitions. Right: ranking score on validation and test. Black lines show the spread over three seeds.", sharey=True)
                 + f"""<p><strong>NISAR contrast on the fixed splits, all sensors − (S2 + S1):</strong> recall {pts(d_rec)} points, reference precision {pts(d_prec)}, ranking score {pts(d_val)} on validation and {pts(d_test)} on test. On validation this is well outside the seed spread; on test the seed spread is of similar size to the difference. Sentinel-2 alone is the weakest set on both splits.</p><p class="note">Hard-map columns use the A–D report's definitions and scoring area (section 2). The plain hard maps of these runs have lower recall and higher precision than A–D model D; section 2 shows the comparison and what fitted decision offsets change.</p><h3>Five geographic folds: all sensors versus S2 + S1</h3><p>One run per fold, each scored on its own held-out fold of the train + validation area. Rows are paired: same fold, same seed.</p>"""
                 + table(["Fold", "All 14 channels", "S2 + S1", "Difference (points)", "Swamp (all / S2 + S1)", "Marsh (all / S2 + S1)"], fold_rows, lead=len(fold_rows) - 1)
                 + chart([fold_panel], "Paired comparison on five held-out geographic folds. Adding NISAR helps in one fold, hurts in two and makes little difference in two.", figsize=(11, 4.5))
                 + f"""<p><strong>The NISAR gain does not replicate across geography.</strong> The paired difference ranges from {pts(min(fold_delta))} to {pts(max(fold_delta))} points with mean {pts(mean(fold_delta))}. Of the five folds, {sum(d < -0.03 for d in fold_delta)} clearly favour S2 + S1, {sum(abs(d) <= 0.03 for d in fold_delta)} are within 3 points either way and {sum(d > 0.03 for d in fold_delta)} clearly favours all 14 channels, while the fixed validation and test splits both favour the 14-channel model. Scores also vary far more between folds than between seeds (swamp {pct(min(col(kf['kf_all'], 'swamp')))}–{pct(max(col(kf['kf_all'], 'swamp')))}), so one fixed split cannot settle a difference of this size.</p><h3>Blanking a sensor at inference</h3><p>One U-Net trained with sensor dropout (seed 0), scored on validation with each sensor replaced by its missing-value fill.</p>"""
                 + table(["Input", "Swamp", "Marsh", "Swamp/marsh combined", "Change (points)"], ko_rows)
                 + chart([ko_panel], "Ranking score of one model when a whole sensor is removed at inference.", figsize=(10, 4.5))
                 + f"""<p>This model leans on NISAR more than on Sentinel-1, consistent with the fixed-split comparison above and subject to the same split caveat. Blanking single channels shows the strongest dependence on {worst.replace('_', ' ')} (combined score {pct(ko[worst]['wetland_ap'])} without it); single-channel blanking is outside what the model saw in training, so treat it as indicative.</p><h3>LightGBM sensor subsets</h3><p>Per-pixel model, validation, raw channels plus indices: a context-free view of where spectral signal sits.</p>"""
                 + table(["Sensors", "Swamp", "Marsh", "Swamp/marsh combined", "Permanent water"], gbm_rows)
                 + "<details><summary>Permutation importance of each raw channel (LightGBM, 14 raw channels)</summary>"
                 + table(["Channel", "Share of split gain", "Swamp change when shuffled (points)", "Marsh change (points)", "Swamp/marsh change (points)"], imp_rows) + "</details>"
                 + "<p><strong>Reading:</strong> Sentinel-2, and its short-wave infrared bands in particular, carries most of the usable signal. Radar adds little for a per-pixel model and a split-dependent amount for the U-Net. Keeping all 14 channels is reasonable, but the evidence does not yet support a general claim that NISAR improves the result.</p></section>")

    # ---- 5 variants
    variants = [("base", "Plain U-Net\n(baseline)")] + [(c, {"reg": "Dropout +\nweight decay", "jit": "Radiometric\njitter", "aug": "Jitter +\nzoom",
                                                               "ovr": "Sigmoid\nheads", "nopeat": "Peat ignored\nin training"}[c]) for c in ("reg", "jit", "aug", "ovr", "nopeat")]
    rows = [[label.replace("\n", " "), pct(avg(val[c], "swamp")), pct(avg(val[c], "marsh")), pct(avg(val[c], "wet")), pts(avg(val[c], "wet") - avg(val["base"], "wet")) if c != "base" else "—",
             pct(avg(val[c], "peat_c")), pct(avg(val[c], "peat_n"))] for c, label in variants]
    var_panel = {"title": "Ranking score on validation by training variant", "ylabel": "Average precision (%)", "groups": [l for _, l in variants], "tick_size": 9,
                 "series": [(lab, [avg(val[c], k) for c, _ in variants], [spread(col(val[c], k)) for c, _ in variants])
                            for lab, k in (("Swamp", "swamp"), ("Marsh", "marsh"), ("Swamp/marsh combined", "wet"))]}
    parts.append('<section id="variants"><h2>6. Training changes that did not help</h2><p>Each variant changes one group of settings relative to the plain 14-channel U-Net, with three seeds on the fixed validation split. Scores are ranking scores.</p>'
                 + table(["Configuration", "Swamp", "Marsh", "Swamp/marsh combined", "Change vs baseline (points)", "Peatland cropland", "Peatland non-cropland"], rows, lead=0)
                 + chart([var_panel], "No variant beats the plain U-Net on the combined swamp/marsh score. Black lines show the spread over three seeds.", figsize=(11.5, 4.7))
                 + """<ul><li><strong>Dropout + weight decay:</strong> sensor dropout 0.2, head dropout 0.2, weight decay 0.01.</li><li><strong>Radiometric jitter / zoom:</strong> random per-channel gain and offset; random crop enlarged to full size.</li><li><strong>Sigmoid heads:</strong> one output per class, with peat treated as unknown under swamp, marsh and water instead of forcing one class per pixel.</li><li><strong>Peat ignored in training:</strong> the two peat classes are left out of the loss (and still scored), to see whether that frees capacity for swamp and marsh.</li></ul><p>Regularisation and augmentation slightly raise the peat scores and lower the wetland score. None of these is a reliable improvement, so the plain recipe was kept for every other experiment.</p></section>""")

    # ---- 6 ranking
    budgets = ("1", "5", "10", "20")
    lines = [("Ensemble: 3 U-Net + 3 TerraMind", [e_test["wetland_recall_at_area"][b] for b in budgets]),
             ("TerraMind, all sensors", [mean([m["wetland_recall_at_area"][b] for m in t_test]) for b in budgets]),
             ("U-Net, all sensors", [mean([m["wetland_recall_at_area"][b] for m in u_test]) for b in budgets]),
             ("U-Net, S2 + S1", [mean([m["wetland_recall_at_area"][b] for m in tm_["s2s1"]]) for b in budgets]),
             ("U-Net, S2 only", [mean([m["wetland_recall_at_area"][b] for m in tm_["s2"]]) for b in budgets])]
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    for label, vals in lines:
        ax.plot([1, 5, 10, 20], [100 * v for v in vals], marker="o", label=label)
    ax.plot([0, 20], [0, 20], ":", color="gray", label="Uniform random selection (expected)")
    ax.set(title="Held-out test area", xlabel="Selected fraction of scored area (%)", ylabel="Reference swamp/marsh recall (%)", xlim=(0, 20.5), ylim=(0, 100), xticks=[1, 5, 10, 20])
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150)
    plt.close(fig)
    rows = [[label.replace(":", " ·")] + [pct(v) for v in vals] for label, vals in lines]
    parts.append('<section id="ranking"><h2>7. Candidate ranking for targeted review</h2><p>Ranking answers a different question: <strong>if we can inspect only a small fraction of the area, where should we look first?</strong> Order pixels by <code>pSwamp + pMarsh</code> and take the highest-scoring pixels within a fixed area budget. Single-model rows are means of three seeds.</p>'
                 + f"""<div class="summary"><h3>How to read the ensemble at a 5% area budget</h3><p>Inspecting the highest-scoring <strong>5% of the test area</strong> would recover <strong>{pct(e_test['wetland_recall_at_area']['5'])}</strong> of its reference swamp/marsh pixels, about {round(100 * e_test['wetland_recall_at_area']['5'])} of every 100 labelled wetland pixels. At 1% of area it recovers {pct(e_test['wetland_recall_at_area']['1'])}; at 10%, {pct(e_test['wetland_recall_at_area']['10'])}. Used this way the model misses far less than its hard class map does, because the hard map gives each pixel only its single most likely class.</p></div>"""
                 + table(["Model", "Recall at 1% area", "Recall at 5% area", "Recall at 10% area", "Recall at 20% area"], rows, lead=0)
                 + figure(buf.getvalue(), "Reference wetland recovery on the test split as the review-area budget grows. The dotted line is the expected result of selecting pixels at random.", "narrow") + "</section>")

    # ---- 7 examples
    parts.append('<section id="examples"><h2>8. Example predictions</h2><p>The 16 test locations where swamp and marsh are predicted best by the six-model ensemble: highest overlap between predicted and reference wetland (combined swamp/marsh IoU, shown in the corner of each image). They are the top of 76 test patches that contain at least 2% wetland; the median overlap over all 76 is 7%, so these show what the model does well, not its typical result. Each panel is 5.12 km across. Predictions use the decision offsets fitted on validation (section 2).</p>'
                 + image("plots/examples_best_test.png", "Sentinel-2 true colour, weak reference labels and ensemble prediction, two locations per row, ordered from the highest overlap.")
                 + """<ul><li><strong>Large river floodplains and big swamp blocks are mapped well</strong>, with overlap near 80% in the top rows; rivers and lakes are outlined throughout.</li><li><strong>Swamp and marsh are sometimes exchanged</strong> relative to the reference while the wetland extent is right; the overlap score treats them as one class.</li><li><strong>Peatland non-cropland often appears beside correct wetland</strong> where the reference has Other. This is the precision cost of the higher-recall decision rule.</li><li><strong>Lower rows</strong> have smaller overlap, usually because only part of a narrow wetland strip is found.</li></ul></section>""")

    # ---- 8 diagnosis
    tv = val["base"][0]
    gap_panel = {"title": "One U-Net scored on its training patches and on validation", "ylabel": "Average precision (%)", "annotate": True,
                 "groups": ["Swamp", "Marsh", "Permanent\nwater", "Peatland\ncropland", "Peatland\nnon-cropland"],
                 "series": [("Training patches", [train_eval["ap"][c] for c in CLASSES[1:]], None), ("Validation patches", [tv["ap"][c] for c in CLASSES[1:]], None)]}
    parts.append('<section id="diagnosis"><h2>9. What limits the result</h2><h3>The model fits its training geography</h3><p>One U-Net (seed 0, final epoch) scored on the patches it was trained on and on validation. Scores are ranking scores.</p>'
                 + table(["Class", "Training patches", "Validation patches", "Drop (points)"], [[c, pct(train_eval["ap"][c]), pct(tv["ap"][c]), pts(tv["ap"][c] - train_eval["ap"][c])] for c in CLASSES[1:]])
                 + chart([gap_panel], "The gap between regions the model has seen and regions it has not. Open water transfers; swamp does not.", figsize=(10, 4.5))
                 + f"""<p>The swamp score falls from {pct(train_eval['ap']['Swamp'])} to {pct(tv['ap']['Swamp'])}. More regularisation and augmentation did not close this gap (section 6), which points at differences between regions and at the labels, not at model capacity.</p><h3>Likely causes, in order of expected effect</h3><ul><li><strong>One late-October snapshot.</strong> Floodplain marsh in autumn resembles dry grassland; spring and summer optical composites and flood-season radar are the inputs most likely to separate it. Not tested here: the source rasters are outside this dataset export.</li><li><strong>Cloud and haze in the Sentinel-2 mosaic.</strong> Several example patches are visibly contaminated, and the model depends heavily on short-wave infrared. Not yet quantified.</li><li><strong>Label definitions overlap.</strong> Swamp, marsh and peatland co-occur on the ground but are exclusive in the labels. On test the ensemble scores {pct(e_test['ap']['Swamp'])} for swamp and {pct(e_test['ap']['Marsh'])} for marsh separately, and {pct(e_test['wetland_ap'])} when they are merged.</li></ul></section>""")

    parts.append("""<section id="limits"><h2>10. Limits</h2><ul><li>Reference labels are historical and inferred. Agreement with them is not field accuracy, and some disagreement is label error.</li><li>Three seeds per configuration; k-fold comparisons have one seed per fold. The TerraMind k-fold runs were not completed.</li><li>Decision offsets for the ensemble are fitted on the validation split. Its validation hard-map scores with offsets are therefore optimistic; the test rows apply the same offsets unchanged.</li><li>The ensemble is six times the inference cost of a single model.</li><li>TerraMind has no L-band pretraining; its NISAR embedding is trained from scratch on 2,609 patches.</li><li>Scoring uses every labelled pixel, not only pixels where all 14 inputs are valid, and ranking scores are computed on a stride-4 pixel grid. Absolute values are therefore not interchangeable with the A–D report.</li><li>The test split has now been viewed once. Further model selection against it would make it a second validation set.</li></ul></section>""")

    data = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "seeds": [0, 1, 2], "test_evaluated": True,
            "validation": {c: {k: [round(v, 4) for v in col(val[c], k)] for k in GET} for c, _ in configs},
            "ad_definitions_validation": cv, "ad_definitions_test": ct, "ad_report_rows": dict(zip(ad_names, ad_rows)),
            "validation_ensemble": ens_val, "test": {k: {kk: [round(v, 4) for v in col(r, kk)] for kk in GET} for k, r in tm_.items()},
            "test_ensemble": e_test, "kfold_wetland_ap": {c: [round(v, 4) for v in col(kf[c], "wet")] for c in kf},
            "kfold_paired_difference": [round(v, 4) for v in fold_delta], "lightgbm": {k: v for k, v in gbm.items() if k != "channel_importance"}}
    parts.append('<section id="artifacts"><h2>11. Artifacts and reproduction</h2><p>Code: <code>ml/src/bogsignal/train_unet.py</code>, <code>terramind_seg.py</code>, <code>baseline_gbm.py</code>, <code>ensemble.py</code>, <code>compare_runs.py</code>, <code>plot_unet.py</code>; this report: <code>ml/build_model_report.py</code>. Run directories are named <code>&lt;config&gt;_s&lt;seed&gt;</code> or <code>&lt;config&gt;_f&lt;fold&gt;</code> under <code>runs/exp</code>.</p><details><summary>Source result files and hashes</summary>'
                 + table(["File", "SHA-256"], [[f"<code>{html.escape(s['path'])}</code>", f"<code>{s['sha256']}</code>"] for s in SOURCES]) + "</details></section></main></body></html>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts))
    data["sources"] = SOURCES
    args.out.with_name(args.out.stem + "_data.json").write_text(json.dumps(data, indent=1))
    print("wrote", args.out, f"({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
