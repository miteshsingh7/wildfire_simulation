"""Phase 1B-2 Part B: Multi-source perturbed uncertainty ensemble for routing safety margin.

- Ensemble over seeds AND parameter/forcing perturbations:
  * P_base band (+- dP_base)
  * Wind speed +-15%
  * Wind direction +-15 deg
  * Ember params (+-35% on alpha_launch, +-17.5% on beta_dist)
  * Ignition-extent morphological jitter (+-k cells at T0)
- Tune perturbation sizes on training folds ONLY (Kincade, Glass, Zogg LOFO), targeting
  ~80% P10-P90 coverage of observed arrival in rolling mode (strictly-future progression cells).
- Report coverage with denominators per fire, rolling mode, held-out folds (no cumulative metrics).
"""

from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from wf052.sim.ember_kernel import EmberParams
from wf052.sim.rolling_protocol import (
    PerturbationConfig,
    get_active_rolling_origins,
    run_perturbed_rolling_window,
)
from wf052.sim.teacher_calibrator import FireDataset60m, TeacherParams, load_fire_dataset_60m


def evaluate_fire_rolling_coverage(
    ds: FireDataset60m,
    base_teacher: TeacherParams,
    base_ember: EmberParams,
    use_ember_kernel: bool,
    perturb_cfg: PerturbationConfig,
    max_origins: int = 4,
) -> dict[str, Any]:
    """Evaluate rolling 6h P10-P90 arrival coverage pooled across every 3rd GOFER perimeter."""
    origins = get_active_rolling_origins(ds)[:max_origins]
    tot_denom = 0
    tot_cov_quant = 0
    tot_cov_raw = 0

    for t0_h in origins:
        res = run_perturbed_rolling_window(
            ds=ds,
            base_teacher=base_teacher,
            base_ember=base_ember,
            use_ember_kernel=use_ember_kernel,
            t0_h=t0_h,
            duration_h=6,
            perturb_cfg=perturb_cfg,
            base_seed=1000,
        )
        tot_denom += int(res["denom_obs_progression_cells"])
        tot_cov_quant += int(res["covered_cells"])
        tot_cov_raw += int(res["raw_covered_cells"])

    quant_pct = (100.0 * tot_cov_quant / tot_denom) if tot_denom > 0 else 0.0
    raw_pct = (100.0 * tot_cov_raw / tot_denom) if tot_denom > 0 else 0.0

    return {
        "rolling_origins_evaluated": len(origins),
        "denom_obs_progression_cells": tot_denom,
        "covered_cells": tot_cov_quant,
        "coverage_pct": quant_pct,
        "coverage_with_denom": f"{tot_cov_quant}/{tot_denom} ({quant_pct:.1f}%)",
        "raw_covered_cells": tot_cov_raw,
        "raw_coverage_pct": raw_pct,
        "raw_coverage_with_denom": f"{tot_cov_raw}/{tot_denom} ({raw_pct:.1f}%)",
    }


def tune_perturbation_config_on_wind_folds(
    cov_cache: dict[tuple[str, PerturbationConfig], dict[str, Any]],
    train_fires: list[FireDataset60m],
    base_teacher: TeacherParams,
    base_ember: EmberParams,
    target_coverage_pct: float = 80.0,
) -> PerturbationConfig:
    """Tune perturbation amplitudes (P_base half-width and init_jitter_cells) on training folds only."""
    candidates = [
        PerturbationConfig(num_members=20, p_base_half_width=0.040, wind_speed_frac=0.15, wind_dir_deg=15.0, ember_frac=0.30, init_jitter_cells=2),
        PerturbationConfig(num_members=20, p_base_half_width=0.060, wind_speed_frac=0.15, wind_dir_deg=15.0, ember_frac=0.40, init_jitter_cells=3),
        PerturbationConfig(num_members=20, p_base_half_width=0.080, wind_speed_frac=0.15, wind_dir_deg=15.0, ember_frac=0.50, init_jitter_cells=4),
        PerturbationConfig(num_members=20, p_base_half_width=0.100, wind_speed_frac=0.15, wind_dir_deg=15.0, ember_frac=0.60, init_jitter_cells=5),
    ]
    best_cfg = candidates[1]
    best_err = float("inf")

    for cand in candidates:
        covs = []
        for ds in train_fires:
            key = (ds.fire_name, cand)
            if key not in cov_cache:
                cov_cache[key] = evaluate_fire_rolling_coverage(ds, base_teacher, base_ember, True, cand, max_origins=4)
            covs.append(float(cov_cache[key]["coverage_pct"]))
        mean_cov = float(np.mean(covs))
        err = abs(mean_cov - target_coverage_pct)
        if err < best_err:
            best_err = err
            best_cfg = cand

    return best_cfg


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Phase 1B-2 Part B (Perturbed Uncertainty Ensemble) on {device}...", flush=True)
    t0 = time.perf_counter()

    cfg_v2_path = Path("configs/teacher_v2.yaml")
    if cfg_v2_path.exists():
        with open(cfg_v2_path) as f:
            v2_cfg = yaml.safe_load(f)
        base_tp = TeacherParams(p_base=float(v2_cfg["contact_parameters"]["p_base"]))
        ep_d = v2_cfg["ember_parameters"]
        base_ep = EmberParams(
            alpha_launch=float(ep_d["alpha_launch"]),
            beta_dist_m_per_ms=float(ep_d["beta_dist_m_per_ms"]),
            p_urban_wui=float(ep_d["p_urban_wui"]),
        )
    else:
        base_tp = TeacherParams(p_base=0.20)
        base_ep = EmberParams(alpha_launch=0.0020, beta_dist_m_per_ms=110.0, p_urban_wui=0.16)

    setup_df = pd.read_csv("reports/phase_1a_fires_setup.csv")
    datasets: dict[str, FireDataset60m] = {}
    for _, row in setup_df.iterrows():
        fname = str(row["fire_name"])
        regime = str(row["regime"])
        t_dir = Path("data/tiles/fires") / fname.lower()
        datasets[fname] = load_fire_dataset_60m(fname, regime, t_dir, device=device)

    wind_names = ["Kincade", "Glass", "Zogg"]
    all_names = list(datasets.keys())
    cov_cache: dict[tuple[str, PerturbationConfig], dict[str, Any]] = {}

    # 1. Tune PerturbationConfig via LOFO on Wind-3 folds (target ~80% coverage)
    lofo_perturb_cfgs: dict[str, PerturbationConfig] = {}
    for held_w in wind_names:
        train_w = [datasets[f] for f in wind_names if f != held_w]
        tuned_cfg = tune_perturbation_config_on_wind_folds(cov_cache, train_w, base_tp, base_ep, target_coverage_pct=80.0)
        lofo_perturb_cfgs[held_w] = tuned_cfg
        print(
            f"  LOFO Hold-Out={held_w:8s} | Tuned dP_base=+-{tuned_cfg.p_base_half_width:.3f}, "
            f"wind_speed=+-{tuned_cfg.wind_speed_frac*100:.0f}%, wind_dir=+-{tuned_cfg.wind_dir_deg:.0f} deg, "
            f"ember=+-{tuned_cfg.ember_frac*100:.0f}%, jitter=+-{tuned_cfg.init_jitter_cells} cells",
            flush=True,
        )

    consensus_perturb_cfg = tune_perturbation_config_on_wind_folds(
        cov_cache, [datasets[f] for f in wind_names], base_tp, base_ep, target_coverage_pct=80.0
    )
    print(
        f"  Consensus Tuned Perturbation: dP_base=+-{consensus_perturb_cfg.p_base_half_width:.3f}, "
        f"jitter=+-{consensus_perturb_cfg.init_jitter_cells} cells",
        flush=True,
    )
    for fname in all_names:
        if fname not in lofo_perturb_cfgs:
            lofo_perturb_cfgs[fname] = consensus_perturb_cfg

    # Seed-only baseline config (0 parameter/forcing perturbations)
    seed_only_cfg = PerturbationConfig(
        num_members=20,
        p_base_half_width=0.0,
        wind_speed_frac=0.0,
        wind_dir_deg=0.0,
        ember_frac=0.0,
        init_jitter_cells=0,
    )

    rows: list[dict[str, Any]] = []
    for fname in all_names:
        ds = datasets[fname]
        is_wind = fname in wind_names
        split_label = "held-out (LOFO)" if is_wind else "held-out (unseen in fit)"
        p_cfg = lofo_perturb_cfgs[fname]

        for ens_mode, cfg_curr in [
            ("Seed-Only Ensemble (20 seeds, 0 perturbations)", seed_only_cfg),
            ("Perturbed Ensemble (20 members: Seed + P_base + Wind +-15%/15deg + Ember + Jitter)", p_cfg),
        ]:
            key = (fname, cfg_curr)
            if key not in cov_cache:
                cov_cache[key] = evaluate_fire_rolling_coverage(ds, base_tp, base_ep, True, cfg_curr, max_origins=4)
            res = cov_cache[key]
            rows.append(
                {
                    "fire_name": fname,
                    "regime": ds.regime,
                    "split": split_label,
                    "ensemble_mode": ens_mode,
                    "num_members": cfg_curr.num_members,
                    "rolling_origins_n": res["rolling_origins_evaluated"],
                    "p_base_half_width": cfg_curr.p_base_half_width,
                    "wind_speed_frac": cfg_curr.wind_speed_frac,
                    "wind_dir_deg": cfg_curr.wind_dir_deg,
                    "ember_frac": cfg_curr.ember_frac,
                    "init_jitter_cells": cfg_curr.init_jitter_cells,
                    "denom_obs_progression_cells": res["denom_obs_progression_cells"],
                    "covered_cells": res["covered_cells"],
                    "coverage_pct": res["coverage_pct"],
                    "coverage_with_denom": res["coverage_with_denom"],
                    "raw_covered_cells": res["raw_covered_cells"],
                    "raw_coverage_pct": res["raw_coverage_pct"],
                    "raw_coverage_with_denom": res["raw_coverage_with_denom"],
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv = Path("reports/phase1b2_partB_ensemble_coverage.csv")
    out_df.to_csv(out_csv, index=False)

    print("\n=== Part B Rolling P10-P90 Arrival Coverage Table (Held-Out Folds) ===")
    print(
        out_df[
            [
                "fire_name",
                "regime",
                "split",
                "ensemble_mode",
                "coverage_with_denom",
                "raw_coverage_with_denom",
            ]
        ].to_string(index=False)
    )

    # Plot Part B Coverage Comparison
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=150)
    x = np.arange(len(all_names))
    width = 0.36

    seed_rows = out_df[out_df["ensemble_mode"].str.startswith("Seed-Only")].set_index("fire_name").loc[all_names]
    pert_rows = out_df[out_df["ensemble_mode"].str.startswith("Perturbed")].set_index("fire_name").loc[all_names]

    b1 = ax.bar(x - width / 2, seed_rows["coverage_pct"], width, label="Seed-Only Ensemble (20 seeds, 0 perturbations)", color="#1f77b4")
    b2 = ax.bar(x + width / 2, pert_rows["coverage_pct"], width, label="Perturbed Ensemble (Seeds + P_base + Wind ±15%/15° + Ember + Jitter)", color="#2ca02c")

    for b, (_, r) in zip(b1, seed_rows.iterrows(), strict=True):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{r['coverage_pct']:.1f}%\n({r['covered_cells']}/{r['denom_obs_progression_cells']})", ha="center", va="bottom", fontsize=7.0)
    for b, (_, r) in zip(b2, pert_rows.iterrows(), strict=True):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{r['coverage_pct']:.1f}%\n({r['covered_cells']}/{r['denom_obs_progression_cells']})", ha="center", va="bottom", fontsize=7.0, fontweight="bold")

    ax.axhline(80.0, color="#d62728", linestyle="--", linewidth=1.4, label="Target ~80% P10–P90 Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n({datasets[f].regime})" for f in all_names], fontsize=9.0)
    ax.set_ylabel("Rolling 6h P10–P90 Arrival Coverage (%)", fontsize=10.0)
    ax.set_ylim(0, 108)
    ax.set_title("Phase 1B-2 Part B: Held-Out Rolling P10–P90 Arrival Coverage (with Cell Denominators)", fontsize=11.0, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")

    plt.tight_layout()
    out_png = Path("reports/phase_1b2_partB_ensemble_coverage.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
