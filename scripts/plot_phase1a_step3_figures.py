"""Generate diagnostic figures for Phase 1A Step 3 (OAT sensitivity, LOFO ablation, Held-Out Camp Fire).

Produces:
1. reports/phase_1a_oat_sensitivity.png
2. reports/phase_1a_lofo_comparison.png
3. reports/phase_1a_camp_fire_heldout_60m.png
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from evaluate_heldout_camp_fire import build_camp_fire_dataset_60m, verify_frozen_config
from rasterio.features import rasterize

from wf052.sim.teacher_calibrator import TeacherParams, simulate_teacher_60m


def plot_oat_sensitivity() -> None:
    df = pd.read_csv("reports/parameter_sensitivity_summary.csv")
    df_kin = df[df["fire_name"] == "Kincade"].reset_index(drop=True)
    df_bob = df[df["fire_name"] == "Bobcat"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    params = df_kin["parameter"].tolist()
    y_pos = np.arange(len(params))

    ax0 = axes[0]
    ax0.barh(y_pos - 0.18, df_kin["rel_area_swing_24h_pct"], height=0.35, label="Kincade (Wind-Driven)", color="tab:red")
    ax0.barh(y_pos + 0.18, df_bob["rel_area_swing_24h_pct"], height=0.35, label="Bobcat (Plume/Terrain)", color="tab:orange")
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(params)
    ax0.invert_yaxis()
    ax0.set_xlabel("24h Burned Area Relative Swing (% of Baseline)")
    ax0.set_title("OAT Sensitivity: 24h Burned Area Swing", fontweight="bold")
    ax0.grid(True, linestyle="--", alpha=0.4)
    ax0.legend()

    ax1 = axes[1]
    ax1.barh(y_pos - 0.18, df_kin["mae_swing_24h_min"], height=0.35, label="Kincade (Wind-Driven)", color="tab:blue")
    ax1.barh(y_pos + 0.18, df_bob["mae_swing_24h_min"], height=0.35, label="Bobcat (Plume/Terrain)", color="tab:cyan")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(params)
    ax1.invert_yaxis()
    ax1.set_xlabel("24h Arrival-Time MAE Swing (minutes)")
    ax1.set_title("OAT Sensitivity: 24h Arrival MAE Swing", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    plt.tight_layout()
    out = Path("reports/phase_1a_oat_sensitivity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_lofo_comparison() -> None:
    df = pd.read_csv("reports/lofo_regime_ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    models = [
        ("uncalibrated", "Uncalibrated (P_base=0.50) [Held-Out]", "tab:gray"),
        ("global_slowdown_lofo", "1-Param Slowdown [LOFO Held-Out]", "tab:green"),
        ("calibrated_all7_lofo", "Calibrated All-7 [LOFO Held-Out]", "tab:blue"),
        ("calibrated_wind_only_lofo", "Calibrated Wind-Only [LOFO Held-Out]", "tab:red"),
        ("calibrated_all7_insample", "Calibrated All-7 [In-Sample]", "tab:purple"),
    ]

    # Subplot 1: All 7 Fires across horizons (1, 3, 6, 12h)
    ax0 = axes[0]
    sub_all = df[df["evaluation_group"] == "All 7 Fires"]
    for m_key, m_label, color in models:
        m_df = sub_all[sub_all["model"] == m_key].sort_values("horizon_h")
        ls = "--" if "In-Sample" in m_label else "-"
        ax0.plot(m_df["horizon_h"], m_df["mean_iou_2km"], marker="o", linestyle=ls, linewidth=2, color=color, label=m_label)
    ax0.set_xlabel("Simulation Horizon (Hours)")
    ax0.set_ylabel("Mean ~2 km GOFER IoU")
    ax0.set_title("All 7 Training Fires: ~2 km GOFER IoU vs Horizon", fontweight="bold")
    ax0.set_xticks([1, 3, 6, 12])
    ax0.grid(True, linestyle="--", alpha=0.4)
    ax0.legend(fontsize=8.5)

    # Subplot 2: +12h IoU by Regime Subset
    ax1 = axes[1]
    subsets = [
        "All 7 Fires",
        "Wind-Driven Folds (Kincade, Glass, Zogg)",
        "Plume-Fire Folds (Creek, Bobcat)",
        "Terrain Folds (Caldor, Tamarack)",
    ]
    short_labels = ["All 7 Fires", "Wind-Driven (3)", "Plume (2)", "Terrain (2)"]
    x = np.arange(len(subsets))
    width = 0.16

    for idx, (m_key, m_label, color) in enumerate(models):
        vals = []
        for s_name in subsets:
            row = df[(df["evaluation_group"] == s_name) & (df["model"] == m_key) & (df["horizon_h"] == 12)]
            vals.append(float(row["mean_iou_2km"].iloc[0]) if len(row) > 0 else 0.0)
        ax1.bar(x + (idx - 2) * width, vals, width=width, color=color, label=m_label)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels)
    ax1.set_ylabel("+12 h Mean ~2 km GOFER IoU")
    ax1.set_title("+12 h IoU Across Fire Regimes (LOFO Held-Out vs In-Sample)", fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8)

    plt.tight_layout()
    out = Path("reports/phase_1a_lofo_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_camp_fire_heldout() -> None:
    cfg_teacher = verify_frozen_config(Path("configs/teacher_v1.yaml"), Path("reports/teacher_v1_hash.txt"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds_camp, _, transform_60m = build_camp_fire_dataset_60m(device)
    h, w = ds_camp.p_veg_base.shape

    geomac_path = Path("data/raw/geomac_camp_fire_2018_all_perimeters.geojson")
    gdf_geomac = gpd.read_file(geomac_path).to_crs("EPSG:32610")
    geom_1052 = gdf_geomac[gdf_geomac["OBJECTID"] == 1052].geometry.iloc[0]
    obs_1052_mask = (
        rasterize([(geom_1052, 1)], out_shape=(h, w), transform=transform_60m, fill=0, dtype=np.uint8) == 1
    )

    # Water / unburnable mask
    unburnable_mask = (~ds_camp.is_burnable).cpu().numpy()

    panels = [
        ("Uncalibrated (p_base=0.50)\nSim: 84,036 ha (3.79x) | Prec: 21.5%", TeacherParams(**cfg_teacher["uncalibrated_default_parameters"])),
        ("Frozen wind_only_v1 (p_base=0.22)\nSim: 2,979 ha (0.13x) | Prec: 76.7%", TeacherParams(**cfg_teacher["ablation_wind_only_parameters"])),
        ("Frozen teacher_v1 (p_base=0.21)\nSim: 1,410 ha (0.06x) | Prec: 86.2%", TeacherParams(**cfg_teacher["parameters"])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=150)
    for ax, (title, params) in zip(axes, panels, strict=True):
        sim_out = simulate_teacher_60m(ds_camp, params, duration_hours=12, steps_per_hour=30, seed=42)
        arr_min = sim_out["arrival_time_min"]
        sim_684 = (~np.isnan(arr_min)) & (arr_min <= 684.0)

        canvas = np.ones((h, w, 3), dtype=np.float32) * 0.94
        canvas[unburnable_mask] = [0.75, 0.82, 0.92]  # Water / non-burnable in light blue-gray
        # Observed +11.4h footprint in light gold
        canvas[obs_1052_mask] = [0.98, 0.88, 0.60]
        # False Positive sim in purple/magenta
        canvas[sim_684 & (~obs_1052_mask)] = [0.85, 0.35, 0.55]
        # True Positive overlap in dark red
        canvas[sim_684 & obs_1052_mask] = [0.75, 0.12, 0.12]

        ax.imshow(canvas)
        ax.contour(obs_1052_mask.astype(float), levels=[0.5], colors="black", linewidths=1.2)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Held-Out 2018 Camp Fire (+11.4 h / 684 min at 60 m): Dark Red = TP, Pink = FP, Gold = FN (Unburned Obs), Blue-Gray = Water/Non-Burnable",
        fontsize=10.5,
        fontweight="bold",
    )
    plt.tight_layout()
    out = Path("reports/phase_1a_camp_fire_heldout_60m.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    import sys

    sys.path.insert(0, "scripts")
    plot_oat_sensitivity()
    plot_lofo_comparison()
    plot_camp_fire_heldout()


if __name__ == "__main__":
    main()
