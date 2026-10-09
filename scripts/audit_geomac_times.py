"""Audit and forensic timeline reconciliation for all 28 GeoMAC Camp Fire perimeters (Step 0).

Prints:
- Raw timestamp fields (perimeterdatetime, datecurrent)
- Parsed UTC time under naive UTC vs local PST interpretation
- Hours since ignition (2018-11-08 14:30 UTC = 06:30 PST)
- Area in ha and acres
- Sorted by timestamp

Generates:
- reports/geomac_area_vs_time.png (Plot of observed area vs hours since ignition)
- reports/geomac_timeline_audit.csv
"""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def audit_geomac_perimeters() -> pd.DataFrame:
    data_path = Path("data/raw/camp_fire_2018_perimeter.geojson")
    with open(data_path) as f:
        data = json.load(f)

    # Base ignition time: 2018-11-08 06:30 PST = 2018-11-08 14:30 UTC
    ign_utc = datetime(2018, 11, 8, 14, 30, tzinfo=UTC)
    pst = timezone(timedelta(hours=-8))

    records = []
    for feat in data.get("features", []):
        p = feat.get("properties", {})
        if p.get("state") != "CA" or p.get("incidentname") != "CAMP":
            continue

        obj_id = p.get("OBJECTID")
        raw_perim_dt = p.get("perimeterdatetime")
        raw_datecurr = p.get("datecurrent")
        acres = float(p.get("gisacres", 0.0))
        ha = acres * 0.40468564224
        map_method = str(p.get("mapmethod", ""))
        comments = str(p.get("comments", "")).strip()

        # Interpretation A: Naive UTC (assuming raw_perim_dt is UTC epoch)
        dt_naive_utc = (
            datetime.fromtimestamp(raw_perim_dt / 1000.0, tz=UTC)
            if raw_perim_dt
            else None
        )
        hours_naive_utc = (
            (dt_naive_utc - ign_utc).total_seconds() / 3600.0
            if dt_naive_utc
            else None
        )

        # Interpretation B: Corrected Local-PST Origin
        # NIST TN 2135 p. 127 explicitly documents that the first IR mapping flight occurred
        # at 17:54 on November 8 (local PST time, 11h 24m after ignition).
        # In ESRI/GeoMAC shapefile imports, field local timestamps (17:54) were stored as UTC epochs.
        # Adding +8 hours gives true UTC: 2018-11-09 01:54 UTC = 11.40 hours since ignition.
        dt_corrected_utc = (
            dt_naive_utc + timedelta(hours=8)
            if dt_naive_utc
            else None
        )
        hours_corrected = (
            (dt_corrected_utc - ign_utc).total_seconds() / 3600.0
            if dt_corrected_utc
            else None
        )

        records.append({
            "objectid": obj_id,
            "raw_perimeterdatetime": raw_perim_dt,
            "raw_datecurrent": raw_datecurr,
            "naive_utc_str": dt_naive_utc.strftime("%Y-%m-%d %H:%M:%S UTC") if dt_naive_utc else "N/A",
            "naive_hours_since_ign": hours_naive_utc,
            "corrected_utc_str": dt_corrected_utc.strftime("%Y-%m-%d %H:%M:%S UTC") if dt_corrected_utc else "N/A",
            "corrected_pst_str": dt_corrected_utc.astimezone(pst).strftime("%Y-%m-%d %H:%M:%S PST") if dt_corrected_utc else "N/A",
            "corrected_hours_since_ign": hours_corrected,
            "gis_acres": acres,
            "area_ha": ha,
            "mapmethod": map_method,
            "comments": comments,
        })

    df = pd.DataFrame(records)
    # Sort strictly by timestamp
    df = df.sort_values(by="raw_perimeterdatetime").reset_index(drop=True)

    print("=" * 105)
    print("GEOMAC 2018 CAMP FIRE PERIMETERS TIMELINE AUDIT (28 RECORDS)")
    print("=" * 105)
    header = (
        f"{'#':2s} {'ID':4s} | {'Raw perim_dt':13s} | {'Naive UTC':20s} | "
        f"{'Naive +h':8s} | {'Corrected UTC':20s} | {'Corr +h':7s} | "
        f"{'Acres':9s} | {'Area (ha)':9s} | {'Method':14s}"
    )
    print(header)
    print("-" * 105)

    for idx, r in df.iterrows():
        print(
            f"{idx+1:02d} {r['objectid']:4d} | {str(r['raw_perimeterdatetime']):13s} | "
            f"{r['naive_utc_str']:20s} | {r['naive_hours_since_ign']:8.2f} | "
            f"{r['corrected_utc_str']:20s} | {r['corrected_hours_since_ign']:7.2f} | "
            f"{r['gis_acres']:9.1f} | {r['area_ha']:9.1f} | {r['mapmethod']:14s}"
        )

    out_csv = Path("reports/geomac_timeline_audit.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved audit records to {out_csv}")

    # Plot area vs hours since ignition
    plot_path = Path("reports/geomac_area_vs_time.png")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot both naive interpretation and corrected interpretation
    ax.plot(
        df["naive_hours_since_ign"],
        df["area_ha"],
        "o--",
        color="tab:red",
        alpha=0.6,
        label="Naive UTC Interpretation (Faulty 17:54 UTC = +3.4h)",
    )
    ax.plot(
        df["corrected_hours_since_ign"],
        df["area_ha"],
        "s-",
        color="tab:blue",
        linewidth=2,
        label="Corrected PST Interpretation (17:54 PST Overflight = +11.4h)",
    )

    # Highlight the first perimeter
    first_row = df.iloc[0]
    ax.annotate(
        f"Perimeter #1 (ID {first_row['objectid']})\n54,586 ac (22,090 ha)\nNIST 17:54 PST IR Overflight\n(+11.4h post-ignition)",
        xy=(first_row["corrected_hours_since_ign"], first_row["area_ha"]),
        xytext=(first_row["corrected_hours_since_ign"] + 15, first_row["area_ha"] - 4000),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5},
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "yellow", "alpha": 0.5},
        fontsize=9,
    )

    ax.axhline(62052.7, color="black", linestyle=":", alpha=0.7, label="Final Containment (62,053 ha)")
    ax.set_xlabel("Hours Since Ignition (2018-11-08 06:30 PST / 14:30 UTC)", fontsize=11)
    ax.set_ylabel("Observed Fire Area (ha)", fontsize=11)
    ax.set_title("2018 Camp Fire: GeoMAC Fire Area vs Hours Since Ignition Timeline Audit", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved timeline plot to {plot_path}")

    return df


if __name__ == "__main__":
    audit_geomac_perimeters()
