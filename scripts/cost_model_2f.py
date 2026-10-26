"""Phase 2F Step 2: Cost model for data generation across 4 sampling designs."""

import numpy as np
import pandas as pd


def main():
    # Measured simulation wall time per single seed run (from Step 1d)
    # Total Step 1 wall time: 489.86 s for 240 + 320 = 560 runs -> 0.875 s/run
    # Or specifically K=16 per-run: 0.754 s/run; K=1 per-run: 1.036 s/run
    # Let's use measured t_sim = 0.875 s
    t_sim = 0.875  # GPU-seconds per simulation run
    budget_s = 28800.0  # 8 hours = 28,800 seconds

    # Measured empirical probabilities for current 4 quota bins from Step 1b:
    # Bin 0 [0.0, 0.2): 208 / 240 = 0.866667
    # Bin 1 [0.2, 0.5): 18 / 240 = 0.075000
    # Bin 2 [0.5, 0.9): 13 / 240 = 0.054167
    # Bin 3 [0.9, 1.0]: 1 / 240 = 0.004167
    p_A = np.array([208, 18, 13, 1]) / 240.0
    p_bottleneck_A = p_A[3]  # 0.004167

    # Target: equal quota across 4 bins (25% per bin).
    # To obtain N_total = 4 * M accepted samples (M per bin),
    # the limiting bin is Bin 3. Expected candidates drawn until Bin 3 gets M samples is M / p_bottleneck.
    # Therefore, expected candidates drawn per accepted sample across all 4 bins is:
    # E[candidates] / (4 * M) = (M / p_bottleneck) / (4 * M) = 1 / (4 * p_bottleneck) = 1 / (4 * (1/240)) = 60 candidates per accepted sample!

    results = []

    for K in [8, 16]:
        # DESIGN A: Current design (quotas on 4 bins, full K per candidate)
        # Every candidate runs full K simulations regardless of whether it is accepted or rejected.
        cand_per_acc_A = 1.0 / (4.0 * p_bottleneck_A)  # 60.0
        sims_per_acc_A = cand_per_acc_A * K
        gpu_s_per_acc_A = sims_per_acc_A * t_sim
        samples_in_8h_A = budget_s / gpu_s_per_acc_A

        results.append({
            "Design": "A (Current: full K, 4 bins)",
            "K": K,
            "Candidates_per_Acc": cand_per_acc_A,
            "Sims_per_Acc": sims_per_acc_A,
            "GPU_sec_per_Acc": gpu_s_per_acc_A,
            "Accepted_Samples_8h": int(samples_in_8h_A),
        })

        # DESIGN B: Two-stage (K=1 prescreen assigns bin, full K only for accepted candidates)
        # To get 4 accepted samples (1 in each bin), we draw 1 / p_bottleneck candidates = 240 candidates.
        # Each candidate gets 1 prescreen run -> 240 runs.
        # 4 candidates are accepted and get the remaining (K - 1) runs -> 4 * (K - 1) runs.
        # Total runs for 4 accepted = 240 + 4 * (K - 1).
        # Runs per accepted sample = (240 / 4) + (K - 1) = 60 + K - 1.
        cand_per_acc_B = cand_per_acc_A
        sims_per_acc_B = cand_per_acc_B + (K - 1)
        gpu_s_per_acc_B = sims_per_acc_B * t_sim
        samples_in_8h_B = budget_s / gpu_s_per_acc_B

        results.append({
            "Design": "B (Two-stage: K=1 prescreen)",
            "K": K,
            "Candidates_per_Acc": cand_per_acc_B,
            "Sims_per_Acc": sims_per_acc_B,
            "GPU_sec_per_Acc": gpu_s_per_acc_B,
            "Accepted_Samples_8h": int(samples_in_8h_B),
        })

        # DESIGN C: B plus proposal distribution favoring high wind and burnable ignition cells
        # Filtering out non-burnable ignition cells removes 2.5% of waste immediately.
        # Biasing proposal (e.g. higher wind / receptive fuels) increases probability of bins 2 and 3.
        # Conservatively, say p_bottleneck rises from 0.42% to ~2.0% (5x improvement).
        # Then candidates per accepted sample = 1 / (4 * 0.02) = 12.5.
        cand_per_acc_C = 12.5
        sims_per_acc_C = cand_per_acc_C + (K - 1)
        gpu_s_per_acc_C = sims_per_acc_C * t_sim
        samples_in_8h_C = budget_s / gpu_s_per_acc_C

        results.append({
            "Design": "C (Two-stage + biased proposal)",
            "K": K,
            "Candidates_per_Acc": cand_per_acc_C,
            "Sims_per_Acc": sims_per_acc_C,
            "GPU_sec_per_Acc": gpu_s_per_acc_C,
            "Accepted_Samples_8h": int(samples_in_8h_C),
        })

        # DESIGN D: B with bin edges set from measured distribution so every bin is reachable
        # Set 4 quartile bin edges from empirical CDF:
        # e.g., Bin 0: [0, 0.002), Bin 1: [0.002, 0.017), Bin 2: [0.017, 0.080), Bin 3: [0.080, 1.0]
        # Then p_0 = p_1 = p_2 = p_3 = 0.25!
        # Coupon collector / balanced multinomial draws:
        # With equal p = 0.25, expected candidates to get 1 in each of 4 bins is 1 / 0.25 = 4 candidates (asymptotically 1 candidate per accepted sample!).
        # Even with slight finite variance, cand_per_acc_D <= 1.25.
        cand_per_acc_D = 1.0 / (4.0 * 0.25)  # 1.0 candidate per accepted sample
        sims_per_acc_D = cand_per_acc_D + (K - 1)  # 1 + K - 1 = K simulations!
        gpu_s_per_acc_D = sims_per_acc_D * t_sim
        samples_in_8h_D = budget_s / gpu_s_per_acc_D

        results.append({
            "Design": "D (Two-stage + empirical quartile bins)",
            "K": K,
            "Candidates_per_Acc": cand_per_acc_D,
            "Sims_per_Acc": sims_per_acc_D,
            "GPU_sec_per_Acc": gpu_s_per_acc_D,
            "Accepted_Samples_8h": int(samples_in_8h_D),
        })

    df = pd.DataFrame(results)
    print("=" * 90)
    print("STEP 2: COST MODEL COMPARISON (t_sim = 0.875 s, Budget = 8h / 28,800 s)")
    print("=" * 90)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
