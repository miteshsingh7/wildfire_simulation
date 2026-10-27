"""Unit tests for Phase 2I: Sampler stall fix, strict round-robin, and stall cap closure."""

from unittest.mock import patch
import numpy as np
import pytest

from wf052.surrogate.standin_generator import (
    format_fire_bin_status_table,
    generate_stratified_samples_rejection,
)


class TestSamplerStallFix:
    """Verify sampler stall fix: strict round-robin, STALL_CAP closure, and status reporting."""

    def test_stall_cap_closes_bin4_after_exact_sims(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A fire that never produces bin 4 closes bin 4 after exactly STALL_CAP sims while other fires keep being visited."""
        stall_cap = 10
        fire_names = ["caldor", "glass"]

        # Mock simulate_teacher_with_embers_60m:
        # Glass produces small burned fraction (bin 1, bf ~ 0.005)
        # Caldor produces high burned fraction (bin 4, bf ~ 0.70)
        def mock_sim(ds, params, ember_params, use_ember_kernel, duration_hours, steps_per_hour, seed, init_burned_mask):
            f_name = ds.fire_name
            arr = np.full((256, 256), np.inf, dtype=np.float32)
            if f_name == "glass":
                # Burn a small patch (e.g. 300 cells out of 65536 -> bf ~ 0.0045, bin 1)
                arr[120:135, 120:140] = 60.0
            else:
                # Burn large area (e.g. 45000 cells -> bf ~ 0.68, bin 4)
                arr[20:230, 20:230] = 60.0
            return {"arrival_time_min": arr}

        with patch("wf052.surrogate.standin_generator.simulate_teacher_with_embers_60m", side_effect=mock_sim):
            samples, rep = generate_stratified_samples_rejection(
                fire_names=fire_names,
                source_mode="teacher_v2_ensemble",
                target_quotas=[0, 1, 0, 0, 1],  # Only bin 1 and bin 4 have quota 1
                stall_cap=stall_cap,
                teacher_config_path="configs/teacher_v2_a6a92f81.yaml",
            )

        print(f"\n[Test Output 1c(i)] Visits per fire: {rep.fire_visits}")
        print(f"[Test Output 1c(i)] Glass closed_by: {rep.fire_bin_closed_by['glass']}")
        print(f"[Test Output 1c(i)] Caldor closed_by: {rep.fire_bin_closed_by['caldor']}")

        # 1. Glass bin 4 must be closed by "stall"
        assert rep.fire_bin_closed_by["glass"][4] == "stall", (
            f"Expected Glass bin 4 closed by 'stall', got {rep.fire_bin_closed_by['glass'][4]}"
        )
        # Glass bin 1 filled quota
        assert rep.fire_bin_closed_by["glass"][1] == "quota"

        # 2. Caldor filled both bin 1 and bin 4 by quota or stall
        # Caldor produced bin 4, so bin 4 closed by quota
        assert rep.fire_bin_closed_by["caldor"][4] == "quota"

        # 3. Visits per fire: both fires were visited
        assert rep.fire_visits["glass"] >= stall_cap
        assert rep.fire_visits["caldor"] >= 1

    def test_visit_counts_differ_by_at_most_one_while_all_fires_open(self) -> None:
        """Visit counts differ by at most 1 while all fires have open bins (strict round-robin)."""
        fires = ["bobcat", "caldor", "creek", "glass", "kincade", "tamarack", "zogg"]
        # With 7 fires and max_attempts = 17, visit counts must be 3 for first 3 fires, 2 for the remaining 4 (diff <= 1)
        mock_arr = np.full((256, 256), np.inf, dtype=np.float32)
        mock_arr[128, 128] = 0.0

        with patch("wf052.surrogate.standin_generator.simulate_teacher_with_embers_60m") as mock_sim:
            mock_sim.return_value = {"arrival_time_min": mock_arr}
            samples, rep = generate_stratified_samples_rejection(
                fire_names=fires,
                source_mode="teacher_v2_ensemble",
                target_quotas=[10, 10, 10, 10, 10],  # All bins large quota, none close early
                max_attempts=17,
                teacher_config_path="configs/teacher_v2_a6a92f81.yaml",
            )

        visits = rep.fire_visits
        print(f"\n[Test Output 1c(ii)] Visits after 17 attempts: {visits}")
        v_vals = list(visits.values())
        diff = max(v_vals) - min(v_vals)
        assert diff <= 1, f"Visits differ by {diff} > 1: {visits}"

    def test_summary_lists_per_fire_bin_status(self) -> None:
        """The summary report records and formats per (fire, bin) status (quota, stall, open)."""
        fires = ["caldor", "zogg"]
        mock_arr = np.full((256, 256), np.inf, dtype=np.float32)
        mock_arr[128, 128] = 0.0

        with patch("wf052.surrogate.standin_generator.simulate_teacher_with_embers_60m") as mock_sim:
            mock_sim.return_value = {"arrival_time_min": mock_arr}
            samples, rep = generate_stratified_samples_rejection(
                fire_names=fires,
                source_mode="teacher_v2_ensemble",
                target_quotas=[1, 0, 0, 0, 1],
                stall_cap=5,
                teacher_config_path="configs/teacher_v2_a6a92f81.yaml",
            )

        assert rep.fire_bin_closed_by is not None
        for f in fires:
            assert f in rep.fire_bin_closed_by
            assert len(rep.fire_bin_closed_by[f]) == 5
            for status in rep.fire_bin_closed_by[f]:
                assert status in ("quota", "stall", "open")

        status_tbl = format_fire_bin_status_table(
            fire_names=fires,
            bin_names=["bin0", "bin1", "bin2", "bin3", "bin4"],
            quotas_per_fire={"caldor": [1, 0, 0, 0, 1], "zogg": [1, 0, 0, 0, 1]},
            accepted_by_fire_bin=rep.fire_bin_counts,
            closed_by_fire_bin=rep.fire_bin_closed_by,
        )
        print("\n[Test Output 1c(iii)] Status Table:\n" + status_tbl)
        assert "Closed By" in status_tbl
        assert "quota" in status_tbl
