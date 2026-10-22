"""Split-conformal quantile recalibration for wildfire arrival surrogate (Phase 2B).

Calibrates predicted [q10, q90] quantiles using residuals on uncensored cells from a held-out fire
to guarantee finite-sample marginal coverage >= target (default 80%).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalCalibrationResult:
    """Results from split-conformal quantile calibration."""

    target_coverage: float
    coverage_before: float
    coverage_after: float
    delta_adjustment: float
    num_calibration_points: int
    asymmetric_low: float
    asymmetric_high: float
    median_bias_correction: float


class SplitConformalCalibrator:
    """Split-conformal quantile calibrator for surrogate prediction intervals [q10, q90].

    Based on conformal quantile regression (Romano et al., 2019).
    Computes conformity scores on held-out calibration samples and inflates/deflates
    prediction bounds to guarantee marginal coverage >= target_coverage.
    """

    def __init__(self, target_coverage: float = 0.80, asymmetric: bool = True) -> None:
        if not 0.0 < target_coverage < 1.0:
            raise ValueError(f"Target coverage must be in (0, 1), got {target_coverage}")
        self.target_coverage = target_coverage
        self.asymmetric = asymmetric
        self.alpha = 1.0 - target_coverage
        self.is_fitted = False

        self.delta_symmetric = 0.0
        self.delta_lo = 0.0
        self.delta_hi = 0.0
        self.median_bias = 0.0
        self.num_calibration_points = 0

    def fit(
        self,
        pred_quantiles: np.ndarray,
        true_arrival: np.ndarray,
        censor_mask: np.ndarray | None = None,
    ) -> ConformalCalibrationResult:
        """Fit conformal score adjustments using calibration ground truth.

        Args:
            pred_quantiles: (N, 3, H, W) or (3, ...) array containing [q10, q50, q90].
            true_arrival: (N, H, W) or (...) array of true arrival times.
            censor_mask: Optional binary mask (1 = censored/unburned, 0 = uncensored).
        """
        # Unpack predictions
        if pred_quantiles.ndim == 4:
            q10 = pred_quantiles[:, 0, :, :].ravel()
            q50 = pred_quantiles[:, 1, :, :].ravel()
            q90 = pred_quantiles[:, 2, :, :].ravel()
        elif pred_quantiles.ndim == 3 and pred_quantiles.shape[0] == 3:
            q10 = pred_quantiles[0].ravel()
            q50 = pred_quantiles[1].ravel()
            q90 = pred_quantiles[2].ravel()
        else:
            raise ValueError(f"Unexpected pred_quantiles shape: {pred_quantiles.shape}")

        y = np.asarray(true_arrival).ravel()

        # Filter valid uncensored observations
        if censor_mask is not None:
            c_mask = np.asarray(censor_mask).ravel()
            valid = (c_mask < 0.5) & np.isfinite(y) & np.isfinite(q10) & np.isfinite(q90)
        else:
            valid = np.isfinite(y) & np.isfinite(q10) & np.isfinite(q90)

        q10_valid = q10[valid]
        q50_valid = q50[valid]
        q90_valid = q90[valid]
        y_valid = y[valid]

        n = len(y_valid)
        if n < 10:
            raise ValueError(
                f"Insufficient calibration points for conformal recalibration (got {n})"
            )

        self.num_calibration_points = n

        # Initial coverage before recalibration
        in_interval_before = (y_valid >= q10_valid) & (y_valid <= q90_valid)
        cov_before = float(np.mean(in_interval_before))

        # 1. Symmetric conformal scores: E_i = max(q10 - y, y - q90)
        e_scores = np.maximum(q10_valid - y_valid, y_valid - q90_valid)
        conformal_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.delta_symmetric = float(np.quantile(e_scores, conformal_level, method="higher"))

        # 2. Asymmetric adjustments: separate lower and upper quantile corrections
        # Coverage target per side: 1 - alpha/2
        half_alpha = self.alpha / 2.0
        level_half = min(1.0, np.ceil((n + 1) * (1.0 - half_alpha)) / n)

        err_lo = q10_valid - y_valid
        err_hi = y_valid - q90_valid

        self.delta_lo = float(np.quantile(err_lo, level_half, method="higher"))
        self.delta_hi = float(np.quantile(err_hi, level_half, method="higher"))

        # Median bias
        self.median_bias = float(np.median(y_valid - q50_valid))

        self.is_fitted = True

        # Check coverage on calibration set after adjustment
        if self.asymmetric:
            q10_adj = q10_valid - self.delta_lo
            q90_adj = q90_valid + self.delta_hi
        else:
            q10_adj = q10_valid - self.delta_symmetric
            q90_adj = q90_valid + self.delta_symmetric

        cov_after = float(np.mean((y_valid >= q10_adj) & (y_valid <= q90_adj)))

        return ConformalCalibrationResult(
            target_coverage=self.target_coverage,
            coverage_before=cov_before,
            coverage_after=cov_after,
            delta_adjustment=self.delta_symmetric,
            num_calibration_points=n,
            asymmetric_low=self.delta_lo,
            asymmetric_high=self.delta_hi,
            median_bias_correction=self.median_bias,
        )

    def calibrate(
        self,
        pred_quantiles: np.ndarray,
        horizon_min: float = 720.0,
    ) -> np.ndarray:
        """Apply fitted conformal adjustments to new quantile predictions.

        Args:
            pred_quantiles: Array of shape (N, 3, H, W) or (3, H, W).
            horizon_min: Maximum time horizon in minutes.

        Returns:
            Recalibrated quantiles of same shape with monotonically ordered bounds.
        """
        if not self.is_fitted:
            raise RuntimeError("SplitConformalCalibrator must be fit before calling calibrate.")

        calibrated = pred_quantiles.copy()

        if pred_quantiles.ndim == 4:
            q10 = calibrated[:, 0]
            q50 = calibrated[:, 1]
            q90 = calibrated[:, 2]

            if self.asymmetric:
                q10_new = np.clip(q10 - self.delta_lo, 0.0, horizon_min)
                q90_new = np.clip(q90 + self.delta_hi, 0.0, horizon_min)
            else:
                q10_new = np.clip(q10 - self.delta_symmetric, 0.0, horizon_min)
                q90_new = np.clip(q90 + self.delta_symmetric, 0.0, horizon_min)

            q50_new = np.clip(q50 + self.median_bias, q10_new, q90_new)

            calibrated[:, 0] = q10_new
            calibrated[:, 1] = q50_new
            calibrated[:, 2] = np.maximum(q90_new, q50_new)

        elif pred_quantiles.ndim == 3 and pred_quantiles.shape[0] == 3:
            q10 = calibrated[0]
            q50 = calibrated[1]
            q90 = calibrated[2]

            if self.asymmetric:
                q10_new = np.clip(q10 - self.delta_lo, 0.0, horizon_min)
                q90_new = np.clip(q90 + self.delta_hi, 0.0, horizon_min)
            else:
                q10_new = np.clip(q10 - self.delta_symmetric, 0.0, horizon_min)
                q90_new = np.clip(q90 + self.delta_symmetric, 0.0, horizon_min)

            q50_new = np.clip(q50 + self.median_bias, q10_new, q90_new)

            calibrated[0] = q10_new
            calibrated[1] = q50_new
            calibrated[2] = np.maximum(q90_new, q50_new)

        return calibrated

    @staticmethod
    def evaluate_coverage(
        pred_quantiles: np.ndarray,
        true_arrival: np.ndarray,
        censor_mask: np.ndarray | None = None,
    ) -> float:
        """Compute empirical coverage fraction of [q10, q90] on uncensored cells."""
        if pred_quantiles.ndim == 4:
            q10 = pred_quantiles[:, 0].ravel()
            q90 = pred_quantiles[:, 2].ravel()
        elif pred_quantiles.ndim == 3 and pred_quantiles.shape[0] == 3:
            q10 = pred_quantiles[0].ravel()
            q90 = pred_quantiles[2].ravel()
        else:
            raise ValueError(f"Invalid shape: {pred_quantiles.shape}")

        y = np.asarray(true_arrival).ravel()

        if censor_mask is not None:
            c = np.asarray(censor_mask).ravel()
            valid = (c < 0.5) & np.isfinite(y) & np.isfinite(q10) & np.isfinite(q90)
        else:
            valid = np.isfinite(y) & np.isfinite(q10) & np.isfinite(q90)

        if not np.any(valid):
            return 1.0

        covered = (y[valid] >= q10[valid]) & (y[valid] <= q90[valid])
        return float(np.mean(covered))


@dataclass(frozen=True)
class OneSidedCalibrationResult:
    """Results from one-sided lower quantile conformal calibration for q10."""

    target_coverage: float
    coverage_before: float
    coverage_after: float
    delta_adjustment: float
    num_calibration_points: int


class OneSidedLowerConformalCalibrator:
    """One-sided lower-quantile conformal calibrator for q10 (earliest arrival time).

    Calibrates predicted q10 quantiles using one-sided residuals on uncensored cells to guarantee
    finite-sample marginal coverage P(Y >= q10_calibrated) >= target_coverage (default 90%).
    Kept strictly separate from two-sided SplitConformalCalibrator.
    """

    def __init__(self, target_coverage: float = 0.90) -> None:
        if not 0.0 < target_coverage < 1.0:
            raise ValueError(f"Target coverage must be in (0, 1), got {target_coverage}")
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage
        self.is_fitted = False
        self.delta: float = 0.0
        self.num_calibration_points: int = 0

    def fit(
        self,
        pred_q10: np.ndarray,
        true_arrival: np.ndarray,
        censor_mask: np.ndarray | None = None,
    ) -> OneSidedCalibrationResult:
        """Fit one-sided conformal score adjustment for q10.

        Conformity score: E_i = q10(x_i) - y_i.
        Guarantee: P(Y >= q10 - delta) >= 1 - alpha.
        """
        if pred_q10.ndim == 4 and pred_q10.shape[1] == 3:
            q10 = pred_q10[:, 0].ravel()
        elif pred_q10.ndim == 3 and pred_q10.shape[0] == 3:
            q10 = pred_q10[0].ravel()
        else:
            q10 = np.asarray(pred_q10).ravel()

        y = np.asarray(true_arrival).ravel()

        if censor_mask is not None:
            c_mask = np.asarray(censor_mask).ravel()
            valid = (c_mask < 0.5) & np.isfinite(y) & np.isfinite(q10)
        else:
            valid = np.isfinite(y) & np.isfinite(q10)

        q10_valid = q10[valid]
        y_valid = y[valid]

        n = len(y_valid)
        if n < 10:
            raise ValueError(f"Insufficient calibration points for one-sided calibration (got {n})")

        self.num_calibration_points = n

        # Initial non-exceedance coverage before recalibration: P(y >= q10)
        cov_before = float(np.mean(y_valid >= q10_valid))

        # One-sided error scores: E_i = q10_i - y_i
        e_scores = q10_valid - y_valid
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.delta = float(np.quantile(e_scores, level, method="higher"))
        self.is_fitted = True

        q10_adj = q10_valid - self.delta
        cov_after = float(np.mean(y_valid >= q10_adj))

        return OneSidedCalibrationResult(
            target_coverage=self.target_coverage,
            coverage_before=cov_before,
            coverage_after=cov_after,
            delta_adjustment=self.delta,
            num_calibration_points=n,
        )

    def calibrate(
        self,
        pred_q10: np.ndarray,
        horizon_min: float = 720.0,
    ) -> np.ndarray:
        """Apply fitted one-sided calibration adjustment to q10."""
        if not self.is_fitted:
            raise RuntimeError(
                "OneSidedLowerConformalCalibrator must be fit before calling calibrate()"
            )

        calibrated = np.copy(pred_q10)
        if calibrated.ndim == 4 and calibrated.shape[1] == 3:
            q10_adj = np.clip(calibrated[:, 0] - self.delta, 0.0, horizon_min)
            calibrated[:, 0] = q10_adj
        elif calibrated.ndim == 3 and calibrated.shape[0] == 3:
            q10_adj = np.clip(calibrated[0] - self.delta, 0.0, horizon_min)
            calibrated[0] = q10_adj
        else:
            calibrated = np.clip(calibrated - self.delta, 0.0, horizon_min)

        return calibrated

    @staticmethod
    def evaluate_lower_coverage(
        pred_q10: np.ndarray,
        true_arrival: np.ndarray,
        censor_mask: np.ndarray | None = None,
    ) -> float:
        """Compute empirical coverage fraction of lower bound: P(Y >= q10)."""
        if pred_q10.ndim == 4 and pred_q10.shape[1] == 3:
            q10 = pred_q10[:, 0].ravel()
        elif pred_q10.ndim == 3 and pred_q10.shape[0] == 3:
            q10 = pred_q10[0].ravel()
        else:
            q10 = np.asarray(pred_q10).ravel()

        y = np.asarray(true_arrival).ravel()

        if censor_mask is not None:
            c = np.asarray(censor_mask).ravel()
            valid = (c < 0.5) & np.isfinite(y) & np.isfinite(q10)
        else:
            valid = np.isfinite(y) & np.isfinite(q10)

        if not np.any(valid):
            return 1.0

        covered = y[valid] >= q10[valid]
        return float(np.mean(covered))
