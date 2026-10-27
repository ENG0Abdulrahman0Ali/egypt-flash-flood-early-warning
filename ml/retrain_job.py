"""Validate and promote trained ML candidate models."""

from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import joblib

from feature_columns import FEATURE_COLUMNS, FORECAST_HORIZON_HOURS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("retrain_job")

MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_CONFIG = {
    "rainfall": {
        "candidate": "rainfall_candidate.joblib",
        "production": "rainfall_production.joblib",
    },
    "risk_score": {
        "candidate": "risk_score_candidate.joblib",
        "production": "risk_score_production.joblib",
    },
    "risk_level": {
        "candidate": "risk_level_candidate.joblib",
        "production": "risk_level_production.joblib",
    },
}

# Minimum acceptable macro-F1 for the multiclass risk classifier.
MIN_RISK_LEVEL_MACRO_F1 = 0.60


def _finite_metric(metrics: dict[str, Any], name: str) -> float | None:
    """Return a finite numeric metric, or None when it is invalid."""
    value = metrics.get(name)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    value = float(value)
    return value if math.isfinite(value) else None


def _load_valid_artifact(path: Path) -> dict[str, Any] | None:
    """Load a model artifact and validate its required metadata."""
    if not path.is_file():
        logger.warning("Artifact not found: %s", path.name)
        return None

    try:
        artifact = joblib.load(path)
    except Exception:
        logger.exception("Could not load artifact: %s", path.name)
        return None

    if not isinstance(artifact, dict):
        logger.error("Invalid artifact format: %s", path.name)
        return None

    if artifact.get("model") is None:
        logger.error("Artifact has no model: %s", path.name)
        return None

    actual_features = artifact.get("feature_columns")
    expected_features = list(FEATURE_COLUMNS)

    if actual_features != expected_features:
        logger.error("Feature schema mismatch: %s", path.name)
        return None

    if artifact.get("forecast_horizon_hours") != FORECAST_HORIZON_HOURS:
        logger.error("Forecast horizon mismatch: %s", path.name)
        return None

    if not isinstance(artifact.get("metrics"), dict):
        logger.error("Missing metrics: %s", path.name)
        return None

    return artifact


def _passes_quality_gate(
    name: str,
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Apply model-specific quality checks before promotion."""
    metrics = candidate["metrics"]

    if name == "rainfall":
        mae = _finite_metric(metrics, "mae")
        baseline_mae = _finite_metric(
            metrics, "persistence_baseline_mae"
        )

        if mae is None or baseline_mae is None:
            return False, "Missing or invalid rainfall evaluation metrics."

        if mae >= baseline_mae:
            return (
                False,
                f"Candidate MAE ({mae:.6f}) does not beat "
                f"persistence baseline ({baseline_mae:.6f}).",
            )

        if incumbent is not None:
            old_mae = _finite_metric(incumbent["metrics"], "mae")

            if old_mae is None or mae >= old_mae:
                return False, "Candidate does not improve incumbent rainfall MAE."

    elif name == "risk_score":
        mae = _finite_metric(metrics, "mae")
        rmse = _finite_metric(metrics, "rmse")

        if mae is None or rmse is None:
            return False, "Missing or invalid risk-score metrics."

        if mae < 0 or rmse < 0:
            return False, "Risk-score errors cannot be negative."

        if incumbent is not None:
            old_mae = _finite_metric(incumbent["metrics"], "mae")

            if old_mae is None or mae >= old_mae:
                return False, "Candidate does not improve incumbent risk-score MAE."

    elif name == "risk_level":
        macro_f1 = _finite_metric(metrics, "macro_f1")

        if macro_f1 is None or not 0 <= macro_f1 <= 1:
            return False, "Missing or invalid risk-level macro-F1."

        if macro_f1 < MIN_RISK_LEVEL_MACRO_F1:
            return (
                False,
                f"Macro-F1 {macro_f1:.4f} is below "
                f"the minimum {MIN_RISK_LEVEL_MACRO_F1:.2f}.",
            )

        if incumbent is not None:
            old_f1 = _finite_metric(incumbent["metrics"], "macro_f1")

            if old_f1 is None or macro_f1 < old_f1:
                return False, "Candidate does not maintain incumbent macro-F1."

    return True, "Quality checks passed."


def _promote(candidate_path: Path, production_path: Path) -> None:
    """Atomically replace the production artifact with the validated candidate."""
    temporary_path = production_path.with_name(
        production_path.name + ".tmp"
    )

    try:
        shutil.copy2(candidate_path, temporary_path)
        os.replace(temporary_path, production_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    promoted = []
    rejected = []

    for name, config in MODEL_CONFIG.items():
        candidate_path = MODEL_DIR / config["candidate"]
        production_path = MODEL_DIR / config["production"]

        candidate = _load_valid_artifact(candidate_path)

        if candidate is None:
            rejected.append(name)
            continue

        incumbent = None

        if production_path.exists():
            incumbent = _load_valid_artifact(production_path)

            if incumbent is None:
                logger.error(
                    "Existing production artifact is invalid: %s. "
                    "Refusing to overwrite it.",
                    production_path.name,
                )
                rejected.append(name)
                continue

        passed, reason = _passes_quality_gate(
            name,
            candidate,
            incumbent,
        )

        if not passed:
            logger.warning(
                "REJECTED %s candidate: %s",
                name,
                reason,
            )
            rejected.append(name)
            continue

        try:
            _promote(candidate_path, production_path)
        except Exception:
            logger.exception("Promotion failed for %s.", name)
            rejected.append(name)
            continue

        logger.info(
            "PROMOTED %s model: version=%s, path=%s",
            name,
            candidate.get("model_version", "unknown"),
            production_path,
        )
        promoted.append(name)

    logger.info("Promotion summary: promoted=%s rejected=%s", promoted, rejected)

    production_files = [
        MODEL_DIR / config["production"]
        for config in MODEL_CONFIG.values()
    ]

    if not any(path.is_file() for path in production_files):
        raise RuntimeError(
            "No production models are available. "
            "Inference must not run until at least one model is promoted."
        )

    logger.info("Model promotion job completed.")


if __name__ == "__main__":
    main()