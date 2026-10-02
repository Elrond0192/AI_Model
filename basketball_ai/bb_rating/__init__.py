"""BB-Rating — contextual player performance rating.

The BB-Rating layer is intentionally separate from the season-ahead prediction
model. It evaluates observed player performance relative to a contextual peer
population and returns a 1-100 score plus metric-level evidence.
"""

from basketball_ai.bb_rating.calibration import (
    BBRatingCalibrationConfig,
    build_calibration_report,
    write_calibration_report,
)
from basketball_ai.bb_rating.uncertainty import BBRatingUncertainty
from basketball_ai.bb_rating.engine import (
    BB_RATING_VERSION,
    BBRatingEngine,
    BBRatingResult,
    MetricEvidence,
)

__all__ = [
    "BB_RATING_VERSION",
    "BBRatingEngine",
    "BBRatingResult",
    "MetricEvidence",
    "BBRatingCalibrationConfig",
    "build_calibration_report",
    "write_calibration_report",
    "BBRatingUncertainty",
]
