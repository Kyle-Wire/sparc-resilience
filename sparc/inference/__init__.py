"""V4 inference stubs for zero-shot and few-shot prediction."""

from sparc.inference.zero_shot import ZeroShotPrediction, zero_shot_predict
from sparc.inference.few_shot import FewShotPrediction, few_shot_predict

__all__ = [
    "ZeroShotPrediction",
    "zero_shot_predict",
    "FewShotPrediction",
    "few_shot_predict",
]
