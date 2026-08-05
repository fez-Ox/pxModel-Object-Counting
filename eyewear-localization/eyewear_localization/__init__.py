"""Robust eyewear brand attribution pipeline.

The package keeps class-agnostic localization, OCR, cue generation, and
fusion separate.  Brand labels are produced only by the fusion/decision stage.
"""

from eyewear_localization.config import LocalizationConfig, load_config
from eyewear_localization.gazetteer import Gazetteer
from eyewear_localization.pipeline import LocalizationPipeline

__all__ = ["Gazetteer", "LocalizationConfig", "LocalizationPipeline", "load_config"]
