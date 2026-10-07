from pathlib import Path
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = 4


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    model_revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    window_sec: float = Field(default=5.0, gt=0)
    stride_sec: float = Field(default=1.0, gt=0)
    sample_fps: float = Field(default=1.0, gt=0)
    max_pixels: int = Field(default=196608, ge=4096)
    reference_pixels: int = Field(default=49152, ge=4096)
    gallery_page_size: int = Field(default=4, ge=1, le=8)
    gallery_images_per_entity: int = Field(default=2, ge=1, le=3)
    identity_search: Literal["all", "first_confident"] = "all"
    perception: Literal["unified"] = "unified"
    perception_device: Literal["auto", "cpu", "cuda"] = "auto"
    object_confidence: float = Field(default=.70, ge=0, le=1)
    object_model_id: str = "facebook/dinov2-small"
    object_model_revision: str = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    face_similarity: float = Field(default=.55, gt=0, lt=1)
    face_conflict_quality: float = Field(default=80., ge=20)
    face_min_size: float = Field(default=40., ge=16)
    face_min_eye_distance: float = Field(default=12., ge=4)
    face_min_sharpness: float = Field(default=50., ge=0)
    person_relink_similarity: float = Field(default=.86, gt=0, lt=1)
    person_relink_margin: float = Field(default=.035, ge=0, lt=1)
    body_merge_similarity: float = Field(default=.94, gt=0, lt=1)
    body_similarity: float = Field(default=.82, gt=0, lt=1)
    object_similarity: float = Field(default=.90, gt=0, lt=1)
    identity_margin: float = Field(default=.04, ge=0, lt=1)
    detector_size: int = Field(default=1280, ge=320, le=1920)
    person_confidence: float = Field(default=.60, ge=0, le=1)
    tracking_backend: Literal['botsort', 'appearance'] = 'botsort'
    tracking_fps: float = Field(default=10., gt=0, le=60)
    tracking_low_confidence: float = Field(default=.20, gt=0, lt=1)
    track_buffer_sec: float = Field(default=2., gt=0, le=30)
    max_new_tokens: int = Field(default=4096, ge=512)
    json_retries: int = Field(default=2, ge=0, le=4)

    @model_validator(mode="before")
    @classmethod
    def legacy_settings(cls, value):
        # Old saved results/configs remain readable; there is one execution path.
        if isinstance(value, dict) and value.get('tracking_fps', 10.) is None:
            value = dict(value, tracking_fps=10.)
        if isinstance(value, dict) and value.get('perception') in {'qwen', 'person_reid'}:
            value = dict(value, perception='unified')
        return value

    @model_validator(mode="after")
    def windows(self):
        if self.stride_sec > self.window_sec:
            raise ValueError("stride_sec must not exceed window_sec")
        if self.tracking_low_confidence >= min(self.person_confidence, self.object_confidence):
            raise ValueError('tracking_low_confidence must be below detection birth thresholds')
        return self

    @classmethod
    def load(cls, path: Path | None):
        return cls.model_validate(json.loads(path.read_text())) if path else cls()
