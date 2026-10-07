from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    model_revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    window_sec: float = Field(default=5.0, gt=0)
    stride_sec: float = Field(default=1.0, gt=0)
    coarse_fps: float = Field(default=1.5, gt=0)
    refine_fps: float = Field(default=4.0, gt=0)
    refine_padding_sec: float = Field(default=1.0, ge=0)
    max_pixels: int = Field(default=262144, ge=4096)
    reference_pixels: int = Field(default=65536, ge=4096)
    gallery_page_size: int = Field(default=4, ge=1, le=12)
    gallery_images_per_entity: int = Field(default=2, ge=1, le=4)
    max_new_tokens: int = Field(default=2048, ge=128)
    json_retries: int = Field(default=1, ge=0, le=3)
    merge_tolerance_sec: float = Field(default=0.35, ge=0)

    @model_validator(mode="after")
    def validate_windows(self):
        if self.stride_sec > self.window_sec:
            raise ValueError("stride_sec must not exceed window_sec (would leave gaps)")
        return self

    @classmethod
    def load(cls, path: Path | None) -> "Settings":
        return cls.model_validate(json.loads(path.read_text())) if path else cls()
