from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Observation(StrictModel):
    local_id: str = Field(min_length=1)
    kind: Literal["person", "character", "animal", "object", "other"]
    label: str = Field(min_length=1)
    appearance: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def check_box(self):
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("bbox must be [x1,y1,x2,y2], normalized to 0..1000, with positive area")
        return self


EventType = Literal["appear", "disappear", "action", "interaction", "state_change", "ongoing"]


class Candidate(StrictModel):
    event_type: EventType
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    actor_id: str | None = None
    object_id: str | None = None
    action: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)
    description: str = Field(min_length=1)
    state_before: str = ""
    state_after: str = ""
    evidence_frames: list[int] = Field(min_length=1)
    uncertain: bool = False
    continues_before: bool = False
    continues_after: bool = False

    @model_validator(mode="after")
    def chronological(self):
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be >= start_sec")
        return self


class ScanResult(StrictModel):
    entities: list[Observation]
    events: list[Candidate]
    scene_cuts_sec: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def references(self):
        ids = [entity.local_id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity local_id must be unique")
        for event in self.events:
            for ref in (event.actor_id, event.object_id):
                if ref is not None and ref not in ids:
                    raise ValueError(f"Unknown entity reference: {ref}")
        return self

    def validate_clip(self, duration: float, frame_count: int) -> "ScanResult":
        for entity in self.entities:
            if entity.frame_index >= frame_count:
                raise ValueError("Entity frame_index is outside this clip")
        for event in self.events:
            if event.end_sec > duration + 0.01:
                raise ValueError("Event timestamps must be relative to this clip, within its duration")
            if any(index < 0 or index >= frame_count for index in event.evidence_frames):
                raise ValueError("Event evidence_frames index is outside this clip")
        if any(time < 0 or time > duration for time in self.scene_cuts_sec):
            raise ValueError("Scene cut outside clip")
        return self


class IdentityDecision(StrictModel):
    match_id: str | None
    possible_ids: list[str] = Field(default_factory=list)
    reason: str


class GroundedBox(StrictModel):
    local_id: str
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def check_box(self):
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("bbox must be normalized to 0..1000 with positive area")
        return self


class GroundingResult(StrictModel):
    boxes: list[GroundedBox]


class Evidence(StrictModel):
    time_sec: float
    image: str


class Event(StrictModel):
    event_id: str = ""
    event_type: EventType
    start_sec: float
    end_sec: float
    actor_id: str | None
    object_id: str | None
    action: str
    description: str
    state_before: str
    state_after: str
    uncertain: bool
    continues_before: bool
    continues_after: bool
    evidence: list[Evidence]
    source_windows: list[str]
    stage: Literal["coarse", "refined"]
    review_reason: str = ""
    unresolved_local_ids: list[str] = Field(default_factory=list)
