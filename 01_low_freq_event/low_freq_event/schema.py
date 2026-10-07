from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LocalEntity(StrictModel):
    local_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: Literal["person", "character", "animal", "object", "environment", "other"]
    appearance: str = Field(min_length=1)


class EntityList(StrictModel):
    entities: list[LocalEntity]

    @model_validator(mode="after")
    def unique(self):
        ids = [e.local_id for e in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("Each entity must have one unique local_id")
        return self


class Candidate(StrictModel):
    event_type: Literal["appear", "disappear", "action", "interaction", "state_change", "ongoing"]
    action: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    entity_ids: list[str] = Field(min_length=1)
    start_frame: int = Field(ge=0, strict=True)
    end_frame: int = Field(ge=0, strict=True)
    continues_before: bool
    continues_after: bool
    uncertain: bool
    boundary_note: str
    temporal_basis: Literal["legacy_interval", "sampled_state_chain", "sampled_transition_bracket"] = "legacy_interval"
    temporal_evidence: list[dict] = Field(default_factory=list)


class ActivityDescription(StrictModel):
    description: str = Field(min_length=1)
    uncertain: bool


class EntityActivity(StrictModel):
    entity_id: str
    description: str = Field(min_length=1)
    uncertain: bool


class WindowNarrative(StrictModel):
    background: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    activities: list[EntityActivity]


class ScanResult(StrictModel):
    narrative: WindowNarrative | None = None
    entities: list[LocalEntity]
    events: list[Candidate]
    action_candidates: list[str] = Field(default_factory=list)
    frame_facts: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def validate_clip(self, frame_count: int):
        ids = [e.local_id for e in self.entities]
        if len(set(ids)) != len(ids):
            raise ValueError("Each physical entity must have ONE unique local_id for the whole clip")
        for event in self.events:
            if not 0 <= event.start_frame <= event.end_frame < frame_count:
                raise ValueError("Use an inclusive valid frame interval: 0 <= start_frame <= end_frame < frame_count")
            if len(set(event.entity_ids)) != len(event.entity_ids) or not set(event.entity_ids) <= set(ids):
                raise ValueError("Event entity_ids must be unique references to entities")
            if event.continues_before and event.start_frame != 0:
                raise ValueError("continues_before requires start_frame=0")
            if event.continues_after and event.end_frame != frame_count - 1:
                raise ValueError("continues_after requires end_frame=last frame")
        return self


class FrameFact(StrictModel):
    entity_ids: list[str] = Field(min_length=1)
    action: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(default="", max_length=100)
    uncertain: bool

    @model_validator(mode='before')
    @classmethod
    def singular_id(cls, value):
        # Unambiguous JSON spelling variant; does not change identity decisions.
        if isinstance(value, dict) and isinstance(value.get('entity_id'), str) and 'entity_ids' not in value:
            value = dict(value)
            value['entity_ids'] = [value.pop('entity_id')]
        return value


class FrameFacts(StrictModel):
    frame_index: int = Field(ge=0, strict=True)
    states: list[FrameFact]
    transitions_from_previous: list[FrameFact]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def compact_id_map(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        for key in ('states','transitions_from_previous'):
            if isinstance(value.get(key), dict):
                expanded=[]
                for eid, actions in value[key].items():
                    actions = [actions] if isinstance(actions, str) else actions
                    if not isinstance(actions, list) or not all(isinstance(a,str) for a in actions):
                        raise ValueError('Compact observations must map an ID to action strings')
                    expanded.extend({'entity_ids':[eid],'action':action,'uncertain':action.startswith('unknown')} for action in actions)
                value[key]=expanded
        return value


class TemporalEvidence(StrictModel):
    frames: list[FrameFacts]


class ActionVocabulary(StrictModel):
    actions: list[str] = Field(max_length=30)


class Box(StrictModel):
    # Extra descriptive metadata is harmless; retain only ID and geometry.
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    local_id: str
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def bounds(self):
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("bbox must have positive area and normalized coordinates in 0..1000")
        return self


class GroundingResult(StrictModel):
    boxes: list[Box]
    new_entities: list[LocalEntity] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Match(StrictModel):
    query_id: str
    match_id: str | None
    possible_ids: list[str]
    reason: str


class Matches(StrictModel):
    matches: list[Match]
