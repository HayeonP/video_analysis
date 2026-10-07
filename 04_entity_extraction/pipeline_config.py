"""Reusable operating settings; no video identities or expected entity counts."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from strong_detector import StrongDetectorConfig
from efficient_tracks import TrackingConfig
from anchored_matching import MatchingConfig
from visual_association import VisualAssociationConfig
from observation_fusion import _thresholds as fusion_thresholds

COCO_KINDS = dict(person='person', **{name:'animal' for name in (
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe')})


@dataclass(frozen=True)
class FilteringConfig:
    drop_weak_singletons: bool = False
    singleton_confidence: float = .65

    def __post_init__(self):
        if type(self.drop_weak_singletons) is not bool:
            raise ValueError('filtering.drop_weak_singletons must be true or false')
        if not 0 <= self.singleton_confidence <= 1:
            raise ValueError('filtering.singleton_confidence must be in 0..1')


@dataclass(frozen=True)
class AppearanceConfig:
    off_match_threshold: float = .93
    off_match_margin: float = .04

    def __post_init__(self):
        for name in ('off_match_threshold', 'off_match_margin'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'appearance.{name} must be in 0..1')


@dataclass(frozen=True)
class PipelineConfig:
    detector: StrongDetectorConfig = field(default_factory=StrongDetectorConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    visual_association: VisualAssociationConfig = field(default_factory=VisualAssociationConfig)
    fusion: dict = field(default_factory=lambda:fusion_thresholds(None))
    class_kinds: dict = field(default_factory=lambda:dict(COCO_KINDS))

    def kind_for(self, category):
        return self.class_kinds.get(str(category).strip().casefold(),'object')

    def to_dict(self):
        # Tuples (e.g. tile categories) must compare equal after a JSON round trip.
        return json.loads(json.dumps(asdict(self),ensure_ascii=False,allow_nan=False))


def load_config(path):
    raw=json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    if not isinstance(raw,dict):raise ValueError('Pipeline config must be a JSON object')
    allowed={'detector','matching','tracking','filtering','appearance','class_kinds','visual_association','fusion'}
    unknown=set(raw)-allowed
    if unknown:raise ValueError(f'Unknown config sections: {sorted(unknown)}')
    values={}
    for key,cls in (('detector',StrongDetectorConfig),('matching',MatchingConfig),
                    ('tracking',TrackingConfig),('filtering',FilteringConfig),('appearance',AppearanceConfig),
                    ('visual_association',VisualAssociationConfig)):
        section=raw.get(key,{})
        if not isinstance(section,dict):raise ValueError(f'{key} must be an object')
        values[key]=cls(**section)
    overrides=raw.get('class_kinds',{})
    if not isinstance(overrides,dict):raise ValueError('class_kinds must be an object')
    kinds=dict(COCO_KINDS)
    for name,kind in overrides.items():
        if not isinstance(name,str) or not name.strip() or kind not in {'person','animal','object'}:
            raise ValueError('class_kinds maps class names to person, animal, or object')
        kinds[name.strip().casefold()]=kind
    return PipelineConfig(**values,class_kinds=kinds,fusion=fusion_thresholds({'observation_fusion':raw.get('fusion',{})}))


def select_input(path, inputs_root):
    if path is not None:
        source=Path(path).expanduser().resolve()
        if not (source/'summary.json').is_file():
            raise ValueError(f'03 adaptive summary.json이 있는 폴더가 필요합니다: {source}')
        return source
    base=Path(inputs_root)
    candidates=([base] if (base/'summary.json').is_file() else [])
    if base.is_dir():
        candidates.extend(p for p in sorted(base.iterdir()) if p.is_dir() and (p/'summary.json').is_file())
    if len(candidates)!=1:
        raise ValueError(f'입력 후보가 {len(candidates)}개입니다. --input <03 adaptive 결과 폴더>를 지정하세요.')
    return candidates[0].resolve()
