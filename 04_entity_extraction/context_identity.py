"""Contextual identity adjudication. Missing detail is not contradictory evidence."""
from typing import Literal
import json
from pydantic import BaseModel, Field
from low_freq_event.backend import ModelOutputError
from low_freq_event.storage import atomic_json

POLICY = '''Identity policy v2:
Use observed action continuity, interactions, shot summaries and stable visible traits together.
A change of clothes, lighting, hair appearance under lighting, pose, scale, location or a time gap is NOT by itself a different individual.
A silhouette, hidden face, blur, back view or missing detail is UNKNOWN evidence, NEVER a contradiction.
Distinguish observed continuity from an invented story. Do not use movie knowledge, role, species or a shared label alone to merge.
Previous summaries, labels and matching decisions are fallible hypotheses; the actual boxed subjects take priority.
Keep carrier/passenger and distinct simultaneously visible subjects separate. Do not confuse a background subject with the boxed target.
'''


class Contradiction(BaseModel):
    frame_a: str
    frame_b: str
    trait: Literal['distinct_subjects', 'incompatible_anatomy', 'incompatible_face']
    evidence_a: str = Field(min_length=1)
    evidence_b: str = Field(min_length=1)


class ContextDecision(BaseModel):
    decision: Literal['same', 'different', 'uncertain']
    visual_evidence: str
    temporal_evidence: str
    positive_identity_evidence: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)


def normalize_decision(result, frames_a, frames_b):
    """Do not promote absence of evidence to a permanent cannot-link constraint."""
    result = result.model_copy(deep=True)
    result.contradictions = [c for c in result.contradictions
                             if c.frame_a in frames_a and c.frame_b in frames_b]
    if result.decision == 'different' and not result.contradictions:
        result.decision = 'uncertain'
    if result.decision == 'same' and (result.contradictions or not any(s.strip() for s in result.positive_identity_evidence)):
        result.decision = 'uncertain'
    return result


def select_context_frames(frames, entries, *, expanded=False):
    """Sample each target shot and neighboring shots; keep target observations too."""
    shots = list(dict.fromkeys(f['shot_id'] for f in frames))
    targets = {o['shot_id'] for e in entries.values() for o in e['observations']}
    radius = 2 if expanded else 1
    wanted = set()
    for i, shot in enumerate(shots):
        if shot in targets:
            wanted.update(shots[max(0, i-radius):i+radius+1])
    selected = set()
    count = 7 if expanded else 4
    for shot in shots:
        if shot not in wanted:
            continue
        pool = [f for f in frames if f['shot_id'] == shot]
        for i in range(min(count, len(pool))):
            index = round(i*(len(pool)-1)/max(1, min(count, len(pool))-1))
            selected.add(pool[index]['frame_key'])
    for e in entries.values():
        by_shot = {}
        for obs in e['observations']:
            by_shot.setdefault(obs['shot_id'], []).append(obs)
        for pool in by_shot.values():
            pool = sorted(pool, key=lambda o: o['full_time_sec'])
            largest = max(pool, key=lambda o: (o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1]))
            for obs in (pool[0], pool[len(pool)//2], pool[-1], largest):
                selected.add(obs['frame_key'])
    return [f for f in frames if f['frame_key'] in selected]


def adjudicate_pair(a, b, pools, frames, history, output, backend, dest, fps, prior):
    # Local import avoids a module cycle: rendering utilities live with the pipeline.
    from sequential_identity import gallery, scene_clip
    entries = {a: pools[a], b: pools[b]}
    images, metadata = gallery(entries, output, dest, f'{a}__{b}_context')
    observations = {key: {o['frame_key'] for o in e['observations']} for key, e in entries.items()}
    for attempt in range(2):
        selected = select_context_frames(frames, entries, expanded=bool(attempt))
        inventory = dict(version=2, targets=[a,b], entities=metadata, shot_history=history,
                         previous_matching_hypotheses=prior,
                         frames=[dict(frame_key=f['frame_key'], shot_id=f['shot_id'],
                                      time=f['full_time_sec'], targets=[k for k in entries if f['frame_key'] in observations[k]])
                                 for f in selected])
        prompt = POLICY + '''
Adjudicate whether the TWO specified boxed targets are the same physical individual across shots.
Review the chronological full-scene frames, enlarged target galleries and shot history together. The timeline may contain montage cuts or flashbacks; screen order does not guarantee continuous action.
Do not let a low-detail crop veto a supported scene-level identity. Explicitly compare the evidence for same identity and the evidence against it.
Use same when positive visible traits and/or specific observed action/interaction continuity support identity without a clear contradiction. Cite that evidence in positive_identity_evidence. A protagonist assumption alone is insufficient.
Use different ONLY with a concrete observable contradiction in BOTH targets, naming actual supplied target frame keys and incompatible stable traits. One visible face versus one hidden face is not a contradiction. If evidence is insufficient, use uncertain.
Return ONLY JSON: {"decision":"same|different|uncertain", "visual_evidence":"observed agreements and limitations", "temporal_evidence":"specific continuity or its absence", "positive_identity_evidence":["specific grounded evidence"], "contradictions":[{"frame_a":"first target frame key", "frame_b":"second target frame key", "trait":"distinct_subjects|incompatible_anatomy|incompatible_face", "evidence_a":"visible fact", "evidence_b":"incompatible visible fact"}]}.
''' + json.dumps(inventory, ensure_ascii=False)
        tag = f'{a}__{b}_context_{attempt+1}'
        atomic_json(dest/f'{tag}_input.json', inventory)
        selected_keys = {f['frame_key'] for f in selected}
        try:
            result = backend.request(prompt, ContextDecision,
                clip=scene_clip(selected, entries, output, fps), images=images,
                max_tokens=1400, image_pixels=786432)
            result = normalize_decision(result, observations[a] & selected_keys, observations[b] & selected_keys)
        except ModelOutputError as exc:
            result = ContextDecision(decision='uncertain', visual_evidence=str(exc), temporal_evidence='Structured-output failure')
        atomic_json(dest/f'{tag}_result.json', result.model_dump())
        # Conflicting provisional links receive a wider scene review before rejection.
        linked = prior.get('forward_same') or prior.get('reverse_same')
        if result.decision == 'same' or (result.decision == 'different' and not linked):
            break
    return result
