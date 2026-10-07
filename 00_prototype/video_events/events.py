from __future__ import annotations

from .config import Settings
from .schema import Event, Evidence, ScanResult
from .video import Clip, sliding_windows


def global_events(result: ScanResult, clip: Clip, mapping: dict[str, str], window_id: str, stage: str) -> list[Event]:
    events = []
    for candidate in result.events:
        unresolved = sorted({ref for ref in (candidate.actor_id, candidate.object_id)
                             if ref is not None and ref not in mapping})
        # Include the observed interval and its neighboring frames so the exported
        # evidence cannot consist only of an incorrect model-generated frame index.
        inside = [index for index, frame in enumerate(clip.frames)
                  if candidate.start_sec <= frame.time_sec - clip.start_sec <= candidate.end_sec]
        before = [index for index, frame in enumerate(clip.frames)
                  if frame.time_sec - clip.start_sec < candidate.start_sec]
        after = [index for index, frame in enumerate(clip.frames)
                 if frame.time_sec - clip.start_sec > candidate.end_sec]
        evidence_indices = sorted(set(inside + before[-1:] + after[:1] + candidate.evidence_frames))
        events.append(Event(
            **candidate.model_dump(exclude={"start_sec", "end_sec", "actor_id", "object_id", "evidence_frames", "uncertain"}),
            start_sec=round(clip.start_sec + candidate.start_sec, 6),
            end_sec=round(clip.start_sec + candidate.end_sec, 6),
            actor_id=mapping.get(candidate.actor_id), object_id=mapping.get(candidate.object_id),
            evidence=[Evidence(time_sec=clip.frames[index].time_sec, image=clip.frames[index].path)
                      for index in evidence_indices],
            source_windows=[window_id], stage=stage,
            uncertain=candidate.uncertain or bool(unresolved),
            unresolved_local_ids=unresolved,
            review_reason="unlocalized_entities" if unresolved else
                          ("uncertain" if candidate.uncertain else candidate.event_type),
        ))
    return events


def action_key(action: str) -> str:
    key = action.strip().lower().replace(" ", "_").replace("-", "_")
    return {"pickup": "pick_up", "grab": "pick_up", "putdown": "put_down",
            "give": "hand_over", "appear": "become_visible", "disappear": "leave_view"}.get(key, key)


def same_occurrence(a: Event, b: Event, tolerance: float) -> bool:
    if a.stage != b.stage:
        return False  # A coarse fallback must not be presented as a verified refined event.
    if a.unresolved_local_ids or b.unresolved_local_ids:
        return False  # Two unlocalized actors must not become one just because both IDs are null.
    if (a.actor_id, a.object_id, action_key(a.action)) != (b.actor_id, b.object_id, action_key(b.action)):
        return False
    if set(a.source_windows) & set(b.source_windows):
        return False  # Explicitly separate occurrences in the same model response stay separate.
    duration_a, duration_b = a.end_sec - a.start_sec, b.end_sec - b.start_sec
    intersection = max(0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    if min(duration_a, duration_b) <= 1e-6:
        return abs(a.start_sec - b.start_sec) <= tolerance
    if intersection / min(duration_a, duration_b) >= 0.5:
        return True
    return (a.continues_after and b.continues_before and
            -tolerance <= b.start_sec - a.end_sec <= tolerance)


def merge_events(events: list[Event], tolerance: float) -> list[Event]:
    merged: list[Event] = []
    for event in sorted(events, key=lambda item: (item.start_sec, item.end_sec)):
        match = next((previous for previous in reversed(merged) if same_occurrence(previous, event, tolerance)), None)
        if match is None:
            merged.append(event.model_copy(deep=True))
            continue
        # Prefer uncensored estimates where a neighboring window observes the boundary.
        starts = [value.start_sec for value in (match, event) if not value.continues_before]
        ends = [value.end_sec for value in (match, event) if not value.continues_after]
        match.start_sec = min(starts) if starts else min(match.start_sec, event.start_sec)
        match.end_sec = max(ends) if ends else max(match.end_sec, event.end_sec)
        match.end_sec = max(match.start_sec, match.end_sec)
        match.continues_before = match.continues_before and event.continues_before
        match.continues_after = match.continues_after and event.continues_after
        match.uncertain = match.uncertain and event.uncertain
        match.source_windows = sorted(set(match.source_windows + event.source_windows))
        evidence = {item.image: item for item in match.evidence + event.evidence}
        match.evidence = sorted(evidence.values(), key=lambda item: item.time_sec)
        if len(event.description) > len(match.description):
            match.description = event.description
            match.state_before, match.state_after = event.state_before, event.state_after
    merged.sort(key=lambda item: (item.start_sec, item.end_sec))
    for index, event in enumerate(merged, 1):
        event.event_id = f"event_{index:05d}"
    return merged


def refinement_windows(events: list[Event], duration: float, settings: Settings) -> list[tuple[float, float]]:
    intervals = sorted((max(0, event.start_sec - settings.refine_padding_sec),
                        min(duration, event.end_sec + settings.refine_padding_sec)) for event in events)
    unions: list[list[float]] = []
    for start, end in intervals:
        if start >= end:
            end = min(duration, start + 1 / settings.refine_fps)
            if start >= end:
                start = max(0, end - 1 / settings.refine_fps)
        if unions and start <= unions[-1][1]:
            unions[-1][1] = max(unions[-1][1], end)
        else:
            unions.append([start, end])
    windows = []
    for start, end in unions:
        # Merge repeated candidates before refinement; tile long regions with overlap.
        stride = max(settings.window_sec / 2, settings.window_sec - settings.stride_sec)
        for left, right in sliding_windows(end - start, settings.window_sec, stride):
            windows.append((round(start + left, 6), round(start + right, 6)))
    return windows
