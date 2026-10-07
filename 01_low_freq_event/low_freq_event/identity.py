"""Visual identity comparison after all frames in a window have been grounded."""
from pathlib import Path
import logging
from PIL import Image
from .backend import ModelOutputError
from .video import fit_image

LOG = logging.getLogger(__name__)


def kind_group(kind):
    return "subject" if kind in {"person", "character", "animal"} else kind


def canonical(registry, entity_id):
    while registry.get(entity_id, {}).get("alias_of"):
        entity_id = registry[entity_id]["alias_of"]
    return entity_id


def combine_matches(registry, matches, gallery_limit):
    """Two positive image matches imply one instance unless co-visibility contradicts it."""
    ids = sorted({canonical(registry, eid) for eid in matches})
    if len(ids) < 2:
        return ids[0] if ids else None
    if len({kind_group(registry[eid].get("kind", "other")) for eid in ids}) > 1:
        return None
    frame_sets = [{(o["window_id"], o["frame_index"]) for o in registry[eid]["observations"]} for eid in ids]
    if any(a & b for i, a in enumerate(frame_sets) for b in frame_sets[i + 1:]):
        return None
    target = registry[ids[0]]
    for eid in ids[1:]:
        old = registry[eid]
        target["observations"].extend({**o, "entity_id": ids[0]} for o in old["observations"])
        target["first_seen_sec"] = min(target["first_seen_sec"], old["first_seen_sec"])
        target["last_seen_sec"] = max(target["last_seen_sec"], old["last_seen_sec"])
        target["gallery"] = list(dict.fromkeys(target["gallery"] + old["gallery"]))
        old["alias_of"] = ids[0]
        old["observations"] = []
    gallery = target["gallery"]
    target["gallery"] = gallery[:1] + gallery[-(gallery_limit - 1):] if gallery_limit > 1 and len(gallery) > gallery_limit else gallery[-gallery_limit:]
    return ids[0]


def crop_observations(clip, grounded: list[dict], entities: list, output: Path, window_id: str,
                      *, original_images=False) -> list[dict]:
    descriptions = {e.local_id: e for e in entities}
    observations = []
    for frame_index, result in enumerate(grounded):
        frame = clip.frames[frame_index]
        source_image = frame.image
        if original_images:
            with Image.open(output / frame.path) as stored:
                source_image = stored.convert('RGB')
        for index, box in enumerate(result["boxes"]):
            description = descriptions[box["local_id"]]
            observation_id = f"{window_id}_f{frame_index:03d}_o{index:03d}"
            crop_path = f"crops/{observation_id}.jpg"
            x1, y1, x2, y2 = box["bbox"]
            width, height = source_image.size
            left, top = int(x1 * width / 1000), int(y1 * height / 1000)
            right, bottom = max(left + 1, int(x2 * width / 1000)), max(top + 1, int(y2 * height / 1000))
            crop = source_image.crop((left, top, right, bottom))
            (output / "crops").mkdir(exist_ok=True)
            crop.save(output / crop_path, quality=92)
            observations.append({"observation_id": observation_id, "window_id": window_id,
                                 "frame_index": frame_index, "time_sec": frame.time_sec,
                                 "frame": frame.path, "bbox": box["bbox"], "crop": crop_path,
                                 **description.model_dump()})
    return observations


def resolve_frame(queries: list[dict], registry: dict, output: Path, settings, backend,
                  *, unavailable: set[str] | None = None) -> dict:
    """Use Qwen visual matches; never link using IoU or label equality."""
    if settings.identity_search == "first_confident" and len(queries) > 1:
        assignments = {}
        used = set(unavailable or ())
        for query in queries:
            resolved = resolve_frame([query], registry, output, settings, backend, unavailable=used)
            assignments.update(resolved)
            used.update(resolved.values())
        return assignments
    unavailable = {canonical(registry, eid) for eid in unavailable or ()}
    candidates = sorted((e for e in registry.values() if not e.get("alias_of") and e["entity_id"] not in unavailable),
                        key=lambda e: -e["last_seen_sec"])
    decisions = {q["observation_id"]: {"matches": set(), "possible": set(), "reasons": []} for q in queries}
    loaded = []
    for query in queries:
        with Image.open(output / query["crop"]) as img:
            loaded.append({**query, "image": fit_image(img.convert("RGB"), settings.reference_pixels)})
    # A background crop can contain a foreground person. Keep their physical roles
    # separate, so visual overlap cannot turn a landscape into that person.
    for group in dict.fromkeys(kind_group(q["kind"]) for q in loaded):
        queries_in_group = [q for q in loaded if kind_group(q["kind"]) == group]
        candidates_in_group = [e for e in candidates if kind_group(e.get("kind", "other")) == group]
        for offset in range(0, len(candidates_in_group), settings.gallery_page_size):
            refs = []
            for candidate in candidates_in_group[offset:offset + settings.gallery_page_size]:
                images = []
                for path in candidate["gallery"]:
                    with Image.open(output / path) as img:
                        images.append(fit_image(img.convert("RGB"), settings.reference_pixels))
                refs.append({**candidate, "images": images})
            try:
                matches = backend.identify_batch(queries_in_group, refs)
            except ModelOutputError as exc:
                LOG.warning("동일성 응답 확인 불가; 후보를 불확실 상태로 보존: %s", exc)
                for query in queries_in_group:
                    decision = decisions[query["observation_id"]]
                    decision["possible"].update(e["entity_id"] for e in refs)
                    decision["reasons"].append("모델 응답 검증 실패; 이 후보 페이지와의 동일성 미확정")
                continue
            for match in matches.matches:
                decision = decisions[match.query_id]
                if match.match_id:
                    decision["matches"].add(match.match_id)
                decision["possible"].update(match.possible_ids)
                decision["reasons"].append(match.reason)
            if settings.identity_search == "first_confident" and any(d["matches"] for d in decisions.values()):
                break  # One query, one explicit visual match; no further alias search.

    assignments = {}
    used = set()
    for query in queries:
        decision = decisions[query["observation_id"]]
        definite = {canonical(registry, eid) for eid in decision["matches"]}
        match = combine_matches(registry, definite, settings.gallery_images_per_entity)
        alternatives = {canonical(registry, eid) for eid in decision["possible"] | definite} - ({match} if match else set())
        if match in {canonical(registry, eid) for eid in used}:
            if match:
                alternatives.add(match)
            match = None
        if match is None:
            match = f"entity_{len(registry) + 1:04d}"
            registry[match] = {
                "entity_id": match, "label": query["label"], "kind": query["kind"],
                "appearance": query["appearance"], "gallery": [], "observations": [],
                "first_seen_sec": query["time_sec"], "last_seen_sec": query["time_sec"],
                "possible_matches": sorted(alternatives),
                "identity_status": "uncertain" if alternatives else "new",
            }
        else:
            registry[match]["identity_status"] = "matched"
            registry[match]["possible_matches"] = sorted(alternatives)
        used.add(match)
        entry = registry[match]
        observed = {**query, "entity_id": match, "identity_reasons": decision["reasons"],
                    "possible_matches": sorted(alternatives)}
        entry["observations"].append(observed)
        entry["first_seen_sec"] = min(entry["first_seen_sec"], query["time_sec"])
        entry["last_seen_sec"] = max(entry["last_seen_sec"], query["time_sec"])
        gallery = list(dict.fromkeys(entry["gallery"] + [query["crop"]]))
        limit = settings.gallery_images_per_entity
        entry["gallery"] = gallery[:1] + gallery[-(limit - 1):] if len(gallery) > limit and limit > 1 else gallery[-limit:]
        assignments[query["observation_id"]] = match
    return assignments
