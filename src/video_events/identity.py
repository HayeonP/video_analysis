from __future__ import annotations

import itertools
from pathlib import Path

from PIL import Image

from .config import Settings
from .schema import Observation
from .video import Clip, fit_image


def box_iou(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def compatible(a: str, b: str) -> bool:
    return a == b or (a in {"person", "character", "animal", "other"}
                      and b in {"person", "character", "animal", "other"})


class EntityRegistry:
    def __init__(self, output: Path, settings: Settings, state: dict | None = None):
        self.output = output
        self.settings = settings
        self.entries = (state or {}).get("entries", {})
        self.aliases = (state or {}).get("aliases", {})
        self.distinct_pairs = {tuple(pair) for pair in (state or {}).get("distinct_pairs", [])}
        self.next_id = (state or {}).get("next_id", 1)

    def dump(self) -> dict:
        return {"entries": self.entries, "aliases": self.aliases, "next_id": self.next_id,
                "distinct_pairs": sorted(self.distinct_pairs)}

    def canonical(self, entity_id: str | None) -> str | None:
        while entity_id in self.aliases:
            entity_id = self.aliases[entity_id]
        return entity_id

    def _visual_match(self, crop: Image.Image, description: str, candidates: list[dict], backend):
        matches, possible, reasons = set(), set(), []
        size = self.settings.gallery_page_size
        for offset in range(0, len(candidates), size):
            page = candidates[offset:offset + size]
            references = []
            for entry in page:
                images = []
                for name in entry["gallery"][-self.settings.gallery_images_per_entity:]:
                    with Image.open(self.output / name) as image:
                        images.append(image.convert("RGB"))
                references.append({"entity_id": entry["entity_id"], "appearance": entry["appearance"],
                                   "images": images})
            decision = backend.identify(crop, description, references)
            if decision.match_id:
                matches.add(decision.match_id)
            possible.update(decision.possible_ids)
            reasons.append(decision.reason)
        if len(matches) == 1:
            return next(iter(matches)), sorted(possible), reasons
        return None, sorted(possible | matches), reasons

    def resolve(self, observations: list[Observation], clip: Clip, backend) -> dict[str, str]:
        assignments, used = {}, set()
        for observation in observations:
            frame = clip.frames[observation.frame_index]
            x1, y1, x2, y2 = observation.bbox
            crop = frame.image.crop((int(x1 / 1000 * frame.image.width), int(y1 / 1000 * frame.image.height),
                                     max(int(x1 / 1000 * frame.image.width) + 1, int(x2 / 1000 * frame.image.width)),
                                     max(int(y1 / 1000 * frame.image.height) + 1, int(y2 / 1000 * frame.image.height))))
            crop = fit_image(crop, self.settings.reference_pixels)
            candidates = sorted((entry for key, entry in self.entries.items()
                                 if key not in self.aliases and key not in used
                                 and compatible(observation.kind, entry["kind"])),
                                key=lambda entry: -entry["last_seen_sec"])
            # Identical source frame + nearly identical box is a cheap exact-overlap anchor.
            anchors = [entry["entity_id"] for entry in candidates if any(
                abs(previous["time_sec"] - frame.time_sec) < 1e-6
                and box_iou(previous["bbox"], observation.bbox) >= 0.85
                for previous in entry["observations"])]
            match, possible, reasons = None, [], []
            if len(anchors) == 1:
                match, reasons = anchors[0], ["동일 원본 프레임의 동일 위치에서 연결"]
            elif candidates:
                match, possible, reasons = self._visual_match(crop, observation.appearance, candidates, backend)
            if match is None:
                match = f"entity_{self.next_id:04d}"
                self.next_id += 1
                self.entries[match] = {
                    "entity_id": match, "kind": observation.kind, "label": observation.label,
                    "appearance": observation.appearance, "first_seen_sec": frame.time_sec,
                    "last_seen_sec": frame.time_sec, "gallery": [], "observations": [],
                    "identity_status": "provisional" if possible else "new",
                    "possible_matches": possible, "match_evidence": [],
                }
            else:
                self.entries[match]["identity_status"] = "matched"
            entry = self.entries[match]
            crop_path = f"entities/{match}/t{round(frame.time_sec * 1_000_000):012d}.jpg"
            path = self.output / crop_path
            path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(path, quality=92)
            if crop_path not in entry["gallery"]:
                entry["gallery"].append(crop_path)
                # Preserve first view plus recent views, keeping prompt size bounded.
                limit = self.settings.gallery_images_per_entity
                entry["gallery"] = (entry["gallery"][:1] + entry["gallery"][-(limit - 1):]
                                    if limit > 1 and len(entry["gallery"]) > limit else entry["gallery"][-limit:])
            entry["observations"].append({"time_sec": frame.time_sec, "bbox": list(observation.bbox),
                                          "frame": frame.path, "crop": crop_path,
                                          "appearance": observation.appearance})
            entry["first_seen_sec"] = min(entry["first_seen_sec"], frame.time_sec)
            entry["last_seen_sec"] = max(entry["last_seen_sec"], frame.time_sec)
            entry["match_evidence"].append({"time_sec": frame.time_sec, "reasons": reasons})
            assignments[observation.local_id] = match
            used.add(match)
        # One extraction represents each distinct entity once. Prevent later merging
        # of different entities explicitly distinguished within that extraction.
        for a, b in itertools.combinations(sorted(used), 2):
            self.distinct_pairs.add((a, b))
        return assignments

    def reconcile(self, backend) -> None:
        for key, entry in list(self.entries.items()):
            if key in self.aliases or entry["identity_status"] != "provisional":
                continue
            candidates = [other for other_id, other in self.entries.items()
                          if other_id != key and other_id not in self.aliases
                          and compatible(entry["kind"], other["kind"])
                          and tuple(sorted((key, other_id))) not in self.distinct_pairs]
            if not candidates:
                continue
            with Image.open(self.output / entry["gallery"][-1]) as image:
                match, possible, reasons = self._visual_match(image.convert("RGB"), entry["appearance"], candidates, backend)
            if match:
                destination = self.entries[match]
                self.aliases[key] = match
                destination["observations"].extend(entry["observations"])
                destination["first_seen_sec"] = min(destination["first_seen_sec"], entry["first_seen_sec"])
                destination["last_seen_sec"] = max(destination["last_seen_sec"], entry["last_seen_sec"])
                destination["identity_status"] = "matched"
                destination["match_evidence"].append({"merged_id": key, "reasons": reasons})
                self.distinct_pairs = {tuple(sorted((self.canonical(a), self.canonical(b))))
                                       for a, b in self.distinct_pairs if self.canonical(a) != self.canonical(b)}
            else:
                entry["possible_matches"] = possible

    def context(self, start: float, end: float) -> list[dict]:
        entries = sorted((entry for key, entry in self.entries.items() if key not in self.aliases),
                         key=lambda entry: min(abs(entry["last_seen_sec"] - start),
                                               abs(entry["first_seen_sec"] - end)))
        return [{key: entry[key] for key in ("entity_id", "kind", "label", "appearance")}
                for entry in entries[:12]]

    def export(self) -> list[dict]:
        result = []
        for key, entry in self.entries.items():
            if key in self.aliases:
                continue
            record = dict(entry)
            record["aliases"] = [alias for alias in self.aliases if self.canonical(alias) == key]
            record["possible_matches"] = sorted({self.canonical(value) for value in record["possible_matches"]
                                                  if self.canonical(value) != key})
            result.append(record)
        return result
