from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Callable, TypeVar

from PIL import Image
from pydantic import BaseModel

from .config import Settings
from .prompts import GROUND, IDENTITY, SCAN, SYSTEM
from .schema import GroundingResult, IdentityDecision, Observation, ScanResult
from .storage import atomic_json
from .video import Clip

LOG = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ModelOutputError(RuntimeError):
    """The model exhausted its structured-output retries, not a runtime/GPU failure."""


def video_processor_arguments(clip: Clip, settings: Settings) -> dict:
    # Transformers 4.57 ignores flat video_metadata when videos_kwargs is also
    # present. Keep all video options together, including the exact PTS metadata.
    return {"videos": [clip.tensor_array()], "videos_kwargs": {
        "video_metadata": [clip.metadata()], "do_sample_frames": False,
        "size": {"shortest_edge": 32 * 32 * 4,
                 "longest_edge": settings.max_pixels * max(2, len(clip.frames))},
    }}


def parse_json_response(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    decoder = json.JSONDecoder()
    # Some models preface valid JSON with a sentence. Never repair truncated JSON.
    match = re.search(r"\{", text)
    if match:
        try:
            value, consumed = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, dict) and not text[match.start() + consumed:].strip():
                return value
    raise ValueError("Model response did not contain one complete JSON object")


class QwenBackend:
    def __init__(self, settings: Settings, trace_dir: Path):
        self.settings = settings
        self.trace_dir = trace_dir
        self.model = None
        self.processor = None
        self.call_number = len(list(trace_dir.glob("*.json")))

    def recover(self) -> None:
        """Release unused allocations before attempting the next work item."""
        import gc
        import sys

        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA GPU를 찾지 못했습니다. ./run.sh doctor로 확인하세요.")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        LOG.info("Qwen3-VL 로드: %s (NF4 4bit, %s)", self.settings.model_id, dtype)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
        self.processor = AutoProcessor.from_pretrained(
            self.settings.model_id, revision=self.settings.model_revision,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.settings.model_id,
            revision=self.settings.model_revision,
            quantization_config=quantization,
            device_map={"": 0},
            dtype=dtype,
            attn_implementation="sdpa",
        ).eval()
        LOG.info("모델 로드 완료 / GPU 할당 메모리 %.2f GiB", torch.cuda.memory_allocated() / 2**30)

    def _generate(self, prompt: str, clip: Clip | None, labeled_images: list[tuple[str, Image.Image]],
                  max_tokens: int, image_pixels: int | None = None) -> str:
        self.load()
        import torch

        content: list[dict] = []
        images = []
        for label, image in labeled_images:
            content.extend([{"type": "text", "text": label}, {"type": "image", "image": image}])
            images.append(image)
        if clip is not None:
            content.append({"type": "video", "video": "predecoded_clip"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [text], "return_tensors": "pt"}
        if images:
            kwargs.update(images=images, images_kwargs={
                "min_pixels": 32 * 32 * 16,
                "max_pixels": image_pixels or self.settings.reference_pixels,
            })
        if clip is not None:
            kwargs.update(video_processor_arguments(clip, self.settings))
        inputs = self.processor(**kwargs).to(self.model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                repetition_penalty=1.05, temperature=1.0, top_p=1.0, top_k=50,
            )
        answer = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        del inputs, generated
        return answer

    def request(self, prompt: str, schema: type[T], *, clip: Clip | None = None,
                images: list[tuple[str, Image.Image]] | None = None,
                validate: Callable[[T], T] | None = None, max_tokens: int | None = None,
                image_pixels: int | None = None) -> T:
        error = ""
        for attempt in range(self.settings.json_retries + 1):
            retry = f"\nPrevious output was invalid: {error}. Return corrected complete JSON." if error else ""
            answer = self._generate(prompt + retry, clip, images or [], max_tokens or self.settings.max_new_tokens, image_pixels)
            self.call_number += 1
            trace = {"prompt": prompt + retry, "response": answer, "attempt": attempt,
                     "clip": [clip.start_sec, clip.end_sec] if clip else None,
                     "image_labels": [label for label, _ in images or []]}
            try:
                parsed = schema.model_validate(parse_json_response(answer))
                if validate:
                    parsed = validate(parsed)
            except (ValueError, TypeError) as exc:
                error = str(exc)[:1800]
                trace["validation_error"] = error
                atomic_json(self.trace_dir / f"{self.call_number:06d}.json", trace)
                LOG.warning("JSON 검증 재시도 %d: %s", attempt + 1, error[:200])
                continue
            atomic_json(self.trace_dir / f"{self.call_number:06d}.json", trace)
            return parsed
        raise ModelOutputError(f"모델 출력 검증 실패. 원문은 {self.trace_dir}에 저장했습니다: {error}")

    def scan(self, clip: Clip, *, stage: str, context: dict, hints: list[dict]) -> ScanResult:
        task = "Find all candidate events; do not filter out brief or uncertain changes."
        if stage == "refined":
            task = ("This is the final detailed pass. Observe changes across the full frame sequence. "
                    "Output every visible action/interaction with its before/after state and refined times, "
                    "INCLUDING actions already listed in these candidate hints. Correct inaccurate hints based "
                    "on the video. An empty events array is appropriate only if no event is visible. Candidates: "
                    + json.dumps(hints, ensure_ascii=False))
        prompt = SCAN.format(
            duration=clip.duration,
            schema=json.dumps(ScanResult.model_json_schema(), ensure_ascii=False, separators=(",", ":")),
            frame_times=json.dumps({index: round(frame.time_sec - clip.start_sec, 6)
                                    for index, frame in enumerate(clip.frames)}),
            context=json.dumps(context, ensure_ascii=False), task=task,
        )
        return self.request(prompt, ScanResult, clip=clip,
                            validate=lambda result: result.validate_clip(clip.duration, len(clip.frames)))

    def localize(self, clip: Clip, observations: list[Observation]) -> list[Observation]:
        # A video response can select a frame where its described subject is absent.
        # An empty grounding result is valid evidence, not malformed JSON. Search
        # other sampled frames before giving up; never crop an unconfirmed scan box.
        found: dict[str, Observation] = {}
        attempted = {item.local_id: set() for item in observations}

        def locate(frame_index: int, group: list[Observation]) -> None:
            wanted = {item.local_id for item in group}
            for item in group:
                attempted[item.local_id].add(frame_index)

            def validate(result: GroundingResult):
                ids = [box.local_id for box in result.boxes]
                unexpected = sorted(set(ids) - wanted)
                duplicates = sorted({key for key in ids if ids.count(key) > 1})
                if unexpected or duplicates:
                    raise ValueError(f"Grounding IDs: unexpected={unexpected}, duplicates={duplicates}; "
                                     f"allowed={sorted(wanted)}. Omit targets that are not visible.")
                return result

            prompt = GROUND.format(entities=json.dumps([
                {"local_id": item.local_id, "label": item.label, "appearance": item.appearance} for item in group
            ], ensure_ascii=False))
            try:
                result = self.request(prompt, GroundingResult,
                                      images=[(f"SOURCE FRAME {frame_index}", clip.frames[frame_index].image)],
                                      validate=validate,
                                      max_tokens=min(self.settings.max_new_tokens, max(512, 100 * len(group))),
                                      image_pixels=self.settings.max_pixels)
            except ModelOutputError as exc:
                LOG.warning("위치 검출 응답 실패 · 프레임 %d · 대상 %s: %s", frame_index, sorted(wanted), exc)
                return
            boxes = {box.local_id: box.bbox for box in result.boxes}
            for item in group:
                if item.local_id in boxes:
                    found[item.local_id] = item.model_copy(update={"frame_index": frame_index,
                                                                   "bbox": boxes[item.local_id]})
                    if frame_index != item.frame_index:
                        LOG.info("대표 프레임 변경 · %s · %d → %d (원본 %.3f초)",
                                 item.local_id, item.frame_index, frame_index, clip.frames[frame_index].time_sec)

        for frame_index in sorted({item.frame_index for item in observations}):
            locate(frame_index, [item for item in observations if item.frame_index == frame_index])

        # Try later views first (useful for entrances/fades), then cover every other
        # input frame. Each entity/frame pair is inspected at most once, apart from
        # the bounded JSON retries inside request(). Already found IDs are excluded.
        for frame_index in reversed(range(len(clip.frames))):
            missing = [item for item in observations if item.local_id not in found
                       and frame_index not in attempted[item.local_id]]
            if missing:
                locate(frame_index, missing)
            if len(found) == len(observations):
                break
        missing_ids = sorted({item.local_id for item in observations} - found.keys())
        if missing_ids:
            LOG.warning("대상 위치 미확인 · %.3f~%.3f초 · %s · 관련 이벤트를 불확실 상태로 보존",
                        clip.start_sec, clip.end_sec, missing_ids)
        return [found[item.local_id] for item in observations if item.local_id in found]

    def identify(self, query: Image.Image, description: str, references: list[dict]) -> IdentityDecision:
        images = [("QUERY entity crop", query)]
        allowed_ids = set()
        for reference in references:
            allowed_ids.add(reference["entity_id"])
            for image in reference["images"]:
                images.append((f'REFERENCE {reference["entity_id"]}: {reference["appearance"]}', image))

        def validate(result: IdentityDecision):
            if result.match_id is not None and result.match_id not in allowed_ids:
                raise ValueError("match_id must be one of the reference IDs or null")
            if any(value not in allowed_ids for value in result.possible_ids):
                raise ValueError("possible_ids must only contain reference IDs")
            return result

        return self.request(IDENTITY.format(description=description), IdentityDecision,
                            images=images, validate=validate, max_tokens=512)
