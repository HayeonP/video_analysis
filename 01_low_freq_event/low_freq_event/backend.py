from __future__ import annotations

import json
import hashlib
import logging
from contextlib import nullcontext
from pathlib import Path
import re
import time
from typing import Callable, TypeVar

from PIL import Image
from pydantic import BaseModel

from .config import Settings
from .prompts import DISCOVER, GROUND, IDENTITY, SYSTEM, TEMPORAL
from .schema import EntityList, GroundingResult, Matches, ScanResult
from .storage import atomic_json, read_json
from .video import Clip

LOG = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ThinkingBudgetProcessor:
    """End a generated thinking section at a budget, leaving room for JSON.

    Only enabled explicitly for a prompt already ending in <think>. A natural
    </think> always takes precedence. No final-answer tokens are forced.
    """
    def __init__(self, prompt_length, budget, closing_token):
        self.prompt_length = prompt_length
        self.budget = budget
        self.closing_token = closing_token

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] < self.budget:
            return scores
        for index, row in enumerate(generated):
            if not (row == self.closing_token).any():
                scores[index].fill_(float('-inf'))
                scores[index, self.closing_token] = 0
        return scores


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


def close_object_before_array(text: str) -> str:
    """Repair one missing object close before an existing array close.

    Never append content at EOF, repair strings/values, or invent fields. The
    caller must still require a complete JSON object and schema validation.
    """
    stack, result = [], []
    quoted = escaped = False
    repairs = 0
    for char in text:
        if quoted:
            if escaped: escaped = False
            elif char == '\\': escaped = True
            elif char == '"': quoted = False
        elif char == '"': quoted = True
        elif char in '[{': stack.append(char)
        elif char in ']}':
            if char == ']' and stack[-2:] == ['[','{']:
                result.append('}'); stack.pop(); repairs += 1
            if not stack or stack.pop() != ('[' if char == ']' else '{'):
                raise ValueError('Mismatched JSON delimiters')
        result.append(char)
    if stack or quoted or repairs != 1:
        raise ValueError('Not a single repairable object delimiter')
    return ''.join(result)


def parse_json_response(text: str, *, reasoning_expected: bool = False,
                        repair_json_closures: bool = False, repair_root_closure: bool = False, repair_log: list | None = None) -> dict:
    text = text.strip()
    # Reasoning templates may prefill the opening <think> in the prompt, so the
    # generated answer has only a closing tag. Never parse a draft JSON object
    # from reasoning when a final-answer boundary is available.
    if reasoning_expected and '</think>' not in text:
        raise ValueError('Reasoning output ended before the final-answer boundary')
    if '</think>' in text and (reasoning_expected or not text.startswith(('{', '```'))):
        text = text.rsplit('</think>', 1)[1].strip()
    elif text.startswith('<think>'):
        raise ValueError('Reasoning output ended before the final-answer boundary')
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    decoder = json.JSONDecoder()
    # Some models preface valid JSON with a sentence. Never repair truncated JSON.
    match = re.search(r"\{", text)
    if match:
        try:
            value, consumed = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            # Explicit opt-in for frame maps: only a missing OUTERMOST brace,
            # with a fully closed nested value. Never invent a field or value.
            if repair_root_closure and text.endswith('}'):
                try:
                    value=json.loads(text[match.start():]+'}')
                    if isinstance(value,dict):
                        if repair_log is not None:repair_log.append('Appended one outermost }; complete nested values and schema still required.')
                        return value
                except (ValueError,TypeError):pass
            if repair_json_closures:
                try:
                    value=json.loads(close_object_before_array(text[match.start():]))
                    if isinstance(value,dict):
                        if repair_log is not None:repair_log.append('Inserted one missing } before an existing ]; no values changed.')
                        return value
                except (ValueError,TypeError):pass
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
        self.call_number = max((int(p.stem) for p in trace_dir.glob("*.json") if p.stem.isdigit()), default=0)
        self.cache_dir = trace_dir.parent / "request_cache"

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
            raise RuntimeError("NVIDIA CUDA GPU를 찾지 못했습니다.")
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
        self._reasoning_expected = text.rstrip().endswith('<think>')
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
        options=dict(do_sample=False,repetition_penalty=1.05,temperature=1.0,top_p=1.0,top_k=50)
        options.update(getattr(self,'_generation_options',{}) or {})
        seed=options.pop('seed',None)
        thinking_budget=options.pop('thinking_token_budget',None)
        if thinking_budget is not None and self._reasoning_expected:
            from transformers import LogitsProcessorList
            closing=self.processor.tokenizer.encode('</think>',add_special_tokens=False)
            if len(closing)!=1:raise ValueError('Bounded thinking requires a single closing token')
            options['logits_processor']=LogitsProcessorList([ThinkingBudgetProcessor(
                inputs.input_ids.shape[1],thinking_budget,closing[0])])
        rng=torch.random.fork_rng() if seed is not None else nullcontext()
        with torch.inference_mode(),rng:
            if seed is not None:torch.manual_seed(seed)
            generated = self.model.generate(
                **inputs, max_new_tokens=max_tokens, **options,
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
                image_pixels: int | None = None, generation_options: dict | None = None,
                repair_json_closures: bool = False, repair_root_closure: bool = False) -> T:
        if generation_options is not None:
            allowed={'do_sample','temperature','top_p','top_k','repetition_penalty','seed','thinking_token_budget'}
            if set(generation_options)-allowed:raise ValueError('Unsupported generation option')
            budget=generation_options.get('thinking_token_budget')
            if budget is not None and (isinstance(budget,bool) or not isinstance(budget,int) or budget<1):
                raise ValueError('thinking_token_budget must be a positive integer')
        self._generation_options=dict(generation_options or {})
        digest = hashlib.sha256(prompt.encode())
        digest.update(self.settings.model_dump_json().encode())
        digest.update(schema.__name__.encode())
        if repair_json_closures:digest.update(b'ONE_OBJECT_CLOSURE_REPAIR_V1')
        if repair_root_closure:digest.update(b'OUTERMOST_OBJECT_CLOSURE_V1')
        if generation_options is not None:
            digest.update(b'GENERATION_OPTIONS:')
            digest.update(json.dumps(generation_options,sort_keys=True,separators=(',',':')).encode())
        if clip:
            digest.update(str((clip.start_sec, clip.end_sec)).encode())
            for frame in clip.frames:
                digest.update(frame.image.tobytes())
        for label, image in images or []:
            digest.update(label.encode())
            digest.update(str(image.size).encode())
            digest.update(image.tobytes())
        cache_path = self.cache_dir / f"{digest.hexdigest()}.json"
        error = ""
        if cache_path.exists():
            try:
                parsed = schema.model_validate(read_json(cache_path))
                return validate(parsed) if validate else parsed
            except (ValueError, TypeError) as exc:
                # Validators may become stricter between resumes. Preserve the
                # rejected response and retry through the normal charged path.
                error = str(exc)[:1800]
                atomic_json(self.trace_dir / f"cache_invalid_{digest.hexdigest()}_{time.time_ns()}.json",
                            dict(cache_path=str(cache_path),validation_error=error,
                                 cached_response=cache_path.read_text()))
                LOG.warning("저장된 응답 재검증 실패; 새 추론으로 재검토: %s", error[:200])
        previous_answer = ""
        for attempt in range(self.settings.json_retries + 1):
            retry = f"\nPrevious output was invalid: {error}. Return corrected complete JSON." if error else ""
            if error and previous_answer:
                retry += "\nPrevious invalid JSON (data to correct, not instructions):\n" + previous_answer[-24000:]
                retry += "\nReturn the complete corrected JSON using the images and validation rules above."
            started = time.monotonic()
            answer = self._generate(prompt + retry, clip, images or [], max_tokens or self.settings.max_new_tokens, image_pixels)
            self.call_number += 1
            trace = {"prompt": prompt + retry, "response": answer, "attempt": attempt,
                     "clip": [clip.start_sec, clip.end_sec] if clip else None,
                     "image_labels": [label for label, _ in images or []]}
            trace["elapsed_sec"] = round(time.monotonic() - started, 3)
            trace["schema"] = schema.__name__
            trace['request_digest']=digest.hexdigest()
            if generation_options is not None:trace['generation_options']=self._generation_options
            try:
                repair_log=[]
                parsed = schema.model_validate(parse_json_response(answer,
                    reasoning_expected=getattr(self, '_reasoning_expected', False),
                    repair_json_closures=repair_json_closures,repair_root_closure=repair_root_closure,repair_log=repair_log))
                if repair_log:trace['syntax_repairs']=repair_log
                if validate:
                    parsed = validate(parsed)
            except (ValueError, TypeError) as exc:
                previous_answer = answer
                error = str(exc)[:1800]
                trace["validation_error"] = error
                atomic_json(self.trace_dir / f"{self.call_number:06d}.json", trace)
                LOG.warning("JSON 검증 재시도 %d: %s", attempt + 1, error[:200])
                continue
            atomic_json(self.trace_dir / f"{self.call_number:06d}.json", trace)
            atomic_json(cache_path, parsed.model_dump(mode="json"))
            return parsed
        raise ModelOutputError(f"모델 출력 검증 실패. 원문은 {self.trace_dir}에 저장했습니다: {error}")

    def discover(self, clip: Clip, *, known_entities=None, grounded=None) -> EntityList:
        prompt = DISCOVER
        known = {e.local_id for e in known_entities or []}
        if known_entities is not None:
            from .context import context_prompt, annotated_clip
            prompt += context_prompt(clip, known_entities, grounded)
            prompt += ('\n이번 응답에는 위 목록에 없는 추가 물체·동물·배경만 나열하세요. 사람/캐릭터는 추가하지 마세요. '
                       'YOLO가 이미 검출한 물체를 다른 이름으로 중복 생성하지 마세요. 설산·하늘·방 같은 배경은 직접 분석하세요. '
                       '추가 대상의 local_id는 s1, s2처럼 s로 시작해야 합니다. 추가 대상이 없으면 entities는 빈 배열입니다.')
            clip = annotated_clip(clip, grounded)
        def validate(result):
            if known_entities is not None:
                for e in result.entities:
                    if e.local_id in known or not e.local_id.startswith('s') or e.kind in {'person','character'}:
                        raise ValueError('Return supplemental non-person entities only, with new s-prefixed IDs; preserve supplied people')
            return result
        return self.request(prompt, EntityList, clip=clip, validate=validate, max_tokens=1600)

    def scan(self, clip: Clip, entities, grounded) -> ScanResult:
        from .temporal import extract_frame_events
        return extract_frame_events(self, clip, entities, grounded)

    def narrate(self, clip, entities, grounded):
        from .schema import WindowNarrative, ActivityDescription, EntityActivity
        from .context import context_prompt, annotated_clip, entity_token
        prompt = "<entity_context>" + context_prompt(clip, entities, grounded) + "</entity_context>\nOUTPUT TASK" + """
위 엔티티 정보와 윈도의 모든 샘플 이미지를 시간순으로 보고 한국어로 서술하세요.
배경(background)은 장소, 조명, 배치와 시간에 따른 변화를 설명하세요.
summary는 '현재 배경은 …이고, 인물 1은 …하고 있다'처럼 자연스러운 문장으로 작성하세요.
activities에는 주어진 모든 entity_id를 정확히 한 번씩 포함하고, 각 인물/물체가 무엇을
하고 있는지 description으로 설명하세요. 움직이지 않는 물체의 상태도 포함하세요.
관찰 불가능한 행동, 의도, 샘플 사이의 동작을 추측하지 말고 불확실하면 uncertain=true로 표시하세요.
전역 ID를 새로 만들거나 합치지 마세요. 시간 표현은 제공된 원본 시각을 쓰세요.
JSON: {"background":"…", "summary":"…", "activities":[{"entity_id":"…", "description":"…", "uncertain":false}]}
"""
        expected = [e.local_id for e in entities]
        aliases = {entity_token(eid): eid for eid in expected}
        def validate(narrative):
            seen = set()
            for activity in narrative.activities:
                eid = aliases.get(activity.entity_id, activity.entity_id)
                if eid not in expected or eid in seen:
                    raise ValueError(f'Unknown or duplicate entity_id: {eid}. Allowed IDs: {expected}')
                activity.entity_id = eid
                seen.add(eid)
            # Missing entries are analyzed separately, never fabricated.
            return narrative
        narrative = self.request(prompt, WindowNarrative,
            clip=annotated_clip(clip, grounded), validate=validate)
        activities = {a.entity_id: a for a in narrative.activities}
        missing = [e for e in entities if e.local_id not in activities]
        if missing:
            LOG.info('장면 서술에서 누락된 엔티티 %d개를 개별 분석합니다.', len(missing))
        for entity in missing:
            target_frames = [dict(frame, boxes=[box for box in frame['boxes']
                if box['local_id'] == entity.local_id]) for frame in grounded]
            target_prompt = ('<entity_context>' + context_prompt(clip, [entity], target_frames)
                + '</entity_context>\nOUTPUT TASK\n'
                + f'분석 대상은 {entity.local_id} ({entity_token(entity.local_id)}), {entity.label} 하나입니다. '
                + '윈도의 모든 이미지를 보고 이 대상의 행동 또는 정지 상태와 변화를 한국어로 설명하세요. '
                + '대상이 배경이면 배경의 상태를 설명하세요. 보이지 않는 시점의 행동은 추측하지 마세요. '
                + '판단할 수 없으면 그 이유를 description에 쓰고 uncertain=true로 표시하세요. '
                + '다른 대상을 대신 설명하지 마세요. entity_id는 출력하지 마세요. '
                + 'JSON: {"description":"대상의 상태 또는 행동", "uncertain":false}')
            description = self.request(target_prompt, ActivityDescription,
                clip=annotated_clip(clip, target_frames), max_tokens=512)
            activities[entity.local_id] = EntityActivity(entity_id=entity.local_id, **description.model_dump())
        narrative.activities = [activities[eid] for eid in expected]
        return narrative

    def scan_intervals(self, clip: Clip, entities, grounded) -> ScanResult:
        """Legacy interval prediction, retained for old experiment comparisons."""
        visibility = {e.local_id: [i for i, frame in enumerate(grounded)
                                  if any(box["local_id"] == e.local_id for box in frame["boxes"])] for e in entities}
        prompt = TEMPORAL.format(
            duration=clip.duration,
            frame_times=json.dumps({i: round(f.time_sec - clip.start_sec, 6) for i, f in enumerate(clip.frames)}),
            entities=json.dumps([e.model_dump() for e in entities], ensure_ascii=False),
            visibility=json.dumps(visibility, ensure_ascii=False),
        )
        if self.settings.perception == 'person_reid':
            from .perception import annotated_clip
            clip = annotated_clip(clip, grounded)
            prompt += ('\n영상의 색상 박스와 P번호는 별도 검출·동일인 연결 결과입니다. '
                       'P1은 person_0001, P2는 person_0002 등입니다. 위치가 바뀌어도 이 ID를 그대로 사용하세요. '
                       '박스/번호 자체는 영상 속 물건이나 사건이 아닙니다. 화면 밖으로 사라진 사람의 행동을 연장하지 마세요.')

        def validate(result):
            for event in result.events:
                conflicts = []
                if event.continues_before and event.start_frame != 0:
                    event.continues_before = False
                    conflicts.append('시작')
                if event.continues_after and event.end_frame != len(clip.frames)-1:
                    event.continues_after = False
                    conflicts.append('종료')
                if conflicts:
                    # Do not expand a short, explicitly predicted span to the
                    # whole window because contradictory flags say it continues.
                    event.uncertain = True
                    event.boundary_note += ' (모델의 ' + '/'.join(conflicts) + ' 지속 플래그가 프레임 구간과 충돌하여 프레임 구간을 우선함)'
            result.validate_clip(len(clip.frames))
            if {e.local_id for e in result.entities} != set(visibility):
                raise ValueError("Preserve the supplied entity IDs exactly")
            required = {eid for eid, frames in visibility.items() if frames}
            referenced = {eid for event in result.events for eid in event.entity_ids}
            if required - referenced:
                raise ValueError(f"Include an observed event or sustained visible state for: {sorted(required - referenced)}")
            for event in result.events:
                available = {i for eid in event.entity_ids for i in visibility[eid]}
                if available and (event.start_frame not in available or event.end_frame not in available):
                    # Preserve the model estimate but explicitly flag disagreement with image localization.
                    event.uncertain = True
                    event.boundary_note += " (구간 경계와 bbox 관찰 불일치)"
            return result

        return self.request(prompt, ScanResult, clip=clip, validate=validate)

    def ground_frame(self, frame, entities, frame_index=0, *, known_entities=None, detected_boxes=None) -> GroundingResult:
        allowed = {e.local_id for e in entities}
        prefix = f"f{frame_index}_new"

        def validate(result):
            ids = [box.local_id for box in result.boxes]
            new_ids = [e.local_id for e in result.new_entities]
            if known_entities is not None and any(e.kind in {'person', 'character'} for e in result.new_entities):
                raise ValueError('People already have detector IDs; do not create new people or characters')
            if len(new_ids) != len(set(new_ids)) or any(not eid.startswith(prefix) or eid in allowed for eid in new_ids):
                raise ValueError(f"New entity IDs must be unique and start with {prefix}")
            if not set(new_ids) <= set(ids):
                raise ValueError("Each new entity must have a visible bbox in this frame")
            if len(set(ids)) != len(ids) or not set(ids) <= allowed | set(new_ids):
                raise ValueError("Use each supplied local_id at most once; omit absent objects")
            kinds = {e.local_id: e.kind for e in [*entities, *result.new_entities]}
            subject_boxes = {}
            for box in result.boxes:
                if kinds[box.local_id] not in {"person", "character", "animal"}:
                    continue
                if box.bbox in subject_boxes:
                    raise ValueError(
                        f"{subject_boxes[box.bbox]} and {box.local_id} have identical copied coordinates. "
                        "Inspect the image and return a separate tight bbox for each visible individual; "
                        "do not copy the full-frame box to multiple people. Omit a person if not visible."
                    )
                subject_boxes[box.bbox] = box.local_id
            return result

        prompt = GROUND.format(entities=json.dumps([e.model_dump() for e in entities], ensure_ascii=False), prefix=prefix)
        if known_entities is not None:
            prompt += ('\n아래 대상은 YOLO가 이미 검출하고 ID를 배정했습니다. 이 대상의 박스를 반환하거나 같은 대상을 새로 만들지 마세요. '
                       '사람/캐릭터는 추가하지 마세요. 목록 밖 물체·동물·배경은 직접 확인하세요.\n'
                       + json.dumps({'known_entities':[e.model_dump() for e in known_entities], 'detections':detected_boxes or []}, ensure_ascii=False))
        try:
            return self.request(prompt, GroundingResult,
                                images=[("FRAME", frame.image)], validate=validate,
                                max_tokens=max(1200, 140 * len(entities)), image_pixels=self.settings.max_pixels)
        except ModelOutputError as exc:
            # A failed detection is not evidence that the frame is empty. Keep
            # the frame and explicit failure metadata; never crop invalid boxes.
            LOG.warning("프레임 %d bbox 검출 실패; 실패 표시 후 영상 분석 계속: %s", frame_index, exc)
            return GroundingResult(boxes=[], warnings=["bbox 검출 실패: 모델 응답 검증 재시도 소진. " + str(exc)])

    def identify_batch(self, queries: list[dict], references: list[dict]) -> Matches:
        images = []
        query_map = {f"Q{i + 1}": q["observation_id"] for i, q in enumerate(queries)}
        reference_map = {f"R{i + 1}": e["entity_id"] for i, e in enumerate(references)}
        for index, q in enumerate(queries, 1):
            images.append((f'QUERY Q{index}', q["image"]))
        for index, entry in enumerate(references, 1):
            for img in entry["images"]:
                images.append((f'REFERENCE R{index}', img))
        query_ids = set(query_map)
        reference_ids = set(reference_map)

        def validate(result):
            ids = [m.query_id for m in result.matches]
            if len(ids) != len(set(ids)) or set(ids) != query_ids:
                raise ValueError(f"Return each query exactly once: {sorted(query_ids)}")
            for m in result.matches:
                if m.match_id and m.match_id not in reference_ids or not set(m.possible_ids) <= reference_ids:
                    raise ValueError(f"Only reference IDs are allowed: {sorted(reference_ids)}")
            return result

        prompt = IDENTITY.format(queries=sorted(query_ids), references=sorted(reference_ids))
        result = self.request(prompt, Matches, images=images, validate=validate,
                              max_tokens=max(768, 220 * len(queries)))
        # Stable global IDs never depend on the temporary page-local model tokens.
        translated = result.model_copy(deep=True)
        for match in translated.matches:
            match.query_id = query_map[match.query_id]
            match.match_id = reference_map[match.match_id] if match.match_id else None
            match.possible_ids = [reference_map[eid] for eid in match.possible_ids]
        # Joint comparison can collapse several people onto one reference. Ask
        # about the conflicting crops individually instead of discarding every
        # decision on the page and repeating the same failed joint request.
        chosen = [m.match_id for m in translated.matches if m.match_id]
        conflicts = {eid for eid in chosen if chosen.count(eid) > 1}
        if conflicts:
            by_id = {q["observation_id"]: q for q in queries}
            for i, match in enumerate(translated.matches):
                if match.match_id in conflicts:
                    LOG.info("동일성 배치 충돌 · 해당 crop을 개별 재비교: %s", match.query_id)
                    retry = self.identify_batch([by_id[match.query_id]], references).matches[0]
                    retry.reason = "배치 충돌 후 개별 비교: " + retry.reason
                    translated.matches[i] = retry
        # resolve_frame still enforces distinct IDs for co-visible subjects if
        # individual comparisons disagree; an uncertain observation is retained.
        return translated
