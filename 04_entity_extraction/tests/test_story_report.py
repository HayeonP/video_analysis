"""Report runtime configuration and partial quality must stay distinguishable."""
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from story_report import write_report


class ReportTests(unittest.TestCase):
    def make_output(self, files):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output=Path(temporary.name)
        for name,data in files.items():
            path=output/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(data))
        return output

    def render_output(self, output):
        write_report(output)
        return (output/'REPORT.html').read_text(),(output/'REPORT.md').read_text()

    def render(self, files):
        return self.render_output(self.make_output(files))

    def evaluation_output(self, *, mask_enabled=False):
        evaluation=dict(annotation_ids=['a','b','c'],mapped_annotation_ids=['a','b','c'],
                        current=dict(correct_same_id_pairs=3,same_id_pairs=3,false_merge_pairs=0))
        regression=dict(passed=True,final=dict(gate=dict(passed=True),evaluation=evaluation),
                        provisional=dict(gate=dict(passed=False),evaluation=evaluation))
        output=self.make_output({
            'run_config.json':dict(mask_refinement=mask_enabled),
            'global_entities.json':dict(entities=[{}, {}, {}]),
            'global_provisional.json':dict(entities=[{}, {}, {}, {}]),
            'local_entities.json':dict(shots=[dict(shot_id='s',entities=[])]),
            'reviewed_tracking.json':dict(fingerprint='reviewed'),
            'mask_refinement/bundle.json':dict(fingerprint='masked'),
            'mask_refinement/local.json':dict(shots=[dict(shot_id='s',entities=[{}])]),
            'observations.json':dict(observations=[]),
            'labels.json':dict(identities=['a','b','c']),
            'baseline.json':dict(observations=[]),
            'regression_evaluation.json':regression})
        return output

    def bind_evaluation(self, output):
        regression=json.loads((output/'regression_evaluation.json').read_text())
        mask_enabled=json.loads((output/'run_config.json').read_text())['mask_refinement']
        files=dict(final_catalog='global_entities.json',provisional_catalog='global_provisional.json',
                   local_entities='mask_refinement/local.json' if mask_enabled else 'local_entities.json',
                   bundle='mask_refinement/bundle.json' if mask_enabled else 'reviewed_tracking.json',
                   exported_observations='observations.json',run_config='run_config.json',
                   labels='labels.json',baseline='baseline.json')
        regression['artifacts']={name:dict(path=str(output/path),sha256=hashlib.sha256((output/path).read_bytes()).hexdigest())
                                 for name,path in files.items()}
        (output/'regression_evaluation.json').write_text(json.dumps(regression))
        return regression

    def test_runtime_budget_and_skipped_masks_replace_old_defaults(self):
        page,markdown=self.render({
            'run_config.json':dict(mask_refinement=False,detector='custom-detector.pt',
                settings=dict(model_id='example-vlm-8b'),global_model_id='example-vlm-8b',
                qwen_max_calls=48,global_review_max_calls=7),
            'qwen_budget.json':dict(limit=51,generations=13),
            # Stale mask metrics must not turn disabled stages back into actual work.
            'raw_track_masks/summary.json':dict(mask_frames=99999)})
        for text in (page,markdown):
            self.assertIn('전체 생성 상한은 51회',text)
            self.assertIn('집중 동일성 검토 상한은 7회',text)
            self.assertIn('custom-detector.pt',text)
            self.assertIn('example-vlm-8b',text)
            self.assertIn('생략 (실행 설정)',text)
            self.assertNotIn('24회',text)
            self.assertNotIn('17샷',text)
            self.assertNotIn('99999',text)
            self.assertNotIn('독립 SAM → 마스크 비교',text)
            self.assertNotIn('각 트랙의 전경을 독립적으로 추적',text)

    def test_current_evaluator_integer_counts_render_with_verified_provenance(self):
        output=self.evaluation_output()
        result=self.bind_evaluation(output)
        for key in ('final','provisional'):
            result[key]['evaluation'].update(annotation_ids=3,mapped_annotation_ids=3)
        (output/'regression_evaluation.json').write_text(json.dumps(result))
        page,markdown=self.render_output(output)
        self.assertIn('| 최종 목록 | 통과 | 3 / 3 |',markdown)
        self.assertIn('3 / 3',page)

    def test_partial_regression_pass_does_not_override_unresolved_quality(self):
        audit=dict(status='needs_review',initial_entities=6,final_entities=3,
                   generations=4,max_generations=8,joins=[{},{}],
                   unresolved=[dict(reason='검토 필요 <script>')],unreviewed_candidates=[{}],errors=[])
        output=self.evaluation_output()
        (output/'global_entities.json').write_text(json.dumps(dict(entities=[{}, {}, {}],identity_reconciliation=audit)))
        (output/'quality.json').write_text(json.dumps(dict(status='needs_review',issues=['Unreviewed candidates remain.'])))
        regression=self.bind_evaluation(output)
        page,markdown=self.render_output(output)
        self.assertIn('needs_review',page)
        self.assertIn('고정 부분 정답 회귀',page)
        self.assertIn('통과',page)
        self.assertIn('실패',page)
        self.assertIn('영상 전체의 검출·식별 정확도를 의미하지 않습니다',markdown)
        self.assertIn('미해결 / 미검토 / 오류',markdown)
        self.assertNotIn('<script>',page)
        downloads=re.findall(r'href="data:application/json;base64,([^"]+)"',page)
        self.assertEqual([json.loads(base64.b64decode(value)) for value in downloads],[audit,regression])

    def assert_unverified(self, output):
        page,markdown=self.render_output(output)
        for text in (page,markdown):
            self.assertIn('재검증 필요',text)
            self.assertNotIn('회귀 판정',text)
            self.assertNotIn('동일 ID 쌍',text)
        return page,markdown

    def test_legacy_pass_without_input_hashes_is_not_shown_as_current(self):
        self.assert_unverified(self.evaluation_output())

    def test_surviving_evaluated_snapshot_does_not_hide_changed_current_catalog(self):
        output=self.evaluation_output()
        regression=self.bind_evaluation(output)
        snapshot=output/'old_catalog.json'
        snapshot.write_bytes((output/'global_entities.json').read_bytes())
        regression['artifacts']['final_catalog']['path']=str(snapshot)
        (output/'regression_evaluation.json').write_text(json.dumps(regression))
        (output/'global_entities.json').write_text(json.dumps(dict(entities=[{}])))
        _,markdown=self.assert_unverified(output)
        self.assertIn('final_catalog: 현재 파일과 평가 시점의 SHA-256이 다릅니다.',markdown)

    def test_changed_evaluation_inputs_suppress_prior_metrics(self):
        for filename in ['local_entities.json','reviewed_tracking.json','observations.json',
                         'global_provisional.json','labels.json','baseline.json','run_config.json']:
            with self.subTest(filename=filename):
                output=self.evaluation_output()
                self.bind_evaluation(output)
                path=output/filename
                value=json.loads(path.read_text());value['changed']=True
                path.write_text(json.dumps(value))
                self.assert_unverified(output)

    def test_actual_mask_mode_selects_evaluation_inputs(self):
        for mask_enabled in (False,True):
            with self.subTest(mask_enabled=mask_enabled):
                output=self.evaluation_output(mask_enabled=mask_enabled)
                self.bind_evaluation(output)
                unused=['local_entities.json','reviewed_tracking.json'] if mask_enabled else [
                    'mask_refinement/local.json','mask_refinement/bundle.json']
                for name in unused:(output/name).write_text('{}')
                _,markdown=self.render_output(output)
                self.assertIn('| 최종 목록 | 통과 |',markdown)
                self.assertNotIn('재검증 필요',markdown)
                active='mask_refinement/local.json' if mask_enabled else 'local_entities.json'
                (output/active).write_text('{}')
                self.assert_unverified(output)

    def test_missing_records_do_not_invent_a_budget_or_evaluation_pass(self):
        page,markdown=self.render({})
        self.assertIn('전체 생성 상한은 미기록회',markdown)
        self.assertIn('미평가',page)
        self.assertNotIn('고정 부분 정답 회귀</h3>',page)


if __name__=='__main__':
    unittest.main()
