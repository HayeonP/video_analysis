"""CPU integration checks for persisted execution settings and catalog provenance."""
import contextlib
from copy import deepcopy
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event')]
import story_pipeline as pipeline
from video_context import normalize_video_context


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.input=self.root/'input';self.input.mkdir()
        self.output=self.root/'output';self.output.mkdir()
        self.bundle=dict(stats=dict(dense_frames=2,tracks=0),shots=[])
        self.local=dict(shots=[],failures=[])

    def save(self,name,value):
        path=self.output/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def read(self,name):
        return json.loads((self.output/name).read_text())

    def catalog(self,**fields):
        return dict(entities=[],mapping={},external_context={},
                    global_input_fingerprint=pipeline.global_input_fingerprint(self.bundle,self.local),**fields)

    @contextlib.contextmanager
    def invocation(self,*args):
        # No decoder, detector, VLM, GPU initialization or model download is used.
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(pipeline,'ROOT',self.root))
            stack.enter_context(patch.object(sys,'argv',['story_pipeline',
                '--input',str(self.input),'--output',str(self.output),'--analysis-frames','dense',*args]))
            stack.enter_context(patch.dict(sys.modules,{
                'torch':SimpleNamespace(set_num_threads=Mock(),cuda=SimpleNamespace(empty_cache=Mock())),
                'cv2':SimpleNamespace(setNumThreads=Mock())}))
            stack.enter_context(patch.object(pipeline.logging,'FileHandler',return_value=logging.NullHandler()))
            stack.enter_context(patch.object(pipeline.logging,'basicConfig'))
            stack.enter_context(patch.object(pipeline.LOG,'exception'))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            yield

    def test_resume_inherits_budget_models_context_masks_and_original_video(self):
        original=self.root/'original.mp4'
        context=normalize_video_context(dict(title='Unrelated film',media_type='trailer'))
        self.save('run_config.json',dict(input=str(self.input),qwen_max_calls=24,detector='custom-detector.pt',
            settings=dict(model_id='example/local-model',model_revision='local-pinned'),
            global_model_id='example/global-model',global_model_revision='global-pinned',
            global_review_max_calls=5,global_initialization='whole',mask_refinement=False,original_video=str(original),context=context))
        ledger=dict(limit=31,generations=12,events=[])
        self.save('qwen_budget.json',ledger)
        self.save('video_context.json',context)
        with self.invocation('--stage','tracking'), patch.object(
                pipeline,'build_dense_tracks',return_value=self.bundle) as track:
            pipeline.main()
        saved=self.read('run_config.json')
        self.assertEqual(saved['qwen_max_calls'],31)
        self.assertEqual(saved['detector'],'custom-detector.pt')
        self.assertEqual(saved['settings']['model_id'],'example/local-model')
        self.assertEqual(saved['settings']['model_revision'],'local-pinned')
        self.assertEqual(saved['global_model_id'],'example/global-model')
        self.assertEqual(saved['global_model_revision'],'global-pinned')
        self.assertEqual(saved['global_review_max_calls'],5)
        self.assertEqual(saved['global_initialization'],'whole')
        self.assertFalse(saved['mask_refinement'])
        self.assertEqual(saved['original_video'],str(original))
        self.assertEqual(saved['context'],context)
        self.assertEqual(self.read('video_context.json'),context)
        self.assertEqual(self.read('qwen_budget.json'),ledger)
        self.assertEqual(track.call_args.kwargs['original_video'],original)
        self.assertEqual(track.call_args.kwargs['model'],'custom-detector.pt')
        self.assertEqual(self.read('status.json')['status'],'stage_complete')

    def test_new_run_preserves_links_with_bounded_focused_review(self):
        with self.invocation('--stage','tracking'), patch.object(
                pipeline,'build_dense_tracks',return_value=self.bundle):
            pipeline.main()
        config=self.read('run_config.json')
        self.assertEqual(config['global_model_id'],pipeline.THINKING_MODEL)
        self.assertEqual(config['global_model_revision'],pipeline.THINKING_REVISION)
        self.assertEqual(config['global_strategy'],'linked')
        self.assertEqual(config['global_review_max_calls'],96)
        self.assertEqual(config['settings']['model_id'],'Qwen/Qwen3-VL-8B-Instruct')

    def test_converted_checkpoint_requires_matching_source_provenance(self):
        checkpoint=self.root/'models/Qwen3-VL-8B-Thinking-NF4'
        checkpoint.mkdir(parents=True)
        (checkpoint/'config.json').write_text('{}')
        provenance=checkpoint/'source_provenance.json'
        with patch.object(pipeline,'ROOT',self.root):
            self.assertEqual(pipeline.default_global_model(),pipeline.THINKING_MODEL)
            provenance.write_text(json.dumps(dict(source_model=pipeline.THINKING_MODEL,source_revision='other')))
            self.assertEqual(pipeline.default_global_model(),pipeline.THINKING_MODEL)
            provenance.write_text(json.dumps(dict(source_model=pipeline.THINKING_MODEL,
                                                   source_revision=pipeline.THINKING_REVISION)))
            self.assertEqual(pipeline.default_global_model(),str(checkpoint))

    def test_export_rejects_old_catalog_after_local_stage_has_changed_saved_context(self):
        old_context=normalize_video_context(dict(title='Original source',media_type='trailer'))
        new_context=normalize_video_context(dict(title='Corrected source',media_type='trailer'))
        # A local-only restart has already updated both saved settings and context.
        # The original global catalog still has its own immutable input provenance.
        self.save('run_config.json',dict(input=str(self.input),mask_refinement=False,context=new_context))
        self.save('video_context.json',new_context)
        self.save('dense_tracking/bundle.json',self.bundle)
        self.save('local_entities.json',dict(shots=[],failures=[]))
        catalog=dict(entities=[],mapping={},external_context=old_context)
        self.save('global_entities.json',catalog)
        with self.invocation('--stage','export'), patch.object(
                pipeline,'StoryBackend',return_value=SimpleNamespace(model=None)), patch.object(
                pipeline,'export_story') as export:
            with self.assertRaisesRegex(ValueError,'Global catalog used different context'):
                pipeline.main()
        export.assert_not_called()
        self.assertEqual(self.read('global_entities.json'),catalog)
        self.assertEqual(self.read('video_context.json'),new_context)
        self.assertEqual(self.read('status.json')['status'],'failed')
        self.assertFalse((self.root/'outputs/latest_run.json').exists())

    def test_unrecognized_custom_model_needs_an_explicit_revision_before_work(self):
        for flag in ('--model-id','--global-model-id'):
            with self.subTest(flag=flag), self.invocation('--stage','tracking',flag,'example/new-model'), patch.object(
                    pipeline,'build_dense_tracks') as track:
                with self.assertRaises(SystemExit) as error:
                    pipeline.main()
                self.assertEqual(error.exception.code,2)
                track.assert_not_called()
                self.assertFalse((self.output/'run_config.json').exists())

    def test_export_cannot_relabel_saved_artifacts_with_a_different_model(self):
        saved=dict(input=str(self.input),mask_refinement=False,
            settings=dict(model_id='example/local',model_revision='old-local'),
            global_model_id='example/global',global_model_revision='old-global')
        self.save('run_config.json',saved)
        for flag,revision in (('--model-id','--model-revision'),
                              ('--global-model-id','--global-model-revision')):
            with self.subTest(flag=flag), self.invocation('--stage','export',flag,'example/new',revision,'new-revision'), patch.object(pipeline,'export_story') as export:
                with self.assertRaises(SystemExit):pipeline.main()
                export.assert_not_called()
                self.assertEqual(self.read('run_config.json'),saved)

    def test_catalog_model_provenance_survives_a_later_config_change(self):
        self.save('run_config.json',dict(input=str(self.input),mask_refinement=False,
            global_model_id='example/new',global_model_revision='new-revision'))
        self.save('dense_tracking/bundle.json',self.bundle)
        self.save('local_entities.json',self.local)
        self.save('global_entities.json',self.catalog(
            review_model=dict(model_id='example/actual',model_revision='actual-revision')))
        with self.invocation('--stage','export'), patch.object(
                pipeline,'StoryBackend',return_value=SimpleNamespace(model=None)), patch.object(pipeline,'export_story') as export:
            with self.assertRaisesRegex(ValueError,'Global catalog was generated by a different model'):
                pipeline.main()
            export.assert_not_called()

    def test_new_dense_tracking_cannot_hide_stale_derived_source_pixels(self):
        from shot_tracking import discover_shots
        from test_input_media import write_video
        source=self.input/'clip.mp4'
        write_video(source,frames=2)
        old=dict(fingerprint='old',stats=dict(dense_frames=2,tracks=0),shots=discover_shots(self.input))
        for shot in old['shots']:shot['tracks']=[]
        write_video(source,frames=2,offset=70)
        fresh=dict(fingerprint='new',stats=dict(dense_frames=2,tracks=0),shots=discover_shots(self.input))
        for shot in fresh['shots']:shot['tracks']=[]
        self.assertNotEqual(old['shots'][0]['source_sha256'],fresh['shots'][0]['source_sha256'])
        # This models a successful tracking-only restart after replacing the
        # video. Each later stage must validate the bundle it actually selects.
        derived=('mask_track_graph/bundle.json','reviewed_tracking.json','mask_refinement/bundle.json')
        for stale_path in derived:
            with self.subTest(stale_path=stale_path):
                self.save('run_config.json',dict(input=str(self.input),mask_refinement=True))
                self.save('dense_tracking/bundle.json',fresh)
                for path in derived:self.save(path,old if path==stale_path else fresh)
                self.save('local_entities.json',dict(shots=[],failures=[]))
                self.save('mask_refinement/local.json',dict(shots=[],failures=[]))
                latest=self.root/'outputs/latest_run.json';latest.parent.mkdir(exist_ok=True)
                latest.write_text(json.dumps(dict(output='previous_completed_run')))
                with self.invocation('--stage','export'), patch.object(
                        pipeline,'StoryBackend',return_value=SimpleNamespace(model=None)), patch.object(
                        pipeline,'export_story') as export, patch.dict(sys.modules,{
                            'mask_track_graph':SimpleNamespace(build_mask_track_graph=Mock(),encode_masked_representatives=Mock())}):
                    with self.assertRaisesRegex(ValueError,'source content/metadata changed'):
                        pipeline.main()
                export.assert_not_called()
                self.assertEqual(self.read('status.json')['status'],'failed')
                self.assertEqual(json.loads(latest.read_text())['output'],'previous_completed_run')

    def test_new_detector_or_association_cannot_reuse_old_reviewed_tracks(self):
        old=dict(self.bundle,version=1,detector_cache_key='old-detector',
                 config=dict(track_high_thresh=.5),tracker='BoT-SORT',
                 tracker_class_policy='class-agnostic',appearance_model='example/appearance')
        for change in ('detector_cache_key','config'):
            with self.subTest(change=change):
                fresh=deepcopy(old)
                fresh[change]='new-detector' if change=='detector_cache_key' else dict(track_high_thresh=.6)
                self.save('run_config.json',dict(input=str(self.input),mask_refinement=False))
                self.save('dense_tracking/bundle.json',fresh)
                self.save('reviewed_tracking.json',old)
                self.save('local_entities.json',self.local)
                self.save('global_entities.json',dict(entities=[],mapping={},external_context={},
                    global_input_fingerprint=pipeline.global_input_fingerprint(old,self.local)))
                with self.invocation('--stage','export'), patch.object(
                        pipeline,'StoryBackend',return_value=SimpleNamespace(model=None)), patch.object(pipeline,'export_story') as export:
                    with self.assertRaisesRegex(ValueError,'Derived tracking uses different detector/association settings'):
                        pipeline.main()
                export.assert_not_called()

    def test_mask_refinement_may_change_tracker_without_changing_dense_lineage(self):
        dense=dict(self.bundle,version=1,detector_cache_key='same-detector',config=dict(track_high_thresh=.5),
                   tracker='BoT-SORT',tracker_class_policy='class-agnostic',appearance_model='example/appearance')
        refined=dict(dense,fingerprint='refined',tracker='Local Entity consolidation + SAM2.1')
        self.save('run_config.json',dict(input=str(self.input),mask_refinement=True))
        self.save('dense_tracking/bundle.json',dense)
        self.save('mask_track_graph/bundle.json',dict(dense,tracker='SAM2.1 raw track masks'))
        self.save('reviewed_tracking.json',dense)
        self.save('local_entities.json',self.local)
        self.save('mask_refinement/bundle.json',refined)
        self.save('mask_refinement/local.json',self.local)
        self.save('global_entities.json',dict(entities=[],mapping={},external_context={},
            global_input_fingerprint=pipeline.global_input_fingerprint(refined,self.local)))
        validation=dict(simultaneous_identity_conflicts=0,event_reference_errors=[],
            coverage=dict(unreviewed_tracks=0),structurally_complete=True,
            entities_without_observations=[],empty_summaries=[],missing_representatives=[])
        with self.invocation('--stage','export'), patch.object(
                pipeline,'StoryBackend',return_value=SimpleNamespace(model=None,ledger={'generations':0})), patch.object(
                pipeline,'export_story',return_value=dict(validation=validation)) as export, patch('story_report.write_report'), patch.dict(sys.modules,{
                    'mask_track_graph':SimpleNamespace(build_mask_track_graph=Mock(),encode_masked_representatives=Mock())}):
            pipeline.main()
        self.assertEqual(export.call_args.args[0]['tracker'],refined['tracker'])
        self.assertEqual(self.read('status.json')['status'],'complete')

    def test_mask_mode_changes_require_all_before_overwriting_saved_settings(self):
        for previous_mode,flag in ((False,'--mask-refinement'),(True,'--skip-mask-refinement')):
            for stage in ('tracking','geometry','local','mask','global','narrative','export'):
                with self.subTest(previous_mode=previous_mode,stage=stage):
                    saved=dict(input=str(self.input),mask_refinement=previous_mode)
                    self.save('run_config.json',saved)
                    with self.invocation('--stage',stage,flag), patch.object(pipeline,'build_dense_tracks') as track:
                        with self.assertRaises(SystemExit) as error:
                            pipeline.main()
                    self.assertEqual(error.exception.code,2)
                    track.assert_not_called()
                    self.assertEqual(self.read('run_config.json'),saved)

    def test_full_pipeline_allows_explicit_mask_mode_change(self):
        for previous_mode,flag in ((False,'--mask-refinement'),(True,'--skip-mask-refinement')):
            with self.subTest(previous_mode=previous_mode):
                self.save('run_config.json',dict(input=str(self.input),mask_refinement=previous_mode))
                # Stop at the first regenerating stage: the configuration guard
                # must allow the full rebuild without needing a detector or GPU.
                with self.invocation('--stage','all',flag), patch.object(
                        pipeline,'build_dense_tracks',side_effect=RuntimeError('rebuild reached')) as track:
                    with self.assertRaisesRegex(RuntimeError,'rebuild reached'):
                        pipeline.main()
                track.assert_called_once()
                self.assertEqual(self.read('run_config.json')['mask_refinement'],not previous_mode)

    def test_later_stages_reject_missing_or_stale_global_input_provenance(self):
        original_bundle=dict(self.bundle,fingerprint='original_tracking')
        original_local=dict(shots=[dict(shot_id='shot_a',summary='A person walks',
            entities=[dict(local_id='shot_a:L01',appearance='red coat',track_ids=['T01'])])],failures=[])
        for stage in ('narrative','export'):
            for change in ('local_appearance','local_summary','bundle','legacy_catalog'):
                with self.subTest(stage=stage,change=change):
                    bundle=deepcopy(original_bundle);local=deepcopy(original_local)
                    catalog=dict(entities=[],mapping={},external_context={},
                        global_input_fingerprint=pipeline.global_input_fingerprint(bundle,local))
                    if change=='local_appearance':local['shots'][0]['entities'][0]['appearance']='blue coat'
                    elif change=='local_summary':local['shots'][0]['summary']='A person runs'
                    elif change=='bundle':bundle['fingerprint']='repaired_tracking'
                    else:catalog.pop('global_input_fingerprint')
                    self.save('run_config.json',dict(input=str(self.input),mask_refinement=False))
                    self.save('dense_tracking/bundle.json',bundle)
                    self.save('local_entities.json',local)
                    self.save('global_entities.json',catalog)
                    with self.invocation('--stage',stage), patch.object(
                            pipeline,'StoryBackend',return_value=SimpleNamespace(model=None)), patch.object(
                            pipeline,'export_story') as export, patch.object(pipeline,'revise_narrative') as narrative:
                        with self.assertRaisesRegex(ValueError,'Global catalog input provenance is missing or changed'):
                            pipeline.main()
                    export.assert_not_called();narrative.assert_not_called()
                    self.assertEqual(self.read('global_entities.json'),catalog)
                    self.assertEqual(self.read('status.json')['status'],'failed')
                    self.assertFalse((self.root/'outputs/latest_run.json').exists())

    def test_structural_export_failures_leave_quality_and_do_not_promote_latest(self):
        for failed_field in ('entities_without_observations','empty_summaries'):
            with self.subTest(failed_field=failed_field):
                self.save('run_config.json',dict(input=str(self.input),mask_refinement=False))
                self.save('dense_tracking/bundle.json',self.bundle)
                self.save('local_entities.json',dict(shots=[],failures=[]))
                self.save('global_entities.json',self.catalog())
                latest=self.root/'outputs/latest_run.json'
                latest.parent.mkdir(exist_ok=True)
                previous=dict(output='previous_completed_run')
                latest.write_text(json.dumps(previous))
                validation=dict(simultaneous_identity_conflicts=0,event_reference_errors=[],
                    coverage=dict(unreviewed_tracks=0),structurally_complete=False,
                    entities_without_observations=[],empty_summaries=[],missing_representatives=[])
                validation[failed_field]=['unusable_result']
                with self.invocation('--stage','export'), patch.object(
                        pipeline,'StoryBackend',return_value=SimpleNamespace(model=None,ledger={'generations':0})), patch.object(
                        pipeline,'export_story',return_value=dict(validation=validation)):
                    with self.assertRaisesRegex(ValueError,'Export structure validation failed'):
                        pipeline.main()
                self.assertEqual(self.read('quality.json')['status'],'failed_structure')
                self.assertEqual(self.read('quality.json')['export']['validation'][failed_field],['unusable_result'])
                self.assertEqual(self.read('status.json')['status'],'failed')
                self.assertEqual(json.loads(latest.read_text()),previous)

    def test_identity_warnings_keep_structurally_complete_result_reviewable(self):
        self.save('run_config.json',dict(input=str(self.input),mask_refinement=False))
        self.save('dense_tracking/bundle.json',self.bundle)
        self.save('local_entities.json',dict(shots=[],failures=[]))
        self.save('global_entities.json',self.catalog(
            identity_reconciliation=dict(status='needs_review',unresolved=[{}])))
        validation=dict(simultaneous_identity_conflicts=1,event_reference_errors=[],
            coverage=dict(unreviewed_tracks=0),structurally_complete=True,
            entities_without_observations=[],empty_summaries=[],missing_representatives=[])
        with self.invocation('--stage','export'), patch.object(
                pipeline,'StoryBackend',return_value=SimpleNamespace(model=None,ledger={'generations':0})), patch.object(
                pipeline,'export_story',return_value=dict(validation=validation)), patch('story_report.write_report'):
            pipeline.main()
        self.assertEqual(self.read('status.json')['status'],'complete_with_warnings')
        self.assertEqual(self.read('quality.json')['status'],'needs_review')
        self.assertEqual(json.loads((self.root/'outputs/latest_run.json').read_text())['output'],str(self.output))


if __name__=='__main__':
    unittest.main()


class SampledExecutionTests(unittest.TestCase):
    setUp=ResumeTests.setUp
    save=ResumeTests.save
    read=ResumeTests.read
    invocation=ResumeTests.invocation
    """Use existing isolated pipeline harness, overriding legacy-mode argv."""
    def test_sampled_tracking_records_progress_and_time_on_failure(self):
        with self.invocation('--stage','tracking','--analysis-frames','sampled'), patch.object(
                pipeline,'build_dense_tracks',side_effect=RuntimeError('test detector failure')):
            with self.assertRaisesRegex(RuntimeError,'test detector failure'):pipeline.main()
        self.assertEqual(self.read('run_config.json')['analysis_frames'],'sampled')
        self.assertFalse(self.read('run_config.json')['mask_refinement'])
        self.assertEqual(self.read('run_timing.json')['status'],'failed')
        self.assertGreaterEqual(self.read('run_timing.json')['elapsed_sec'],0)
        self.assertEqual(self.read('progress.json')['completed'],0)
        self.assertIn('04 전체 시간',(self.output/'timing_summary.txt').read_text())

    def test_sampled_cannot_silently_resume_dense_results(self):
        self.save('run_config.json',dict(input=str(self.input),mask_refinement=False))
        with self.invocation('--stage','tracking','--analysis-frames','sampled'):
            with self.assertRaises(SystemExit):pipeline.main()

    def test_sampled_rejects_dense_sam_option(self):
        with self.invocation('--analysis-frames','sampled','--mask-refinement'):
            with self.assertRaises(SystemExit):pipeline.main()
