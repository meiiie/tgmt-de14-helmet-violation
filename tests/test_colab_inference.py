"""Transfer runs on CPU fixtures; actual cloud CUDA evidence is supplied separately."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from de14.remote_jobs import JobOptions,create_request,read_request,read_result,create_result


class ColabInferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.image=self.root/'scene.png';cv2.imwrite(str(self.image),np.zeros((200,200,3),np.uint8))
        self.request=self.root/'job.zip';self.manifest=create_request(self.image,JobOptions(kind='image'),self.request)

    def runtime(self,config):
        from de14.pipeline import Pipeline
        def detector(_):return {'motorcycle':[[20,70,180,190]],'motorcycle_score':[.9],'person':[],
                                'head':[] if getattr(self,'headless',False) else [[40,10,70,40]],'head_detection':True}
        class Runtime:
            device='cuda'
            def pipeline(self,**kw):return Pipeline(detector,lambda cs:[.9]*len(cs),**kw)
            def metadata(self):return {'torch_device':'cuda','cuda_device':'test fixture, not actual GPU','head_providers':['CUDAExecutionProvider']}
        self.config=config
        return Runtime()

    def test_missing_cuda_fails_instead_of_using_local_cpu(self):
        from run_gpu_job import execute_job
        with patch('torch.cuda.is_available',return_value=False),patch('de14.runtime.build_runtime') as factory:
            with self.assertRaisesRegex(RuntimeError,'CUDA'):execute_job(self.request,self.root/'out')
            factory.assert_not_called()

    def test_image_job_uses_shared_pipeline_and_returns_verified_gpu_result(self):
        from run_gpu_job import execute_job
        with patch('torch.cuda.is_available',return_value=True),patch('de14.runtime.build_runtime',side_effect=self.runtime):
            result_zip=execute_job(self.request,self.root/'out')
        meta,folder=read_result(result_zip,self.root/'received',self.manifest['job_id'],self.manifest['input']['sha256'])
        self.assertEqual(self.config.device,'cuda');self.assertTrue(self.config.head_enabled);self.assertFalse(self.config.accurate)
        self.assertEqual(meta['result']['runtime']['torch_device'],'cuda')
        self.assertTrue((folder.parent/meta['result']['media_path']).is_file())
        self.assertEqual(meta['result']['counts']['people'],1)

    def test_video_job_keeps_all_source_frames_at_original_fps(self):
        from run_gpu_job import execute_job
        video=self.root/'source.mp4';w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(200,200))
        for _ in range(12):w.write(np.zeros((200,200,3),np.uint8))
        w.release();job=self.root/'video_job.zip'
        manifest=create_request(video,JobOptions(kind='video',stride=3,n_min=1),job)
        with patch('torch.cuda.is_available',return_value=True),patch('de14.runtime.build_runtime',side_effect=self.runtime):
            archive=execute_job(job,self.root/'video_out')
        meta,folder=read_result(archive,self.root/'video_received',manifest['job_id'],manifest['input']['sha256'])
        self.assertEqual(meta['result']['media']['frame_count'],12)
        self.assertEqual(meta['result']['media']['fps'],30)
        self.assertTrue((folder.parent/meta['result']['media_path']).is_file())

    def test_static_image_does_not_apply_temporal_f2_to_unknown_vehicle(self):
        from run_gpu_job import execute_job
        self.headless=True
        with patch('torch.cuda.is_available',return_value=True),patch('de14.runtime.build_runtime',side_effect=self.runtime):
            archive=execute_job(self.request,self.root/'headless_out')
        meta,_=read_result(archive,self.root/'headless_received')
        self.assertEqual(meta['result']['counts']['vehicles'],1)
        self.assertEqual(meta['result']['counts']['unknown'],1)

    def test_local_export_and_import_do_not_load_models(self):
        import app
        with patch.object(app,'get_runtime',side_effect=AssertionError('No local inference')):
            archive,state,_=app.export_colab_job(str(self.image),None,app.BACKENDS[1],3,3,1280,True,False,0,'full')
            request,_=read_request(archive,self.root/'local_request')
            self.assertEqual(request['options']['detector'],'yolo')
            self.assertFalse(request['options']['accurate'])
            payload={'kind':'image','media_path':'artifacts/result.png','tracks_path':None,
                     'runtime':{'torch_device':'cuda','cuda_device':'Tesla T4'},'counts':{'vehicles':1,'people':1,'violations':1},
                     'timings':{'sampled_frames':1,'processing_seconds':1.0}}
            result=self.root/'result.zip';create_result(state['job_id'],state['input_sha256'],payload,{'result.png':self.image},result)
            image,video,summary,csv=app.import_colab_result(str(result),state)
            self.assertTrue(Path(image).is_file());self.assertIsNone(video);self.assertIn('Tesla T4',summary)

    def test_benchmark_archive_is_explained_not_treated_as_inference_result(self):
        import app
        bad=self.root/'benchmark.zip'
        with ZipFile(bad,'w') as z:z.writestr('cloud_cuda/report.json','{}')
        with self.assertRaises(app.gr.Error) as error:app.import_colab_result(str(bad),None)
        self.assertIn('benchmark',str(error.exception).lower())

    def test_code_update_preserves_downloaded_weights_and_updates_runner(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('colab_cell',ROOT/'notebooks/colab_gpu_inference_cell.py')
        cell=importlib.util.module_from_spec(spec);spec.loader.exec_module(cell)
        project=self.root/'project';(project/'weights').mkdir(parents=True)
        (project/'weights/dino_probe.joblib').write_bytes(b'keep trained probe')
        archive=self.root/'code.zip'
        with ZipFile(archive,'w') as z:
            z.writestr('scripts/run_gpu_job.py','updated runner')
            z.writestr('weights/dino_probe.joblib',b'different probe')
        cell.update_project_sources(archive,project)
        self.assertEqual((project/'weights/dino_probe.joblib').read_bytes(),b'keep trained probe')
        self.assertEqual((project/'scripts/run_gpu_job.py').read_text(),'updated runner')

    def test_unsafe_source_zip_rejected_before_updating_any_code(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('colab_cell',ROOT/'notebooks/colab_gpu_inference_cell.py')
        cell=importlib.util.module_from_spec(spec);spec.loader.exec_module(cell)
        project=self.root/'project';(project/'scripts').mkdir(parents=True)
        runner=project/'scripts/run_gpu_job.py';runner.write_text('old runner')
        archive=self.root/'unsafe_code.zip'
        with ZipFile(archive,'w') as z:
            z.writestr('scripts/run_gpu_job.py','new runner');z.writestr('../escape.py','no')
        with self.assertRaises(ValueError):cell.update_project_sources(archive,project)
        self.assertEqual(runner.read_text(),'old runner')

if __name__=='__main__':unittest.main()
