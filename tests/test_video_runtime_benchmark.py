"""Check experiment comparability and real media export using model fixtures."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]

class VideoRuntimeBenchmarkTest(unittest.TestCase):
    def test_speedup_requires_identical_experiment_not_identical_device(self):
        from benchmark_video_runtime import compare_reports
        cpu={'experiment':{'source_sha256':'video','weights':{'head':'a'},'stride':3,'encoder':{'state':'b'}},
             'profile':'openvino_cpu','timing':{'processing_seconds':100,'video_wall_seconds':110}}
        gpu=copy.deepcopy(cpu);gpu.update(profile='cloud_cuda',timing={'processing_seconds':20,'video_wall_seconds':30})
        comparison=compare_reports(cpu,gpu)
        self.assertEqual(comparison['processing_speedup'],5)
        gpu['experiment']['stride']=10
        with self.assertRaisesRegex(ValueError,'experiment'):compare_reports(cpu,gpu)

    def test_shared_video_runner_preserves_full_source_and_unknown_not_accuracy(self):
        import benchmark_video_runtime as b
        from de14.pipeline import Pipeline
        from de14.runtime import load_profile
        config=load_profile('openvino_cpu',accurate=False,detect_heads=True)
        class Runtime:
            device='cpu'
            classifier=None
            detector=SimpleNamespace(hd=SimpleNamespace(backend='openvino',compiled=SimpleNamespace(get_property=lambda _:['CPU'])))
            def metadata(self):return {'torch_device':'cpu','config':{},'head_providers':['OpenVINO:CPU']}
            def pipeline(self,**kw):
                return Pipeline(lambda _: {'motorcycle':[[5,20,60,63]],'motorcycle_score':[.9],
                                           'person':[],'head':[],'head_detection':True},lambda _:[],**kw)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);video=root/'input.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,64))
            for _ in range(12):writer.write(np.zeros((64,64,3),np.uint8))
            writer.release()
            with patch.object(b,'build_runtime',return_value=Runtime()),patch.object(b,'experiment_identity',return_value={'fixture':True,'filter':None}):
                report=b.benchmark(video,'openvino_cpu',root/'out',stride=3,n_min=1)
            self.assertEqual(report['media']['output']['frame_count'],12)
            self.assertEqual(report['media']['output']['fps'],30)
            self.assertEqual(report['sampled_frames'],4)
            self.assertIsNone(report['precision']);self.assertIsNone(report['recall'])
            self.assertIsNone(report['vram']['process_peak_mib'])
            self.assertTrue((root/'out/report.json').is_file())

    def test_cuda_check_disables_runtime_retry_and_reads_live_provider(self):
        from benchmark_video_runtime import validate_runtime
        from unittest.mock import Mock
        session=Mock();session.get_providers.return_value=['CUDAExecutionProvider','CPUExecutionProvider']
        runtime=SimpleNamespace(device='cuda',detector=SimpleNamespace(hd=SimpleNamespace(backend='onnx',s=session)))
        self.assertEqual(validate_runtime(runtime,'cloud_cuda',disable_fallback=True)[0],'CUDAExecutionProvider')
        session.disable_fallback.assert_called_once()
        session.get_providers.return_value=['CPUExecutionProvider']
        with self.assertRaisesRegex(RuntimeError,'CUDA'):validate_runtime(runtime,'cloud_cuda')

    def test_hard_cases_export_review_without_inventing_helmet_gt(self):
        from benchmark_video_runtime import hard_cases
        from benchmark_video_sampling import sha256
        from de14.pipeline import Pipeline
        class Runtime:
            def pipeline(self,**kw):
                return Pipeline(lambda _: {'motorcycle':[[5,20,60,63]],'motorcycle_score':[.9],
                                          'person':[],'head':[[20,1,50,32]],'head_detection':True},lambda cs:[.9]*len(cs),**kw)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=root/'image.png';cv2.imwrite(str(image),np.zeros((64,64,3),np.uint8))
            manifest=root/'cases.json';manifest.write_text(json.dumps({'images':[{'id':'fixture','image':str(image),
                'image_sha256':sha256(image),'gt_heads':[{'box':[20,1,50,32]}]}]}))
            report=hard_cases(Runtime(),manifest,root/'out')
            self.assertEqual(report['head_box_metrics']['recall'],1)
            self.assertIsNone(report['helmet_precision']);self.assertIsNone(report['helmet_recall'])
            self.assertTrue((root/'out/fixture_annotated.jpg').is_file())
            import csv
            with (root/'out/helmet_review.csv').open(encoding='utf-8-sig') as file:rows=list(csv.DictReader(file))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['human_label'],'')
            self.assertEqual(rows[0]['hat_type'],'');self.assertEqual(rows[0]['reviewed'],'False')

if __name__=='__main__':unittest.main()
