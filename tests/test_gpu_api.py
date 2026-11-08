"""Real HTTP protocol; fake GPU executor boundary only. No GPU performance claim."""
import sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
import unittest
from contextlib import contextmanager
import socket,threading

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

def job_folders(root):return [p for p in root.iterdir() if p.is_dir()]

@contextmanager
def serve(app):
    """Use an actual TCP socket, not mocked requests transport."""
    import uvicorn
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error',lifespan='on'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    try:
        deadline=time.monotonic()+10
        while not server.started:
            if not thread.is_alive() or time.monotonic()>deadline:raise RuntimeError('Fixture server failed startup')
            time.sleep(.01)
        yield f'http://127.0.0.1:{port}'
    finally:
        server.should_exit=True;thread.join(10);sock.close()


class GpuApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.token='a-secure-test-token-only'

    def test_production_backend_refuses_cpu(self):
        from de14.gpu_api import create_app
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):create_app(self.root,self.token)

    def test_authenticated_submit_poll_result_and_source_identity(self):
        from fastapi.testclient import TestClient
        from de14.gpu_api import create_app
        from de14.remote_jobs import create_request,create_result,JobOptions,read_request,read_result
        media=self.root/'scene.png';media.write_bytes(b'input fixture')
        archive=self.root/'request.zip';manifest=create_request(media,JobOptions(kind='image'),archive)
        def execute(path,out):
            m,p=read_request(path,Path(out)/'in');self.assertEqual(p.read_bytes(),b'input fixture')
            result=Path(out)/'result.zip';create_result(m['job_id'],m['input']['sha256'],{'runtime':{'torch_device':'cuda'}},{'image.png':p},result)
            return result
        app=create_app(self.root/'jobs',self.token,probe=lambda:{'cuda_available':True,'gpu':'fixture, not real GPU'},runner=execute)
        with TestClient(app) as client:
            self.assertEqual(client.get('/health').status_code,401)
            h={'Authorization':'Bearer '+self.token,'Content-Type':'application/zip'}
            self.assertTrue(client.get('/health',headers=h).json()['cuda_available'])
            r=client.post('/jobs',headers=h,content=archive.read_bytes());self.assertEqual(r.status_code,202)
            job=r.json()['job_id'];self.assertEqual(job,manifest['job_id'])
            for _ in range(100):
                status=client.get('/jobs/'+job,headers=h).json()
                if status['status'] in {'completed','error'}:break
                time.sleep(.01)
            self.assertEqual(status['status'],'completed',status)
            result=self.root/'received.zip';result.write_bytes(client.get('/jobs/'+job+'/result',headers=h).content)
            m,_=read_result(result,self.root/'received',job,manifest['input']['sha256'])
            self.assertEqual(m['result']['runtime']['torch_device'],'cuda')

    def test_environment_secret_is_not_forwarded_to_overridden_server(self):
        import app
        with patch.dict(app.os.environ,{'HELMET_GPU_API_URL':'https://trusted.example','HELMET_GPU_API_TOKEN':'private-env-token'}):
            client=app._gpu_client('https://other.example','')
            self.assertEqual(client.token,'')
            client=app._gpu_client('','')
            self.assertEqual(client.token,'private-env-token')

    def test_queue_is_bounded_and_rejected_uploads_are_removed(self):
        from fastapi.testclient import TestClient
        from de14.gpu_api import create_app
        from de14.remote_jobs import create_request,JobOptions
        entered=threading.Event();release=threading.Event()
        def execute(path,out):
            entered.set();release.wait(5);raise ValueError('fixture GPU failure')
        root=self.root/'queue'
        app=create_app(root,self.token,probe=lambda:{'cuda_available':True},runner=execute,max_pending=1,max_upload_bytes=20000)
        h={'Authorization':'Bearer '+self.token}
        media=self.root/'img.png';media.write_bytes(b'input');archive=self.root/'request.zip'
        create_request(media,JobOptions(kind='image'),archive)
        with TestClient(app) as client:
            self.assertEqual(client.post('/jobs',headers=h,content=b'invalid').status_code,400)
            self.assertEqual(job_folders(root),[])
            self.assertEqual(client.post('/jobs',headers=h,content=b'x'*20001).status_code,413)
            self.assertEqual(job_folders(root),[])
            r=client.post('/jobs',headers=h,content=archive.read_bytes());self.assertEqual(r.status_code,202)
            self.assertTrue(entered.wait(2))
            try:
                self.assertEqual(client.post('/jobs',headers=h,content=archive.read_bytes()).status_code,429)
                self.assertEqual(len(job_folders(root)),1)
            finally:release.set()
            job=r.json()['job_id']
            for _ in range(100):
                if client.get('/jobs/'+job,headers=h).json()['status']=='error':break
                time.sleep(.01)
            self.assertEqual(client.get('/health',headers=h).json()['pending_jobs'],0)
            self.assertEqual(client.get('/jobs/'+job+'/result',headers=h).status_code,409)
            self.assertEqual(client.post('/jobs',headers=h,content=archive.read_bytes()).status_code,409)
            self.assertEqual(len(job_folders(root)),1)

    def test_web_image_and_video_roundtrip_actual_http_shared_pipeline(self):
        import app as web
        import cv2,numpy as np
        from PIL import Image
        from de14.gpu_api import create_app
        from scripts.run_gpu_job import execute_job
        from de14.pipeline import Pipeline
        from de14.video_io import probe_video
        def runtime(config):
            class Fixture:
                device='cuda'
                def metadata(self):return {'torch_device':'cuda','cuda_device':'HTTP test fixture, not real GPU','head_providers':['CUDAExecutionProvider']}
                def pipeline(self,**kw):
                    def detect(_):return {'motorcycle':[[20,70,180,190]],'motorcycle_score':[.9],'person':[],
                                          'head':[[40,10,70,40]],'head_detection':True}
                    return Pipeline(detect,lambda crops:[.9]*len(crops),**kw)
            return Fixture()
        api=create_app(self.root/'http',self.token,probe=lambda:{'cuda_available':True,'gpu':'HTTP fixture'},runner=execute_job)
        video=self.root/'source.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(200,200))
        for _ in range(12):writer.write(np.zeros((200,200,3),np.uint8))
        writer.release()
        with serve(api) as url, patch('torch.cuda.is_available',return_value=True),patch('de14.runtime.build_runtime',side_effect=runtime),patch.object(web,'get_runtime',side_effect=AssertionError('Web must not load local model')):
            self.assertIn('GPU đã kết nối',web.check_gpu_connection(url,self.token))
            self.assertIn('chưa kết nối',web.check_gpu_connection(url,'wrong-token'))
            args=(web.BACKENDS[1],3,0,1,1280,True,False,'openvino_cpu','full','gpu_api',url,self.token)
            image_result=web.analyze_web(Image.new('RGB',(200,200)),None,*args)
            self.assertTrue(Path(image_result[0]).is_file());self.assertIsNone(image_result[1])
            self.assertEqual(len(image_result[3]),1);self.assertEqual(image_result[3][0][5],1)
            self.assertTrue(Path(image_result[5]).is_file())
            video_result=web.analyze_web(None,str(video),*args)
            self.assertIsNone(video_result[0]);self.assertTrue(Path(video_result[1]).is_file())
            info=probe_video(video_result[1]);self.assertEqual(info['frame_count'],12);self.assertEqual(info['fps'],30)
            self.assertEqual((info['width'],info['height']),(200,200))

    def test_missing_endpoint_does_not_fall_back_to_local(self):
        import app
        from PIL import Image
        with patch.dict(app.os.environ,{'HELMET_GPU_API_URL':''}),patch.object(app,'get_runtime') as factory:
            with self.assertRaisesRegex(app.gr.Error,'GPU API'):
                app.analyze_web(Image.new('RGB',(20,20)),None,app.BACKENDS[1],3,0,3,1280)
            factory.assert_not_called()

    def test_share_uses_only_the_configured_gpu_endpoint(self):
        import app
        with patch.object(app, 'PUBLIC_SHARE', True), patch.dict(app.os.environ, {
            'HELMET_GPU_API_URL': 'https://gpu.example.com',
            'HELMET_GPU_API_TOKEN': 'fixture',
        }):
            self.assertEqual(app._gpu_client().url, 'https://gpu.example.com')
            with self.assertRaisesRegex(ValueError, 'máy chủ cấu hình'):
                app._gpu_client('http://127.0.0.1:9000')

    def test_shared_realtime_does_not_open_a_server_path(self):
        import app
        with patch.object(app, 'PUBLIC_SHARE', True), patch.object(app.cv2, 'VideoCapture') as capture:
            stream = app.rt_stream('rtsp://internal-camera', None, 'yolo26s', 640, 5, True)
            with self.assertRaisesRegex(app.gr.Error, 'tải lên'):
                next(stream)
            capture.assert_not_called()

    def test_retention_compacts_jobs_and_bad_range_cannot_pin_results(self):
        from fastapi.testclient import TestClient
        from de14.gpu_api import create_app
        from de14.remote_jobs import create_request,create_result,read_request,JobOptions
        def execute(path,out):
            m,p=read_request(path,Path(out)/'in');result=Path(out)/'result.zip'
            create_result(m['job_id'],m['input']['sha256'],{},{'image.png':p},result)
            return result
        root=self.root/'retention'
        api=create_app(root,self.token,probe=lambda:{'cuda_available':True},runner=execute,max_finished=1)
        h={'Authorization':'Bearer '+self.token}
        with TestClient(api) as client:
            for index in range(2):
                media=self.root/f'{index}.png';media.write_bytes(b'fixture')
                archive=self.root/f'{index}.zip';m=create_request(media,JobOptions(kind='image'),archive)
                self.assertEqual(client.post('/jobs',headers=h,content=archive.read_bytes()).status_code,202)
                job=m['job_id']
                for _ in range(100):
                    if client.get('/jobs/'+job,headers=h).json()['status']=='completed':break
                    time.sleep(.01)
                self.assertEqual(client.get('/jobs/'+job+'/result',headers={**h,'Range':'bytes=999999999999-'}).status_code,416)
                client.get('/health',headers=h)
                self.assertEqual(len(job_folders(root)),1)
                self.assertEqual(sorted(p.name for p in job_folders(root)[0].iterdir()),['.helmet_gpu_job','result.zip'])
                if index==0:first_job=job
            self.assertEqual(client.get('/jobs/'+first_job,headers=h).status_code,404)
        self.assertEqual(job_folders(root),[])

    def test_startup_cleans_only_marked_orphan_job_folders(self):
        from fastapi.testclient import TestClient
        from de14.gpu_api import create_app
        root=self.root/'restarted';root.mkdir()
        old=root/('a'*32);old.mkdir();(old/'.helmet_gpu_job').write_bytes(b'de14-gpu-api-v1')
        (old/'leftover_video.mp4').write_bytes(b'interrupted job')
        unrelated=root/('b'*32);unrelated.mkdir();(unrelated/'keep.txt').write_text('user file')
        api=create_app(root,self.token,probe=lambda:{'cuda_available':True},runner=lambda *_:None)
        with TestClient(api):
            self.assertFalse(old.exists());self.assertTrue((unrelated/'keep.txt').is_file())

    def test_main_ui_default_and_key_visibility(self):
        import app,json
        config=app.demo.get_config_file();components=config['components']
        tabs={c['props']['label']:c['props'].get('visible',True) for c in components if c['type']=='tabitem'}
        self.assertTrue(tabs['Nhận diện']);self.assertFalse(tabs['GPU Colab (ảnh / video)'])
        self.assertFalse(tabs['Camera / RTSP (thử nghiệm)'])
        radio=next(c for c in components if c['props'].get('label')=='Nơi xử lý')
        configured_url = app.os.environ.get('HELMET_GPU_API_URL', '').strip()
        self.assertEqual(radio['props']['value'], 'gpu_api' if configured_url and not app.kaggle_configured else 'local')
        choices = dict(radio['props']['choices']).values()
        self.assertIn('gpu_api', choices)
        self.assertEqual('kaggle' in choices, app.kaggle_configured)
        settings=next(c for c in components if c['props'].get('label')=='Cài đặt nâng cao')
        self.assertFalse(settings['props']['open'])
        key=next(c for c in components if c['props'].get('label','').startswith('Khóa API'))
        self.assertEqual(key['props']['value'],'');self.assertEqual(key['props']['type'],'password')
        with patch.dict(app.os.environ,{'HELMET_GPU_API_TOKEN':'should-never-be-in-config'}):
            self.assertNotIn('should-never-be-in-config',json.dumps(app.demo.get_config_file(),default=str))

    def test_second_server_cannot_delete_active_storage(self):
        from fastapi.testclient import TestClient
        from de14.gpu_api import create_app
        root=self.root/'exclusive'
        first=create_app(root,self.token,probe=lambda:{'cuda_available':True},runner=lambda *_:None)
        with TestClient(first):
            active=root/('c'*32);active.mkdir();(active/'.helmet_gpu_job').write_bytes(b'de14-gpu-api-v1')
            second=create_app(root,self.token,probe=lambda:{'cuda_available':True},runner=lambda *_:None)
            with self.assertRaisesRegex(RuntimeError,'another API process'):
                with TestClient(second):pass
            self.assertTrue(active.is_dir())

if __name__=='__main__':unittest.main()
