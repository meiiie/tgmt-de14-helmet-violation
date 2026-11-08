"""Authenticated, bounded single-worker API for the shared NVIDIA job executor."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import secrets
import shutil
import re
import os
import threading
import time
import uuid

from fastapi import FastAPI,Request,HTTPException,Depends
from fastapi.responses import FileResponse,JSONResponse
from starlette.concurrency import run_in_threadpool
from .remote_jobs import read_request,PROTOCOL


class DownloadResponse(FileResponse):
    def __init__(self,*args,release,**kwargs):
        super().__init__(*args,**kwargs);self.release=release

    async def __call__(self,scope,receive,send):
        try:return await super().__call__(scope,receive,send)
        finally:self.release()


def lock_storage(root):
    """Kernel-owned lock survives process crashes without a stale-lock timeout."""
    stream=(root/'.gpu_api.lock').open('a+b')
    if stream.tell()==0:stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError('GPU job storage is already owned by another API process') from error
    return stream


def unlock_storage(stream):
    try:
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    finally:stream.close()


def probe_gpu():
    import torch
    if not torch.cuda.is_available():return {'cuda_available':False,'gpu':None}
    import onnxruntime as ort
    return {'cuda_available':True,'gpu':torch.cuda.get_device_name(0),
            'head_providers':ort.get_available_providers(),'torch_cuda':torch.version.cuda}


def create_app(root,token,*,probe=None,runner=None,max_pending=2,max_upload_bytes=512*1024**2,
               max_finished=3,result_ttl=3600):
    if not token or len(token)<16 or not token.isascii():raise ValueError('Set an ASCII GPU API token of at least 16 characters')
    if min(max_pending,max_upload_bytes,max_finished,result_ttl)<1:raise ValueError('Invalid API limits')
    hardware=(probe or probe_gpu)()
    if not hardware.get('cuda_available'):raise RuntimeError('CUDA required: this backend must run on NVIDIA GPU, not local Intel CPU')
    if probe is None and 'CUDAExecutionProvider' not in hardware.get('head_providers',[]):
        raise RuntimeError('CUDAExecutionProvider required; use the working GPU ONNX Runtime environment')
    if runner is None:
        from scripts.run_gpu_job import execute_job
        runner=execute_job
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='helmet-gpu')
    jobs={};lock=threading.Lock();slots=threading.BoundedSemaphore(max_pending)
    @asynccontextmanager
    async def lifespan(app):
        # The launcher runs one process/worker with a dedicated storage root.
        # A crash loses the in-memory registry; only our marked UUID folders
        # can be discarded on the next start, never arbitrary root contents.
        lease=None
        try:
            lease=lock_storage(root)
            for folder in root.iterdir():
                if (re.fullmatch('[0-9a-f]{32}',folder.name) and not folder.is_symlink()
                        and folder.is_dir() and (folder/'.helmet_gpu_job').is_file()
                        and (folder/'.helmet_gpu_job').read_bytes()==b'de14-gpu-api-v1'):
                    await run_in_threadpool(discard,folder)
            yield
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
            # Results are transient delivery files, not a persistent archive.
            try:
                for job in jobs.values():
                    if job['folder'].exists():discard(job['folder'])
            finally:
                if lease is not None:unlock_storage(lease)
    app=FastAPI(title='Helmet GPU API',lifespan=lifespan)

    def auth(request:Request):
        value=request.headers.get('authorization','')
        if not value.startswith('Bearer ') or not secrets.compare_digest(value[7:].encode('utf8'),token.encode('ascii')):
            raise HTTPException(401,'Invalid GPU API token')

    def pending():return sum(j['status'] in {'queued','running'} for j in jobs.values())

    def prune():
        with lock:
            finished=sorted((j['finished'],key) for key,j in jobs.items() if j['status'] in {'completed','error'})
            for index,(ended,key) in enumerate(finished):
                j=jobs[key]
                if j.get('downloads',0):continue
                if index<len(finished)-max_finished or time.time()-ended>result_ttl:
                    if j['folder'].exists():discard(j['folder'])
                    del jobs[key]

    def run(job_id):
        with lock:j=jobs[job_id];j['status']='running'
        try:
            result=Path(runner(j['request'],j['folder']/'execution')).resolve()
            if not result.is_file() or not result.is_relative_to(j['folder']):raise ValueError('Invalid executor result path')
            saved=j['folder']/'result.zip';result.replace(saved)
            discard_execution(j)
            with lock:j.update(status='completed',result=saved,finished=time.time())
        except Exception as error:
            with lock:j.update(status='error',finished=time.time(),error=(type(error).__name__+': '+str(error)).replace(token,'[redacted]')[:1500])
            discard_execution(j)
        finally:slots.release()

    def discard(folder):
        # Only our randomly created direct child may be recursively removed.
        if folder.resolve().parent != root:raise ValueError('Unsafe job cleanup path')
        shutil.rmtree(folder)

    def discard_execution(job):
        job['request'].unlink(missing_ok=True)
        execution=(job['folder']/'execution').resolve()
        if execution.parent != job['folder']:raise ValueError('Unsafe executor cleanup path')
        if execution.exists():shutil.rmtree(execution)

    @app.get('/health',dependencies=[Depends(auth)])
    def health():
        prune()
        with lock:count=pending()
        return {'service':'de14-gpu-api','protocol':PROTOCOL,**hardware,'pending_jobs':count,'max_pending':max_pending}

    @app.post('/jobs',dependencies=[Depends(auth)])
    async def submit(request:Request):
        await run_in_threadpool(prune)
        if not slots.acquire(blocking=False):raise HTTPException(429,'GPU queue full; try again after the current job')
        folder=root/uuid.uuid4().hex;accepted=False
        try:
            folder.mkdir();(folder/'.helmet_gpu_job').write_bytes(b'de14-gpu-api-v1')
            path=folder/'request.zip';size=0
            with path.open('xb') as stream:
                async for chunk in request.stream():
                    size+=len(chunk)
                    if size>max_upload_bytes:raise HTTPException(413,'Media request exceeds upload limit')
                    stream.write(chunk)
            manifest,_=await run_in_threadpool(read_request,path,folder/'validated')
            # Validation extracts once; the executor later owns its own copy.
            validated=(folder/'validated').resolve()
            if validated.parent != folder:raise ValueError('Unsafe validation path')
            await run_in_threadpool(shutil.rmtree,validated)
            job_id=manifest['job_id']
            with lock:
                if job_id in jobs:raise HTTPException(409,'Job ID already submitted')
                jobs[job_id]={'status':'queued','request':path,'folder':folder}
                pool.submit(run,job_id)
                accepted=True
        except HTTPException:raise
        except Exception as error:raise HTTPException(400,'Invalid job archive: '+str(error)[:300]) from error
        finally:
            if not accepted:
                slots.release()
                if folder.exists():await run_in_threadpool(discard,folder)
        return JSONResponse({'job_id':job_id,'status':'queued'},status_code=202)

    @app.get('/jobs/{job_id}',dependencies=[Depends(auth)])
    def status(job_id:str):
        with lock:
            if job_id not in jobs:raise HTTPException(404,'Unknown job')
            j=jobs[job_id];return {'job_id':job_id,'status':j['status'],'error':j.get('error')}

    @app.get('/jobs/{job_id}/result',dependencies=[Depends(auth)])
    def result(job_id:str):
        with lock:
            if job_id not in jobs:raise HTTPException(404,'Unknown job')
            j=jobs[job_id]
            if j['status']!='completed':raise HTTPException(409,'Result is not ready')
            path=j['result'];j['downloads']=j.get('downloads',0)+1
        def downloaded():
            with lock:j['downloads']-=1
        return DownloadResponse(path,media_type='application/zip',filename=f'helmet_gpu_result_{job_id}.zip',release=downloaded)
    return app
