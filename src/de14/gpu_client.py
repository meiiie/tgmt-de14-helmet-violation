"""Streaming web-to-GPU transport. Media/job identity is checked after download."""
from pathlib import Path
import time
from urllib.parse import urlsplit
import requests
from .remote_jobs import PROTOCOL,MAX_BYTES


class GpuApiClient:
    def __init__(self,url,token,timeout=30):
        url=url.strip().rstrip('/');parts=urlsplit(url)
        if parts.scheme not in {'https','http'} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError('GPU API URL must be http(s) with no embedded credentials/query')
        self.url=url;self.token=token;self.timeout=timeout

    def request(self,method,path,**kwargs):
        r=requests.request(method,self.url+path,headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/zip'},
                           timeout=self.timeout,allow_redirects=False,**kwargs)
        if not 200 <= r.status_code < 300:
            try:message=r.json().get('detail',r.reason)
            except ValueError:message=r.reason
            r.close()
            raise RuntimeError(f'GPU API HTTP{r.status_code}: {message}'.replace(self.token,'[redacted]') if self.token else f'GPU API HTTP{r.status_code}: {message}')
        return r

    def health(self):
        with self.request('GET','/health') as r:info=r.json()
        if info.get('service')!='de14-gpu-api' or info.get('protocol')!=PROTOCOL or not info.get('cuda_available'):
            raise RuntimeError('Endpoint is not a ready Helmet CUDA GPU API')
        return info

    def submit(self,archive,expected_job_id):
        with Path(archive).open('rb') as source, self.request('POST','/jobs',data=source) as r:info=r.json()
        if info.get('job_id')!=expected_job_id:raise RuntimeError('GPU API returned a different job ID')

    def wait(self,job_id,timeout=3600,interval=1):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            with self.request('GET','/jobs/'+job_id) as r:info=r.json()
            if info['status']=='completed':return
            if info['status']=='error':raise RuntimeError('GPU inference failed: '+str(info.get('error','Unknown error')))
            time.sleep(interval)
        raise TimeoutError('GPU job timeout; the backend may still be processing this job')

    def download(self,job_id,target):
        target=Path(target);size=0
        created=False
        try:
            with self.request('GET','/jobs/'+job_id+'/result',stream=True) as r, target.open('xb') as output:
                created=True
                for chunk in r.iter_content(1024**2):
                    size+=len(chunk)
                    if size>MAX_BYTES:raise RuntimeError('GPU result exceeds size limit')
                    output.write(chunk)
        except BaseException:
            if created:target.unlink(missing_ok=True)
            raise
        return target
