#@title Chạy ảnh/video trên T4 trong phiên Colab đã cài model (không chạy lại benchmark)
from pathlib import Path
import json
import os
import stat
import subprocess
import sys
import uuid
from zipfile import ZipFile


def update_project_sources(archive_path, project):
    """Update trusted source ZIP; preserve all existing model/probe files."""
    project=Path(project).resolve()
    with ZipFile(archive_path) as archive:
        infos=archive.infolist()
        if len(infos)>500 or sum(i.file_size for i in infos)>16*1024**2:
            raise ValueError('Source ZIP unexpectedly large; use the small current code ZIP, not model/data archives.')
        if len({i.filename for i in infos})!=len(infos):raise ValueError('Duplicate ZIP members')
        targets=[]
        for entry in infos:
            name=entry.filename
            if '\\' in name or ':' in name or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe ZIP path')
            if stat.S_ISLNK(entry.external_attr>>16):raise ValueError('Source ZIP symlink rejected')
            target=(project/name).resolve()
            if not target.is_relative_to(project):raise ValueError('Unsafe source destination')
            if not entry.is_dir() and name.split('/')[0] in {'src','scripts','configs','weights'}:
                targets.append((entry,target))
        if 'scripts/run_gpu_job.py' not in archive.namelist():
            raise ValueError('Code ZIP is old or is a benchmark result. Download the current code ZIP from GPU Colab tab.')
        for entry,target in targets:
            if entry.filename.startswith('weights/') and target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.read(entry))


def run_colab_inference():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This Python session has no CUDA. In Colab select the T4 GPU runtime; do not run this cell on local Intel.')
    from google.colab import files
    project=Path(globals().get('PROJECT','/content/helmet_runtime')).resolve()
    if not (project/'scripts/benchmark_runtime.py').is_file():
        raise RuntimeError('Use the Colab notebook/session where the benchmark ran, or run its setup/model cells first. No benchmark rerun is needed.')
    print('GPU:',torch.cuda.get_device_name(0))
    print('Select TWO ZIP files: current code ZIP + image/video request ZIP. Do not select helmet_runtime_benchmark.zip.')
    uploaded=files.upload()
    paths=[Path(name).resolve() for name in uploaded if name.lower().endswith('.zip')]
    del uploaded
    code=[];jobs=[]
    for path in paths:
        with ZipFile(path) as archive:
            names=set(archive.namelist())
            if 'scripts/run_gpu_job.py' in names:code.append(path)
            if 'request.json' in names:jobs.append(path)
    if len(code)!=1 or len(jobs)!=1:
        raise RuntimeError('Select one current code ZIP and one request ZIP. Benchmark ZIP contains measurements, not input media.')
    update_project_sources(code[0],project)
    # Preserve dependencies, downloaded models, HF_TOKEN and the active T4 runtime.
    os.chdir(project)
    if not os.environ.get('HF_TOKEN'):
        try:
            from google.colab import userdata
            token=userdata.get('HF_TOKEN')
            if token:os.environ['HF_TOKEN']=token
            del token
        except Exception:
            pass  # An already authorized model cache may suffice; runtime reports real access errors.
    output=project/'outputs/colab_gpu_notebook'/uuid.uuid4().hex
    subprocess.run([sys.executable,'scripts/run_gpu_job.py',str(jobs[0]),'--out',str(output)],cwd=project,check=True)
    results=list(output.glob('*/helmet_gpu_result_*.zip'))
    if len(results)!=1:raise RuntimeError('Expected one completed GPU result ZIP; inspect the preceding inference error.')
    with ZipFile(results[0]) as archive:metadata=json.loads(archive.read('result.json'))['result']
    print('Completed on GPU:',metadata['runtime'].get('cuda_device'))
    print('Media:',metadata['media'])
    print('Download this ZIP and import it in the local app GPU Colab tab:',results[0].name)
    files.download(str(results[0]))


if __name__=='__main__':run_colab_inference()
