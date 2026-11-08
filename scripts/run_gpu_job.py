"""Execute a checked media job on an NVIDIA host or interactive GPU notebook.

Uses the existing runtime and pipeline. Never runs a web server or falls back CPU.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def execute_job(request,out):
    import cv2
    import torch
    from de14.remote_jobs import JobOptions,read_request,create_result,MAX_FILES
    from de14.runtime import load_profile,build_runtime
    from de14.video_io import run_video,json_default
    from de14.core import VIOL,UNDET
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run this job on the NVIDIA GPU backend, not local Intel Python.')
    started=time.perf_counter()
    work=Path(out).resolve()/uuid.uuid4().hex
    manifest,media=read_request(request,work/'input_job')
    options=JobOptions(**manifest['options'])
    config=load_profile('cloud_cuda',imgsz=options.imgsz,detector_kind=options.detector,
                        classifier_kind=options.classifier,accurate=options.accurate,detect_heads=options.detect_heads)
    load_start=time.perf_counter();runtime=build_runtime(config);load_seconds=time.perf_counter()-load_start
    if not runtime.device.startswith('cuda'):
        raise RuntimeError('GPU job resolved to CPU; refusing to label CPU execution as GPU.')
    metadata=runtime.metadata()
    if options.detect_heads and 'CUDAExecutionProvider' not in metadata.get('head_providers',[]):
        raise RuntimeError('Head detector is not using CUDAExecutionProvider; fix onnxruntime-gpu before inference.')
    output=work/'output';output.mkdir(parents=True,exist_ok=False)
    filtering=json.loads((ROOT/'configs/track_filter.json').read_text(encoding='utf8'))['track_filter'] if options.kind=='video' and options.use_filter else None
    pipeline=runtime.pipeline(n_min=options.n_min if options.kind=='video' else 1,track_filter=filtering)
    if options.kind=='video':
        result=run_video(media,pipeline,output,'video',stride=options.stride,max_seconds=options.max_seconds,
                         max_frames=options.max_frames,output_mode=options.output_mode)
        annotated=Path(result['video_path'])
        info=result['media']['output']
        media_info={k:info[k] for k in ('frame_count','fps','width','height','video_duration')}
        media_info['audio_streams']=len(info['audio'])
        media_info['output_mode']=options.output_mode
        prefix='video'
    else:
        image=cv2.imread(str(media))
        if image is None:raise ValueError('Cannot decode input image')
        result=pipeline.run([(0,image)],out_dir=str(output),video_name='image',tracking=False,write_video=False,render=True)
        annotated=output/'image_annotated.png'
        if not cv2.imwrite(str(annotated),pipeline.last_vis):raise IOError('Cannot save annotated image')
        media_info={'width':image.shape[1],'height':image.shape[0],'frame_count':1}
        prefix='image'
    audit=output/'pipeline_audit.json'
    audit.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=json_default,allow_nan=False),encoding='utf8')
    artifacts={annotated.name:annotated,'pipeline_audit.json':audit}
    tracks=output/f'{prefix}_tracks.csv'
    for name in (f'{prefix}_tracks.csv',f'{prefix}_observations.csv',f'{prefix}_filter_audit.json'):
        p=output/name
        if p.is_file():artifacts[name]=p
    # Evidence is optional for display; only generated, bounded job outputs are exported.
    all_evidence=sorted((output/'evidence').glob('*.jpg'))
    for p in all_evidence[:MAX_FILES-1-len(artifacts)]:artifacts['evidence/'+p.name]=p
    payload={'kind':options.kind,'options':manifest['options'],'runtime':metadata,
             'media_path':'artifacts/'+annotated.name,'tracks_path':'artifacts/'+tracks.name if tracks.is_file() else None,
             'media':media_info,
             'counts':{'vehicles':len(result['tracks']),'people':sum(t['n_riders'] for t in result['tracks'].values()),
                       'violations':sum(t['decision']==VIOL for t in result['tracks'].values()),
                       'unknown':sum(t['decision']==UNDET for t in result['tracks'].values())},
             'timings':{'sampled_frames':result['frames'],'processing_seconds':result['sec'],
                        'sampled_processing_fps':result['frames']/result['sec'] if result['sec'] else None,
                        'model_load_seconds':load_seconds,'total_seconds_before_packaging':time.perf_counter()-started},
             'f2':result['f2'],'temporal_backfill':{'vehicle_frames':result['temporal_backfill']['vehicle_frames']},
             'untracked_head_retention':result.get('untracked_head_retention', {}),
             'evidence_included':sum(n.startswith('evidence/') for n in artifacts),'evidence_generated':len(all_evidence)}
    archive=work/f"helmet_gpu_result_{manifest['job_id']}.zip"
    create_result(manifest['job_id'],manifest['input']['sha256'],payload,artifacts,archive)
    print(json.dumps({'gpu':metadata.get('cuda_device'),'torch_device':metadata['torch_device'],
                      'head_providers':metadata.get('head_providers'),'counts':payload['counts'],
                      'timings':payload['timings'],'media':media_info,'result_zip':str(archive)},ensure_ascii=True),flush=True)
    return archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path);parser.add_argument('--out',type=Path,default=ROOT/'outputs/colab_gpu')
    args=parser.parse_args()
    path=execute_job(args.request,args.out)
    print('GPU_JOB_RESULT='+str(path),flush=True)


if __name__=='__main__':main()
