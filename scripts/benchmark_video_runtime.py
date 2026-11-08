"""Paired whole-video benchmark; common model pipeline on Intel or NVIDIA."""
import argparse,csv,json,platform,sys,time
import importlib.metadata
from pathlib import Path
from dataclasses import asdict
import cv2,numpy as np,torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from de14.runtime import load_profile,build_runtime
from de14.video_io import run_video,json_default
from de14.core import UNDET,VIOL,decide
from benchmark_runtime import VramSampler
from benchmark_video_sampling import sha256,encoder_identity,save


def experiment_identity(source,config,runtime,stride,n_min):
    paths=[config.yolo_weights,config.head_path,config.probe_path]
    modules=['core','pipeline','tracking','detectors','headdet','classifiers','runtime','video_io']
    codes=[f'src/de14/{name}.py' for name in modules]+['scripts/benchmark_video_runtime.py']
    base=getattr(runtime.classifier,'base',runtime.classifier)
    encoder=encoder_identity(base.e)
    encoder['model_config']={k:v for k,v in base.e.model.config.to_dict().items()
                             if k not in {'_name_or_path','transformers_version','_commit_hash','dtype','torch_dtype'}
                             and not k.startswith('_attn')}
    return {'source_sha256':sha256(source),'stride':stride,'n_min':n_min,'imgsz':config.imgsz,
            'detector':config.detector_kind,'classifier':config.classifier_kind,'head_enabled':config.head_enabled,
            'clip':config.accurate and config.verify_with_clip,'batch_heads':config.batch_heads,'output_mode':'full',
            'filter':json.loads((ROOT/'configs/track_filter.json').read_text())['track_filter'],
            'weights':{p:sha256(ROOT/p) for p in paths},'encoder':encoder,
            'code':{p:sha256(ROOT/p) for p in codes}}


def compare_reports(cpu,gpu):
    if cpu['experiment']!=gpu['experiment']:
        raise ValueError('Different experiment input/weights/encoder/code/options; refusing misleading speedup')
    if cpu['profile']!='openvino_cpu' or gpu['profile']!='cloud_cuda':
        raise ValueError('Expected openvino_cpu and cloud_cuda measured reports')
    return {'comparable':True,
            'processing_speedup':cpu['timing']['processing_seconds']/gpu['timing']['processing_seconds'],
            'video_wall_speedup':cpu['timing']['video_wall_seconds']/gpu['timing']['video_wall_seconds'],
            'note':'CPU/GPU hardware and execution providers differ; not an isolated algorithm optimization.'}


def validate_runtime(runtime,profile,disable_fallback=False):
    head=runtime.detector.hd
    if profile=='cloud_cuda':
        if not runtime.device.startswith('cuda') or head.backend!='onnx':raise RuntimeError('CUDA runtime required')
        if disable_fallback:head.s.disable_fallback()
        providers=head.s.get_providers()
        if not providers or providers[0]!='CUDAExecutionProvider':raise RuntimeError('Head session lost CUDA; no CPU fallback')
        return providers
    devices=head.compiled.get_property('EXECUTION_DEVICES') if head.backend=='openvino' else []
    if runtime.device!='cpu' or set(devices)!={'CPU'}:raise RuntimeError('Baseline requires real OpenVINO CPU head runtime')
    return ['OpenVINO:'+d for d in devices]


def hard_cases(runtime,manifest,out):
    from debug_rider_pipeline import head_metrics
    source=json.loads(Path(manifest).read_text(encoding='utf8'));results={};review=[]
    for row in source['images']:
        path=ROOT/row['image']
        if sha256(path)!=row['image_sha256']:raise ValueError('Hard-case image checksum differs')
        image=cv2.imread(str(path))
        if image is None:raise ValueError('Cannot decode hard-case image')
        result=runtime.pipeline(n_min=1).run([(0,image)],out_dir=str(out/row['id']),video_name=row['id'],
                                            tracking=False,write_video=False,evidence=False,render=True)
        results[row['id']]=result
        for _,records in result['per_frame']:
            for tid,_,_,heads in records:
                for box,p in heads:
                    review.append({'case':row['id'],'vehicle_id':tid,'box':json.dumps(box,default=json_default),
                                   'p_no_helmet':p,'predicted_label':decide([p],1)[0],
                                   'human_label':'','hat_type':'','reviewed':False})
        # Pipeline.last_vis is on the instance, so use the shared renderer with
        # static causal decisions and the exact measured records for this image.
        from de14.pipeline import draw_annotations
        vis=draw_annotations(image,result['per_frame'][0][1],result['frame_decisions'][0],0,result['tracks'])
        destination=out/f"{row['id']}_annotated.jpg";destination.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(destination),vis):raise IOError('Cannot save hard-case review image')
    with (out/'helmet_review.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=['case','vehicle_id','box','p_no_helmet','predicted_label','human_label','hat_type','reviewed'])
        writer.writeheader();writer.writerows(review)
    save(out/'predictions.json',results)
    return {'head_box_metrics':head_metrics(source['images'],results),'helmet_precision':None,'helmet_recall':None,
            'note':'Development regressions/coarse head GT only; helmet/knit-hat GT pending human review. Missed heads must also be annotated, not only exported predictions.'}


def benchmark(source,profile,out,stride=3,n_min=3,hard_manifest=None):
    source=Path(source);out=Path(out)
    if stride<1 or n_min<1:raise ValueError('Positive stride/min observations required')
    if (out/'report.json').exists():raise ValueError('Use a fresh output directory to preserve previous evidence')
    config=load_profile(profile,imgsz=1280,accurate=False,detect_heads=True)
    started=time.perf_counter();runtime=build_runtime(config)
    cuda=runtime.device.startswith('cuda')
    def sync():
        if cuda:torch.cuda.synchronize(runtime.device)
    sync();load_seconds=time.perf_counter()-started
    meta=runtime.metadata();meta['head_providers']=validate_runtime(runtime,profile,disable_fallback=True)
    identity=experiment_identity(source,config,runtime,stride,n_min)
    identity['hard_manifest_sha256']=sha256(hard_manifest) if hard_manifest else None
    memory=None
    try:
        capture=cv2.VideoCapture(str(source))
        try:ok,first=capture.read()
        finally:capture.release()
        if not ok:raise ValueError('Unreadable source video')
        warm=time.perf_counter()
        runtime.pipeline(n_min=1).run([(0,first)],tracking=False,write_video=False,evidence=False,render=False)
        sync();warm_seconds=time.perf_counter()-warm
        validate_runtime(runtime,profile)
        if cuda:torch.cuda.reset_peak_memory_stats(runtime.device)
        memory=VramSampler(cuda)
        pipeline=runtime.pipeline(n_min=n_min,track_filter=identity['filter']);original=pipeline.det;count=0
        def detect(image):
            nonlocal count
            output=original(image);count+=1
            if count==1 or count%10==0:print(f'{profile}: analyzed {count} frames',flush=True)
            return output
        pipeline.det=detect
        sync();wall=time.perf_counter()
        result=run_video(source,pipeline,out,'video',stride=stride,output_mode='full')
        sync();wall_seconds=time.perf_counter()-wall
        meta['head_providers']=validate_runtime(runtime,profile)
        memory.close()
        video_vram={'process_peak_mib':memory.peak,
                    'torch_peak_allocated_mib':torch.cuda.max_memory_allocated(runtime.device)/2**20 if cuda else None,
                    'note':memory.note,'scope':'Resident models + whole-video run after warmup; excludes hard-case image evaluation'}
        review=hard_cases(runtime,hard_manifest,out/'hard_cases') if hard_manifest else None
        meta['head_providers']=validate_runtime(runtime,profile)
        latency=result['frame_latency_ms'];tracks=result['tracks']
        unknown=sum(t['decision']==UNDET for t in tracks.values())
        report={'profile':profile,'experiment':identity,'runtime':meta,'host':platform.platform(),
                'sampled_frames':result['frames'],'source_frames':result['media']['source']['frame_count'],
                'timing':{'model_load_seconds':load_seconds,'warmup_seconds':warm_seconds,
                          'processing_seconds':result['sec'],'video_wall_seconds':wall_seconds,
                          'inference_fps':result['frames']/max(result['t_det']+result['t_cls'],1e-9),
                          'sampled_processing_fps':result['frames']/max(result['sec'],1e-9),
                          'video_wall_sampled_fps':result['frames']/wall_seconds,
                          'latency_p50_ms':float(np.percentile(latency,50)) if latency else None,
                          'latency_p95_ms':float(np.percentile(latency,95)) if latency else None,
                          'render_encode_seconds':result['media']['render_encode_seconds']},
                'vram':video_vram,
                'vehicle_tracks':len(tracks),'person_detections':result['counters']['person_detections'],
                'violation_tracks':sum(t['decision']==VIOL for t in tracks.values()),
                'unknown_rate':unknown/len(tracks) if tracks else None,'media':result['media'],'f2':result['f2'],
                'untracked_head_retention':result.get('untracked_head_retention'),
                'precision':None,'recall':None,'fp':None,'fn':None,'id_switches':None,
                'quality_note':'Full-video verified GT is incomplete; no inferred labels or ID-switch claims',
                'hard_cases':review,'config':asdict(config),'packages':{}}
        for name in ('torch','torchvision','ultralytics','transformers','scikit-learn','openvino','onnxruntime','onnxruntime-gpu','numpy'):
            try:report['packages'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:pass
        save(out/'report.json',report)
    finally:
        if memory and not memory.stop.is_set():memory.close()
    print(json.dumps({'report':str(out/'report.json'),'timing':report['timing'],'unknown_rate':report['unknown_rate']},ensure_ascii=True),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path,nargs='?');parser.add_argument('--profile',choices=['openvino_cpu','cloud_cuda'],default='cloud_cuda')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--stride',type=int,default=3)
    parser.add_argument('--min-observations',type=int,default=3);parser.add_argument('--hard-cases',type=Path)
    parser.add_argument('--compare',nargs=2,type=Path,metavar=('CPU_REPORT','GPU_REPORT'))
    args=parser.parse_args()
    if args.compare:
        save(args.out/'comparison.json',compare_reports(*(json.loads(p.read_text(encoding='utf8')) for p in args.compare)));return
    if not args.source:parser.error('source required for measured run')
    benchmark(args.source,args.profile,args.out,args.stride,args.min_observations,args.hard_cases)

if __name__=='__main__':main()
