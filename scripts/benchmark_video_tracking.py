"""Measured-history ablation: public live smoke + frozen user-video replay.

Assisted labels are never GT. Replayed detector timings are never inference FPS.
Uses the shared CPU/CUDA runtime and unchanged detector/classifier models.
"""
from collections import defaultdict
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from de14.core import crop, decide, iou, UNDET, VIOL, OK
from de14.headdet import square
from de14.pipeline import Pipeline
from de14.runtime import load_profile, build_runtime
from de14.metrics import evaluate_frames
from de14.video_io import sampled_frames, render_video, json_default
from benchmark_video_sampling import encoder_identity


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=json_default,allow_nan=False),encoding='utf8')


def predictions(result,clip):
    frames=[]
    for fi,records in result['per_frame']:
        vs=[]
        for tid,box,_,heads in records:
            labs=[decide([p],1)[0] for _,p in heads]
            raw_decision=VIOL if VIOL in labs else OK if labs and all(l==OK for l in labs) else UNDET
            vs.append({'track_id':tid,'box':box,'decision':result['frame_decisions'][fi][tid],
                       'raw_frame_decision':raw_decision,'n_riders':len(heads)})
        frames.append({'clip':clip,'frame':fi,'vehicles':vs})
    return frames


def matches(gt,ps,threshold):
    gs=set();pp=set()
    for _,i,j in sorted((-iou(g,p),i,j) for i,g in enumerate(gt) for j,p in enumerate(ps) if iou(g,p)>=threshold):
        if i not in gs and j not in pp:gs.add(i);pp.add(j)
    return len(gs)


def supplied_box_coverage(result,labels):
    rec={result['timeline'][fi]['source_index']:records for fi,records in result['per_frame']}
    counts=defaultdict(int)
    for f in labels['frames']:
        if not f['reviewed']:continue
        records=[r for r in rec[f['source_index']] if r[0] in result['tracks']]
        hg=[h['box'] for v in f['vehicles'] for h in v['heads']]
        vg=[v['box'] for v in f['vehicles'] if v['occupancy']=='occupied']
        counts['provided_head_boxes']+=len(hg);counts['provided_occupied_vehicle_boxes']+=len(vg)
        counts['matched_head_boxes']+=matches(hg,[h for _,_,_,hs in records for h,_ in hs],.5)
        counts['matched_occupied_vehicle_boxes']+=matches(vg,[b for _,b,_,_ in records],.3)
    return {'scope':'original49explicitlyreviewed anchors only; diagnostic coverage, NOT final precision/recall',**counts}


class CropCache:
    def __init__(self,classifier):self.classifier=classifier;self.values={};self.misses=0;self.hits=0;self.forward_seconds=0.
    @staticmethod
    def key(c):return hashlib.sha256(str(c.shape).encode()+c.tobytes()).hexdigest()
    def __call__(self,crops):
        keys=[self.key(c) for c in crops];missing={k:c for k,c in zip(keys,crops) if k not in self.values}
        if missing:
            start=time.perf_counter();ps=self.classifier(list(missing.values()));self.forward_seconds+=time.perf_counter()-start
            if len(ps)!=len(missing):raise ValueError('Classifier result count mismatch')
            self.values.update(zip(missing,ps));self.misses+=len(missing)
        self.hits+=sum(k not in missing for k in keys)
        return [self.values[k] for k in keys]


def public_smoke(runtime,manifest,out,repeats):
    groups=defaultdict(list)
    for row in manifest['frames']:
        if sha(ROOT/row['image'])!=row['sha256']:raise ValueError('Public frame hash mismatch')
        image=cv2.imread(str(ROOT/row['image']))
        if image is None:raise ValueError('Unreadable public image')
        image=cv2.resize(image,tuple(manifest['source_resolution']),interpolation=cv2.INTER_CUBIC)
        groups[row['clip']].append((row['frame'],image))
    modes={'baseline':(False,True),'backfill':(True,True),'backfill_iou_experiment':(True,False)}
    warm=next(iter(groups.values()))[0]
    runtime.pipeline(n_min=1,backfill_tentative=False).run([warm],write_video=False,evidence=False)
    rows=[]
    for repeat in range(repeats):
        collected={m:[] for m in modes};latencies={m:[] for m in modes};secs=defaultdict(float);backfilled=defaultdict(int)
        for ci,(clip,frames) in enumerate(sorted(groups.items())):
            order=list(modes) if (repeat+ci)%2==0 else list(reversed(modes))
            for mode in order:
                bf,fuse=modes[mode]
                result=runtime.pipeline(fps=10,n_min=1,backfill_tentative=bf,tracker_fuse_score=fuse).run(sorted(frames),write_video=False,evidence=False)
                collected[mode].extend(predictions(result,clip));secs[mode]+=result['sec'];latencies[mode].extend(result['frame_latency_ms'])
                backfilled[mode]+=result['temporal_backfill']['vehicle_frames']
                print(f'public repeat={repeat+1} {mode} {clip}: {result["sec"]:.2f}s',flush=True)
        for mode in modes:
            metrics=evaluate_frames(manifest['frames'],collected[mode])
            rows.append({'mode':mode,'repeat':repeat+1,'metrics':metrics,'processing_seconds':secs[mode],
                         'inference_fps':len(manifest['frames'])/secs[mode],
                         'latency_ms_p50':float(np.median(latencies[mode])),
                         'latency_ms_p95':float(np.percentile(latencies[mode],95)),
                         'backfilled_vehicle_frames':backfilled[mode]})
            save(out/f'public_{mode}_repeat{repeat+1}.json',collected[mode])
        save(out/'public_smoke.json',{'rows':rows,'limits':manifest['limitations'],'inference':'live, uncached, one shared warmed runtime; alternating variant order; timings include decode-free pipeline calls'})
    return rows


def cached_user_video(runtime,source,cache_dir,out,strides,render,render_strides=None):
    provenance=json.loads((cache_dir/'provenance.json').read_text(encoding='utf8'))
    if sha(source)!=provenance['source_sha256']:raise ValueError('Source video hash mismatch')
    for path,digest in provenance['weights_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError(f'Cached weight mismatch: {path}')
    labels=json.loads((ROOT/'outputs/label_review_20261106/labels_received.json').read_text(encoding='utf8'))
    if labels['source']['sha256']!=sha(source):raise ValueError('Supplied label source mismatch')
    rows=[];filtering=provenance['filter']
    for stride in strides:
        folder=cache_dir/f'stride{stride}'
        old=json.loads((folder/'benchmark_measurements.json').read_text(encoding='utf8'))['result']
        info=old['media']['source'];cached={};raw={}
        for line in (folder/'video_stages.jsonl').open(encoding='utf8'):
            e=json.loads(line)
            if e['stage']=='detector_output':raw[e['frame']]=e['detections']
        by_source={old['timeline'][str(fi)]['source_index']:records for fi,records in old['per_frame']}
        classifier=CropCache(runtime.classifier)
        # Use exact decoded source pixels to identify previously classified crops.
        for _,image in sampled_frames(source,info,stride,0,0,timeline:={}):
            n=timeline[max(timeline)]['source_index']
            for _,_,_,heads in by_source[n]:
                for head,p in heads:
                    c=crop(image,square(head,margin=.15),8)
                    if c is not None:classifier.values[CropCache.key(c)]=p
        for mode,bf,fuse in [('baseline',False,True),('backfill',True,True),('backfill_iou_experiment',True,False)]:
            sequence=iter(raw.values());timeline={};before_misses=classifier.misses;before_forward=classifier.forward_seconds
            start=time.perf_counter()
            result=Pipeline(lambda _:next(sequence),classifier,fps=info['fps']/stride,n_min=5,
                            batch_heads=True,track_filter=filtering,backfill_tentative=bf,tracker_fuse_score=fuse).run(
                sampled_frames(source,info,stride,0,0,timeline),write_video=False,evidence=False,frame_metadata=timeline)
            result.update(hi=.6,lo=.4)
            result['media']={'source':info,'stride':stride,'output_mode':'full',
                             'unsampled_policy':'original_unannotated','interpolation':False,
                             'inference_scope':'frozen detector replay; cached old classification and live classification of new crops'}
            replay_seconds=time.perf_counter()-start
            if mode=='baseline':
                # Identity, boxes, votes, F2 and source times must match saved pre-fix output.
                saved_tracks={int(k):v for k,v in old['tracks'].items()}
                if json.dumps(result['per_frame'],default=json_default)!=json.dumps(old['per_frame'],default=json_default):raise AssertionError('Legacy per-frame parity failed')
                if result['tracks']!=saved_tracks:raise AssertionError('Legacy track/vote parity failed')
            metrics={'mode':mode,'stride':stride,'scope':'frozen detector + cached old classification, real classifier only for new crops; NOT live inference latency',
                     'supplied_box_coverage':supplied_box_coverage(result,labels),'vehicle_tracks':len(result['tracks']),
                     'vehicle_records':sum(len(rs) for _,rs in result['per_frame']),
                     'unknown_tracks':sum(r['decision']==UNDET for r in result['tracks'].values()),
                     'violation_tracks':sum(r['decision']==VIOL for r in result['tracks'].values()),
                     'f2':result['f2'],'head_crops':result['counters']['associated_heads'],
                     'backfill':result['temporal_backfill'],'replay_wall_seconds':replay_seconds,
                     'new_classifier_crops':classifier.misses-before_misses,'new_classifier_seconds':classifier.forward_seconds-before_forward,
                     'precision':None,'recall':None,'fp':None,'fn':None,'id_switches':None,'quality_status':'N/A: supplied labels incomplete; assisted draft deliberately excluded'}
            if render and mode=='backfill' and (render_strides is None or stride in render_strides):
                path=out/f'1102_backfill_stride{stride}_full_length.mp4'
                metrics['output_video']=render_video(source,info,result,path)
                metrics['output']=result['media']
            save(out/f'stride{stride}_{mode}_predictions.json',result)
            rows.append(metrics);save(out/'user_video_replay.json',{'rows':rows,'legacy_parity':'asserted exact per_frame and tracks/votes for every baseline','detector_cache_sha256':{str(s):sha(cache_dir/f'stride{s}/video_stages.jsonl') for s in strides}})
            print(f'user replay stride={stride} {mode}: heads={metrics["head_crops"]} records={metrics["vehicle_records"]} coverage={metrics["supplied_box_coverage"]["matched_head_boxes"]}/68',flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/video_tracking_20261106')
    p.add_argument('--source',type=Path,required=True,help='Path to the original benchmark video')
    p.add_argument('--profile',default='openvino_cpu');p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--strides',type=int,nargs='+',default=[10,5,2]);p.add_argument('--render',action='store_true')
    p.add_argument('--render-strides',type=int,nargs='+',default=[2],help='Render only selected strides; default2')
    p.add_argument('--replay-only',action='store_true',help='Load only the identical classifier for frozen-detector replay')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    config=load_profile(args.profile,accurate=False,detect_heads=True)
    if args.replay_only:
        from types import SimpleNamespace
        import torch
        from de14.classifiers import DinoProbe,VerifiedHelmetCls
        from de14.runtime import resolve_device
        torch.set_num_threads(config.cpu_threads)
        device=resolve_device(config.device)
        print('Frozen detector replay: loading identical DINO classifier only; no detector model needed',flush=True)
        base=DinoProbe.load(str(ROOT/config.probe_path),device=device,local_files_only=config.local_files_only)
        identity=encoder_identity(base.e)
        save(args.out/'replay_encoder_candidate.json',identity)
        live_path=ROOT/'outputs/video_tracking_20261106/tracking_benchmark_provenance.json'
        if live_path.exists() and json.dumps(identity,sort_keys=True)!=json.dumps(json.loads(live_path.read_text(encoding='utf8'))['encoder'],sort_keys=True):
            raise ValueError('Replay encoder/processor differs from live benchmark')
        runtime=SimpleNamespace(classifier=VerifiedHelmetCls(base,None))
        save(args.out/'replay_provenance.json',{'encoder':identity,'config':asdict(config),
            'classifier_only_load':True,'detector_source':'frozen original raw outputs, weight hashes checked',
            'pipeline_sha256':sha(ROOT/'src/de14/pipeline.py'),'tracker_sha256':sha(ROOT/'src/de14/tracking.py'),
            'benchmark_sha256':sha(Path(__file__)),'source_sha256':sha(args.source)})
        cached_user_video(runtime,args.source,ROOT/'outputs/benchmarks/video_sampling_20261106',args.out,args.strides,args.render,args.render_strides)
        return
    print('Loading shared runtime (YOLO26s + DEIMv2 + DINO; CLIP off)',flush=True)
    start=time.perf_counter();runtime=build_runtime(config);load_seconds=time.perf_counter()-start
    base=getattr(runtime.classifier,'base',runtime.classifier)
    save(args.out/'tracking_benchmark_provenance.json',{'config':asdict(config),'runtime':runtime.metadata(),
        'encoder':encoder_identity(base.e),'load_seconds':load_seconds,
        'code_sha256':{str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),ROOT/'src/de14/tracking.py',ROOT/'src/de14/pipeline.py']},
        'weights_sha256':{path:sha(ROOT/path) for path in [config.yolo_weights,config.head_path,config.probe_path]},
        'public_manifest_sha256':sha(ROOT/'data/benchmark/manifest.json'),'source_sha256':sha(args.source),
        'assisted_draft_is_ground_truth':False})
    manifest=json.loads((ROOT/'data/benchmark/manifest.json').read_text(encoding='utf8'))
    public_smoke(runtime,manifest,args.out,args.repeats)
    cached_user_video(runtime,args.source,ROOT/'outputs/benchmarks/video_sampling_20261106',args.out,args.strides,args.render,args.render_strides)
    print('Completed; public smoke is live, user video is frozen replay. Never combine their timings.',flush=True)


if __name__=='__main__':main()
