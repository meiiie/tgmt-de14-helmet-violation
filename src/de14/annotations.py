"""Source-bound human labels; uncertain/draft annotations never become safe GT."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


def file_sha256(path):
    with open(path,'rb') as file: return hashlib.file_digest(file,'sha256').hexdigest()


def vehicle_violation(vehicle):
    if vehicle['occupancy']=='unknown': return None
    if vehicle['occupancy']=='empty': return False
    labels=[head['helmet'] for head in vehicle['heads']]
    if 'no_helmet' in labels: return True
    if len(labels)==vehicle['rider_count'] and labels and all(lab=='helmet' for lab in labels): return False
    return None


def validate_labels(data,expected_source_sha256=None):
    if data.get('schema_version')!=1: raise ValueError('Unsupported annotation schema_version')
    source=data.get('source',{})
    if not re.fullmatch(r'[a-f0-9]{64}',source.get('sha256','')): raise ValueError('Invalid source SHA256')
    if expected_source_sha256 and source['sha256']!=expected_source_sha256: raise ValueError('Annotation source does not match video')
    for key in ('width','height','frame_count'):
        if type(source.get(key)) is not int or source[key]<1: raise ValueError(f'Invalid source {key}')
    fps=source.get('fps')
    if isinstance(fps,bool) or not isinstance(fps,(int,float)) or not math.isfinite(fps) or fps<=0: raise ValueError('Invalid source FPS')
    def box(value):
        if (not isinstance(value,list) or len(value)!=4 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in value)
                or not (0<=value[0]<value[2]<=source['width'] and 0<=value[1]<value[3]<=source['height'])):
            raise ValueError('Invalid box in original source coordinates')
    frames=data.get('frames')
    if not isinstance(frames,list) or not frames: raise ValueError('No annotation frames')
    seen=set(); reviewed=vehicles=reviewed_vehicles=known=0
    for frame in frames:
        index=frame.get('source_index')
        if type(index) is not int or not 0<=index<source['frame_count'] or index in seen: raise ValueError('Invalid or duplicate source frame')
        seen.add(index)
        timestamp=frame.get('timestamp_s')
        if isinstance(timestamp,bool) or not isinstance(timestamp,(int,float)) or not math.isfinite(timestamp) or abs(timestamp-index/fps)>1e-4:
            raise ValueError('Invalid source timestamp')
        if type(frame.get('reviewed')) is not bool: raise ValueError('Frame reviewed flag must be explicit boolean')
        reviewed+=frame['reviewed']; ids=set()
        if not isinstance(frame.get('vehicles'),list): raise ValueError('Vehicles must be a list')
        for vehicle in frame['vehicles']:
            identifier=vehicle.get('id','')
            if not isinstance(identifier,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',identifier): raise ValueError('Invalid vehicle ID')
            if identifier in ids: raise ValueError('duplicate vehicle ID within one frame')
            ids.add(identifier); box(vehicle.get('box'))
            if vehicle.get('occupancy') not in {'occupied','empty','unknown'}: raise ValueError('Invalid occupancy')
            count=vehicle.get('rider_count')
            if type(count) is not int or count<0: raise ValueError('Invalid rider count')
            heads=vehicle.get('heads')
            if not isinstance(heads,list): raise ValueError('Heads must be a list')
            if vehicle['occupancy']=='empty' and (count or heads): raise ValueError('An empty vehicle cannot have riders/heads')
            if vehicle['occupancy']=='occupied' and count<1: raise ValueError('Occupied vehicle needs rider_count>=1')
            if len(heads)>count: raise ValueError('Head count exceeds rider count')
            head_ids=set()
            for head in heads:
                hid=head.get('id','')
                if not isinstance(hid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',hid) or hid in head_ids: raise ValueError('Invalid or duplicate head ID')
                head_ids.add(hid);box(head.get('box'))
                if head.get('helmet') not in {'helmet','no_helmet','unknown'}: raise ValueError('Invalid helmet label')
            vehicles+=1; reviewed_vehicles+=frame['reviewed']
            known+=frame['reviewed'] and vehicle_violation(vehicle) is not None
    return {'frames':len(frames),'reviewed_frames':reviewed,'vehicles':vehicles,
            'reviewed_vehicles':reviewed_vehicles,'reviewed_known_vehicle_labels':known,
            'all_anchor_frames_reviewed':reviewed==len(frames),'scope':'sampled_source_frames_only'}


def prepare_package(video,out_dir,stride=10,source_info=None,expected_source_sha256=None):
    """Extract clean full-resolution images; never run models or overwrite label files."""
    import cv2
    from .video_io import probe_video
    if type(stride) is not int or stride<1: raise ValueError('Invalid annotation sampling stride')
    video=Path(video);out=Path(out_dir);digest=file_sha256(video)
    if expected_source_sha256 and digest!=expected_source_sha256: raise ValueError('Cached source metadata hash mismatch')
    if source_info is not None and not expected_source_sha256: raise ValueError('Cached source metadata requires source SHA256')
    info=source_info or probe_video(video)
    source={'name':video.name,'sha256':digest,'fps':info['fps'],'fps_rational':info['fps_rational'],
            'width':info['width'],'height':info['height'],'frame_count':info['frame_count'],'duration_s':info['video_duration']}
    frames=[{'source_index':i,'timestamp_s':info['timestamps'][i],'image':f'frames/{i:06d}.jpg','reviewed':False,'vehicles':[]}
            for i in range(0,info['frame_count'],stride)]
    template={'schema_version':1,'status':'draft','source':source,'scope':'sampled_source_frames','sample_stride':stride,'frames':frames}
    validate_labels(template)
    manifest=out/'manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text(encoding='utf-8'))
        if (previous.get('source',{}).get('sha256')!=digest or previous.get('sample_stride')!=stride
                or any(f.get('reviewed') or f.get('vehicles') for f in previous.get('frames',[]))):
            raise ValueError('Existing annotation package differs or contains labels; use a new output folder')
    (out/'frames').mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(str(video));wanted={f['source_index']:f for f in frames};decoded=written=0
    try:
        while True:
            ok,image=capture.read()
            if not ok: break
            if decoded in wanted:
                if (image.shape[1],image.shape[0])!=(source['width'],source['height']): raise ValueError('Source image dimensions mismatch')
                if not cv2.imwrite(str(out/wanted[decoded]['image']),image,[cv2.IMWRITE_JPEG_QUALITY,92]): raise IOError('Cannot write annotation image')
                written+=1
            decoded+=1
    finally: capture.release()
    if decoded!=info['frame_count'] or written!=len(frames): raise ValueError('Source decode count mismatch')
    encoded=json.dumps(template,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page=Path(__file__).with_name('annotation_ui.html').read_text(encoding='utf-8').replace('__ANNOTATION_DATA__',encoded)
    (out/'index.html').write_text(page,encoding='utf-8')
    manifest.write_text(json.dumps(template,ensure_ascii=False,indent=2),encoding='utf-8')
    return template
