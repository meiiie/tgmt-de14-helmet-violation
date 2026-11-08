"""Build a private, explicit input bundle: current source, known weights and test media."""
import argparse,json,tempfile,importlib.metadata,hashlib
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from package_cloud_workspace import package_workspace
from benchmark_video_sampling import sha256

ROOT=Path(__file__).resolve().parents[1]


def package(video,output,cpu_report=None):
    video=Path(video).resolve();output=Path(output).resolve()
    manifest=ROOT/'data/benchmark/crowded_20261106.json'
    rows=json.loads(manifest.read_text(encoding='utf8'))['images']
    assets={'data/kaggle_video/1102.mp4':video,'data/benchmark/crowded_20261106.json':manifest,
            **{p:ROOT/p for p in ('weights/yolo26s.pt','weights/deimv2_wholebody49_boxes_only.onnx')}}
    for row in rows:
        path=(ROOT/row['image']).resolve()
        if not path.is_relative_to(ROOT) or sha256(path)!=row['image_sha256']:raise ValueError('Hard-case input differs')
        assets[row['image']]=path
    if cpu_report:assets['data/kaggle_video/openvino_cpu_report.json']=Path(cpu_report).resolve()
    for path in assets.values():
        if not path.is_file() or path.is_symlink():raise ValueError('Expected regular input file: '+str(path))
    if output in assets.values():raise ValueError('Output must not overwrite input')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.zip.partial')
    try:
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.zip';package_workspace(ROOT,source,include_probe=True)
            with ZipFile(source) as original,ZipFile(temporary,'w',compression=ZIP_DEFLATED) as target:
                code_hashes={}
                for name in original.namelist():
                    payload=original.read(name);target.writestr(name,payload)
                    code_hashes[name]=hashlib.sha256(payload).hexdigest()
                for name,path in assets.items():target.write(path,name)
                meta={'private_input_only':True,'source_code_zip_sha256':sha256(source),'source_video_sha256':sha256(video),
                      'reference_packages':{name:importlib.metadata.version(name) for name in ('ultralytics','transformers','numpy','scikit-learn')},
                      'files':{**code_hashes,**{name:sha256(path) for name,path in assets.items()}},
                      'no_hf_cache_or_token':True,'no_training':True,
                      'quality_scope':'Six development regressions, coarse head boxes only; no confirmed knit-hat classification GT'}
                target.writestr('kaggle_input.json',json.dumps(meta,indent=2))
        temporary.replace(output)
    finally:temporary.unlink(missing_ok=True)
    print(json.dumps({'bundle':str(output),'bytes':output.stat().st_size,'private':True,'video_sha256':meta['source_video_sha256']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('video',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/cloud/helmet_kaggle_input.zip')
    parser.add_argument('--cpu-report',type=Path);args=parser.parse_args();package(args.video,args.out,args.cpu_report)
