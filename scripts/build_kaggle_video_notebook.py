"""Build the Kaggle evaluation notebook from the existing shared environment setup."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
existing=json.loads((ROOT/'notebooks/helmet_runtime_colab_kaggle.ipynb').read_text(encoding='utf8'))
cells=[]
def add(kind,text):
    cell={'cell_type':kind,'metadata':{},'id':f'kg-video-{len(cells):02d}','source':text.strip().splitlines(keepends=True)}
    if kind=='code':cell.update(execution_count=None,outputs=[])
    cells.append(cell)

add('markdown','''# Kaggle GPU: YOLO26s + DEIM + DINOv3-S, video 35 giây
Chỉ benchmark/inference, không training hoặc chạy web service. Website riêng giữ nguyên.

1. Import notebook này vào Kaggle. Tạo/add **private Dataset** từ `helmet_kaggle_input.zip` đã chuẩn bị; không publish dữ liệu/weights người dùng.
2. Settings → Accelerator: ưu tiên GPU T4 x2 nếu còn quota; code dùng **cuda:0**, không tự sử dụng cả hai GPU. Internet: On để cài package/tải DINOv3.
3. Add-ons → Secrets: thêm `HF_TOKEN` được cấp quyền DINOv3, bật quyền cho notebook. Không dán token vào code.
4. Chạy phần 1–3; sau cài dependencies restart Python, chạy lại phần1 rồi tiếp tục phần4–7. Không chạy lại installation mỗi lần benchmark.
5. Tải `helmet_kaggle_results.zip` sau khi xong. GPU/model/provider và thời gian là số đo của phiên thật, không phải kết quả có sẵn.

Giữ imgsz1280, stride3, min_obs3, F2 bật, CLIP tắt; video xuất đầy đủ frame/FPS/resolution/audio tương thích. Chỉ so speedup nếu cùng source/trọng số/encoder/code/cấu hình. Quota và GPU được cấp phụ thuộc tài khoản/availability, xem trực tiếp trong Kaggle.''')
add('markdown','## 1. Mở private input và xác minh file')
add('code',r'''
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, subprocess, sys, zipfile
IS_KAGGLE = Path('/kaggle/working').is_dir()
if not IS_KAGGLE:
    raise RuntimeError('Run this notebook on Kaggle. Local CPU uses the same benchmark CLI.')
BASE = Path('/kaggle/working')
PROJECT = BASE / 'helmet_kaggle_video'
INPUT_ZIP = ''  # Optional exact path if more than one matching private Dataset is attached.
matches = [Path(INPUT_ZIP)] if INPUT_ZIP else list(Path('/kaggle/input').rglob('helmet_kaggle_input.zip'))
if len(matches) > 1:
    raise RuntimeError('Set INPUT_ZIP to the exact intended bundle; do not guess between datasets.')
archive_path = matches[0] if matches else None
if archive_path:
    with zipfile.ZipFile(archive_path) as archive:
        meta_bytes = archive.read('kaggle_input.json')
        source_root = None
else:
    # Kaggle may unpack uploaded archives automatically into the input dataset.
    metadata_files = list(Path('/kaggle/input').rglob('kaggle_input.json'))
    if len(metadata_files) != 1:
        raise RuntimeError('Add exactly one private helmet_kaggle_input bundle as notebook input.')
    source_root = metadata_files[0].parent
    meta_bytes = metadata_files[0].read_bytes()
if PROJECT.exists():
    if not (PROJECT/'kaggle_input.json').is_file() or (PROJECT/'kaggle_input.json').read_bytes() != meta_bytes:
        raise RuntimeError('Existing project belongs to another/partial bundle; choose a fresh PROJECT directory.')
else:
    PROJECT.mkdir(parents=True)
    if archive_path:
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            if len(entries)>1000 or sum(e.file_size for e in entries)>2*1024**3:
                raise ValueError('Bundle exceeds file/byte limit')
            if len({e.filename for e in entries}) != len(entries):raise ValueError('Duplicate ZIP member')
            for entry in entries:
                name=entry.filename; relative=PurePosixPath(name)
                if (chr(92) in name or ':' in name or relative.is_absolute() or '..' in relative.parts
                        or (entry.external_attr>>16)&0o170000==0o120000
                        or not (PROJECT/name).resolve().is_relative_to(PROJECT.resolve())):
                    raise ValueError('Unsafe ZIP member')
            archive.extractall(PROJECT)
    else:
        shutil.copytree(source_root,PROJECT,dirs_exist_ok=True,symlinks=False)
meta=json.loads(meta_bytes)
for relative,expected in meta['files'].items():
    path=(PROJECT/relative).resolve()
    if not path.is_relative_to(PROJECT.resolve()):raise ValueError('Unsafe input path')
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest != expected:raise RuntimeError('Input checksum differs: '+relative)
os.chdir(PROJECT)
def run_python(*arguments):
    return subprocess.run([sys.executable,*map(str,arguments)],cwd=PROJECT,check=True)
print('Verified input:',meta['source_video_sha256'])
''')
add('markdown','## 2. Kiểm tra GPU thực và chọn ORT tương thích')
add('code',r'''
import torch
if not torch.cuda.is_available():raise RuntimeError('Enable Kaggle GPU; no CPU fallback.')
cuda_version=tuple(map(int,torch.version.cuda.split('.')[:2]))
cudnn_version=torch.backends.cudnn.version()
if not cudnn_version or not 90000<=cudnn_version<100000:
    raise RuntimeError('Review ORT CUDA/cuDNN compatibility for this runtime before installing.')
if cuda_version[0]==12:
    ORT_VERSION='1.22.0' if cuda_version>=(12,8) else '1.20.2'
elif cuda_version[0]==13:
    ORT_VERSION='1.30.0'  # Stack used in the user's actual T4 benchmark; still verify the live session.
else:
    raise RuntimeError('Unsupported prepared CUDA stack; do not silently install CPU ORT.')
# Catch a Torch wheel that cannot execute on an allocated older GPU (e.g. P100).
try:
    x=torch.ones((8,8),device='cuda:0');y=x@x;torch.cuda.synchronize();del x,y
except RuntimeError as error:
    raise RuntimeError('Torch cannot execute on this GPU. Select supported T4 or review the GPU wheel.') from error
print({'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':cudnn_version,
       'gpu':torch.cuda.get_device_name(0),'allocated_gpus':torch.cuda.device_count(),
       'used_gpu':'cuda:0','onnxruntime_gpu_to_install':ORT_VERSION})
subprocess.run(['nvidia-smi'],check=True)
''')
add('markdown','''## 3. Cài dependencies một lần
Giữ Torch/torchvision GPU của Kaggle. Pin thư viện model theo CPU report nếu có; ORT khác theo CPU/CUDA. Sau cell này restart Python, chạy lại mục1, bỏ qua2–3 rồi tiếp tục4. Không factory-reset/xóa runtime data.''')
install=''.join(existing['cells'][6]['source'])
marker='with tempfile.TemporaryDirectory() as temp:'
install=install.replace(marker,"""packages=meta['reference_packages']
cpu_report=PROJECT/'data/kaggle_video/openvino_cpu_report.json'
if cpu_report.is_file():
    packages=json.loads(cpu_report.read_text())['packages']
for package in ('ultralytics','transformers','numpy'):
    if package in packages:constraints.append(f'{package}=={packages[package]}')
"""+marker)
add('code',install)
add('markdown','## 4. Secrets, probe và provider GPU sau restart')
add('code',''.join(existing['cells'][8]['source']))
add('markdown','## 5. Benchmark video đầy đủ và sáu ảnh khó, cùng model mặc định')
add('code',r'''
import shutil
for command in ('ffmpeg','ffprobe'):
    if not shutil.which(command):raise RuntimeError('Missing '+command+' in runtime PATH')
for relative in ('weights/yolo26s.pt','weights/deimv2_wholebody49_boxes_only.onnx','scripts/benchmark_video_runtime.py'):
    if not (PROJECT/relative).is_file():raise FileNotFoundError('Repackage current input; missing '+relative)
OUTPUT=BASE/'helmet_kaggle_measured'
run_python('scripts/benchmark_video_runtime.py',PROJECT/'data/kaggle_video/1102.mp4',
           '--profile','cloud_cuda','--stride','3','--min-observations','3',
           '--hard-cases',PROJECT/'data/benchmark/crowded_20261106.json','--out',OUTPUT)
report=json.loads((OUTPUT/'report.json').read_text())
print(json.dumps({k:report[k] for k in ('runtime','sampled_frames','timing','vram','vehicle_tracks','unknown_rate','hard_cases')},indent=2,ensure_ascii=False))
''')
add('markdown','''## 6. So sánh với Intel OpenVINO CPU
Không chạy OpenVINO CPU trong máy Kaggle để đại diện cho laptop Intel. CPU report trong bundle là số đo local. Nếu fingerprint khác, giữ kết quả GPU nhưng không báo speedup; xem khác biệt trước khi chạy lại.''')
add('code',r'''
cpu=PROJECT/'data/kaggle_video/openvino_cpu_report.json'
if cpu.is_file():
    shutil.copy2(cpu,OUTPUT/'openvino_cpu_report.json')
    try:
        run_python('scripts/benchmark_video_runtime.py','--compare',cpu,OUTPUT/'report.json','--out',OUTPUT)
        print(json.loads((OUTPUT/'comparison.json').read_text()))
        import csv
        from IPython.display import Markdown,display
        cpu_data=json.loads(cpu.read_text())
        measured=json.loads((OUTPUT/'report.json').read_text())
        fields=[('processing_seconds','timing'),('model_load_seconds','timing'),('warmup_seconds','timing'),
                ('video_wall_seconds','timing'),('sampled_processing_fps','timing'),('inference_fps','timing'),
                ('latency_p50_ms','timing'),('latency_p95_ms','timing'),('process_peak_mib','vram'),
                ('sampled_frames',None),('vehicle_tracks',None),('person_detections',None),
                ('unknown_rate',None),('precision',None),('recall',None),('fp',None),('fn',None)]
        table=[]
        for key,section in fields:
            values=[(data[section] if section else data).get(key) for data in (cpu_data,measured)]
            table.append([key,*values])
        for key in ('frame_count','fps','video_duration','width','height'):
            table.append(['output_'+key,*[data['media']['output'][key] for data in (cpu_data,measured)]])
        with (OUTPUT/'comparison_table.csv').open('w',newline='') as file:
            writer=csv.writer(file);writer.writerow(['metric','Intel OpenVINO CPU','Kaggle GPU']);writer.writerows(table)
        def cell(value):return 'N/A' if value is None else f'{value:.3f}' if isinstance(value,float) else str(value)
        display(Markdown('| Metric | Intel OpenVINO CPU | Kaggle GPU |\n|---|---:|---:|\n'+
                         '\n'.join('| '+' | '.join(map(cell,row))+' |' for row in table)))
    except subprocess.CalledProcessError:
        print('Comparison rejected: input/weights/encoder/code/options mismatch. GPU report retained; no speedup claimed.')
else:
    print('Local CPU report not present. Run the same CLI on the Intel laptop before comparing.')
''')
add('markdown','''## 7. Tải kết quả và kiểm tra chất lượng
ZIP chỉ lấy thư mục kết quả, không export project/weights/HF cache/secrets. `hard_cases/*_annotated.jpg` và `helmet_review.csv` chứa dự đoán để kiểm tra; `head_box_metrics` là đo box đầu trên nhãn thô của ảnh phát triển, không phải helmet precision/recall. Bộ có mũ lông, nhiều người/trẻ em, đêm/mưa; chưa xác nhận đủ mẫu mũ len. Cần thêm GT cả đầu bị bỏ sót trước khi đo recall phân loại. Sau khi tải kết quả hãy dừng session GPU để giữ quota.''')
add('code',r'''
from IPython.display import FileLink,display
if not (OUTPUT/'report.json').is_file():
    raise RuntimeError('Benchmark incomplete: fix the GPU execution error before exporting measured results.')
archive=Path(shutil.make_archive(str(BASE/'helmet_kaggle_results'),'zip',root_dir=OUTPUT))
print('Result:',archive,'bytes:',archive.stat().st_size)
display(FileLink(str(archive)))
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                  'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
target=ROOT/'notebooks/helmet_kaggle_video_benchmark.ipynb'
target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
for index,cell in enumerate(cells):
    if cell['cell_type']=='code':compile(''.join(cell['source']),f'notebook_cell_{index}','exec')
print(f'Created {target} ({len(cells)} cells; code syntax checked, not executed on GPU)')
