"""Portable, checksummed inference jobs; no code, credentials or weights in archives."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
from zipfile import ZipFile, ZIP_STORED

PROTOCOL = 'de14-inference-job-v1'
MAX_BYTES = 2 * 1024**3
MAX_FILES = 256
MAX_MANIFEST = 2 * 1024**2


@dataclass(frozen=True)
class JobOptions:
    kind: str
    detector: str = 'yolo'
    classifier: str = 'dino'
    imgsz: int = 1280
    accurate: bool = False
    detect_heads: bool = True
    stride: int = 3
    n_min: int = 3
    use_filter: bool = True
    max_seconds: float = 0
    max_frames: int = 0
    output_mode: str = 'full'

    def __post_init__(self):
        if self.kind not in {'image', 'video'}: raise ValueError('Invalid media kind')
        if self.detector not in {'yolo', 'sam3.1'}: raise ValueError('Invalid detector')
        if self.classifier not in {'dino', 'yolo'}: raise ValueError('Invalid classifier')
        if type(self.imgsz) is not int or self.imgsz not in {640,960,1280}: raise ValueError('Invalid imgsz')
        for name, lower, upper in (('stride',1,100), ('n_min',1,100), ('max_frames',0,1000000)):
            value = getattr(self,name)
            if type(value) is not int or not lower <= value <= upper: raise ValueError(f'Invalid {name}')
        for name in ('accurate','detect_heads','use_filter'):
            if type(getattr(self,name)) is not bool: raise ValueError(f'Invalid {name}')
        if type(self.max_seconds) not in (int,float) or not math.isfinite(self.max_seconds) or not 0 <= self.max_seconds <= 86400:
            raise ValueError('Invalid max_seconds')
        if self.output_mode not in {'full','sampled_preview'}: raise ValueError('Invalid output_mode')


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')


def _name(name):
    if not isinstance(name,str) or not name or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('Unsafe archive path')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
        raise ValueError('Unsafe archive path')
    return name


def _identity(manifest, expected_type):
    if manifest.get('protocol') != PROTOCOL or manifest.get('type') != expected_type: raise ValueError('Unsupported job protocol')
    if not re.fullmatch('[0-9a-f]{32}', manifest.get('job_id','')): raise ValueError('Invalid job ID')


def _inspect(archive):
    infos = archive.infolist()
    if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_BYTES: raise ValueError('Archive exceeds size/file limit')
    names = [i.filename for i in infos]
    if len(set(names)) != len(names): raise ValueError('Duplicate archive member')
    for info in infos:
        _name(info.filename)
        if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16): raise ValueError('Archive directories/symlinks are not allowed')
        if info.flag_bits & 1: raise ValueError('Encrypted archive is not supported')
    return set(names)


def _manifest(archive, name):
    if archive.getinfo(name).file_size > MAX_MANIFEST: raise ValueError('Manifest too large')
    value = json.loads(archive.read(name).decode('utf-8'))
    if not isinstance(value,dict): raise ValueError('Manifest must be an object')
    return value


def _check_file(archive, descriptor):
    path = _name(descriptor['path'])
    if type(descriptor['size']) is not int or descriptor['size'] < 0: raise ValueError('Invalid file size')
    if archive.getinfo(path).file_size != descriptor['size']: raise ValueError('File size mismatch')
    if not re.fullmatch('[0-9a-f]{64}', descriptor['sha256']): raise ValueError('Invalid checksum')
    with archive.open(path) as source:
        digest = hashlib.file_digest(source,'sha256').hexdigest()
    if digest != descriptor['sha256']: raise ValueError('File checksum mismatch')


def _extract(archive, names, output):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise ValueError('Extraction needs a new or empty directory')
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = (output / _name(name)).resolve()
        if not target.is_relative_to(output): raise ValueError('Unsafe output path')
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(name) as source, target.open('xb') as sink: shutil.copyfileobj(source,sink,1024**2)
    return output


def _write_file(archive, source, name):
    digest = hashlib.sha256(); size = 0
    with Path(source).open('rb') as src, archive.open(_name(name),'w',force_zip64=True) as sink:
        for chunk in iter(lambda: src.read(1024**2), b''):
            size += len(chunk)
            if size > MAX_BYTES-MAX_MANIFEST: raise ValueError('Media exceeds size limit')
            digest.update(chunk); sink.write(chunk)
    return {'path':name, 'size':size, 'sha256':digest.hexdigest()}


@contextmanager
def _archive(output):
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects an existing result; failures remove only our new file.
    with output.open('xb') as stream:
        try:
            with ZipFile(stream,'w',compression=ZIP_STORED) as archive: yield archive
        except BaseException:
            stream.close(); output.unlink(missing_ok=True)
            raise


def create_request(media, options, output):
    media = Path(media); output = Path(output)
    if not isinstance(options,JobOptions): raise TypeError('Expected JobOptions')
    suffix = media.suffix.lower()
    if suffix not in {'.png','.jpg','.jpeg','.mp4','.avi','.mov','.mkv','.webm','.webp'}:
        raise ValueError('Unsupported media extension')
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest = {'protocol':PROTOCOL, 'type':'request', 'job_id':uuid.uuid4().hex, 'options':asdict(options)}
    with _archive(output) as archive:
        manifest['input'] = _write_file(archive, media, 'input/media'+suffix)
        archive.writestr('request.json',_json(manifest))
    return manifest


def read_request(path, output):
    with ZipFile(path) as archive:
        names = _inspect(archive)
        if 'request.json' not in names: raise ValueError('Missing request manifest')
        manifest = _manifest(archive,'request.json'); _identity(manifest,'request')
        JobOptions(**manifest['options'])
        descriptor = manifest['input']
        if not _name(descriptor['path']).startswith('input/'): raise ValueError('Invalid input location')
        if names != {'request.json',descriptor['path']}: raise ValueError('Unexpected request files')
        _check_file(archive,descriptor)
        root = _extract(archive,names,output)
    return manifest, root/descriptor['path']


def create_result(job_id, input_sha256, result, artifacts, output):
    """artifacts maps safe relative names to explicit output files, never whole workspace."""
    manifest = {'protocol':PROTOCOL,'type':'result','job_id':job_id,'input_sha256':input_sha256,
                'result':result, 'files':[]}
    _identity(manifest,'result')
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    if len(artifacts) >= MAX_FILES: raise ValueError('Too many result files')
    if not re.fullmatch('[0-9a-f]{64}',input_sha256): raise ValueError('Invalid input checksum')
    with _archive(output) as archive:
        for name, source in artifacts.items():
            manifest['files'].append(_write_file(archive, source, 'artifacts/'+_name(name)))
            if sum(d['size'] for d in manifest['files']) > MAX_BYTES-MAX_MANIFEST: raise ValueError('Result exceeds size limit')
        archive.writestr('result.json',_json(manifest))
    return manifest


def read_result(path, output, expected_job_id=None, expected_input_sha256=None):
    with ZipFile(path) as archive:
        names = _inspect(archive)
        if 'result.json' not in names: raise ValueError('Missing result manifest')
        manifest = _manifest(archive,'result.json'); _identity(manifest,'result')
        if expected_job_id and manifest['job_id'] != expected_job_id: raise ValueError('Result belongs to a different job')
        if expected_input_sha256 and manifest['input_sha256'] != expected_input_sha256: raise ValueError('Result belongs to different input media')
        descriptors = manifest['files']
        paths = [d['path'] for d in descriptors]
        if len(set(paths)) != len(paths): raise ValueError('Duplicate result file')
        if any(not _name(p).startswith('artifacts/') for p in paths): raise ValueError('Invalid artifact location')
        if names != {'result.json',*paths}: raise ValueError('Unexpected result files')
        for descriptor in descriptors: _check_file(archive,descriptor)
        root = _extract(archive,names,output)
    return manifest, root/'artifacts'
