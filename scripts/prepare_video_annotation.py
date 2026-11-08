"""Prepare a beginner-friendly local labeling page, without model inference."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from de14.annotations import prepare_package


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path);parser.add_argument('--out',type=Path,default=ROOT/'outputs/annotations/1102')
    parser.add_argument('--stride',type=int,default=10)
    parser.add_argument('--audit',type=Path,help='Reuse source metadata from matching video audit; requires --provenance')
    parser.add_argument('--provenance',type=Path)
    parser.add_argument('--serve',action='store_true');parser.add_argument('--port',type=int,default=7862)
    args=parser.parse_args();info=digest=None
    if args.audit:
        if not args.provenance: parser.error('--audit requires --provenance to bind source SHA256')
        info=json.loads(args.audit.read_text(encoding='utf-8'))['media']['source']
        digest=json.loads(args.provenance.read_text(encoding='utf-8'))['source_sha256']
    result=prepare_package(args.video,args.out,args.stride,info,digest)
    print(json.dumps({'frames':len(result['frames']),'out':str(args.out.resolve()),'source_sha256':result['source']['sha256']}),flush=True)
    if args.serve:
        server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(args.out.resolve())))
        print(f'Annotation page: http://127.0.0.1:{args.port}/',flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally: server.server_close()


if __name__=='__main__':main()
