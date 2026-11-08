"""Run the shared inference backend on an NVIDIA host allowing API services."""
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='0.0.0.0');parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--jobs-dir',type=Path,default=ROOT/'outputs/gpu_api_jobs')
    args=parser.parse_args()
    os.chdir(ROOT)
    from de14.gpu_api import create_app
    import uvicorn
    app=create_app(args.jobs_dir,os.environ.get('HELMET_GPU_API_TOKEN',''))
    uvicorn.run(app,host=args.host,port=args.port,workers=1)

if __name__=='__main__':main()
