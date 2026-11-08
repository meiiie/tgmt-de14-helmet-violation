"""Validate user-exported JSON labels without confusing drafts with ground truth."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from de14.annotations import validate_labels,file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels',type=Path);parser.add_argument('--video',type=Path)
    args=parser.parse_args()
    stats=validate_labels(json.loads(args.labels.read_text(encoding='utf-8-sig')),
                          file_sha256(args.video) if args.video else None)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
