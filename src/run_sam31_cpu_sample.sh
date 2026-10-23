#!/usr/bin/env bash
# CPU: SAM 3.1 rò bộ nhớ giữa các frame trên CPU (OOM ~7GB sau frame 2) -> chạy 1 tiến trình / frame.
# Dùng: bash src/run_sam31_cpu_sample.sh <frames_dir> <out.json> <stride> <max>
set -e; cd "$(dirname "$0")/.."; . .venv/bin/activate; set -a; source /workspace/tgmt/.env; set +a
D=$1; OUT=$2; S=${3:-10}; M=${4:-10}
for i in $(seq 0 $((M-1))); do
  python src/sam31_cache.py "$D" --out "$OUT" --stride "$S" --max $((i+1)) 2>&1 | grep -E "^[0-9]+\.jpg|loaded|Error" || true
done
echo SAMDONE
