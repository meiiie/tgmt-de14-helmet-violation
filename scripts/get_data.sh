#!/usr/bin/env bash
# Tải dữ liệu đánh giá: annotation HELMET (Kaggle mirror hozngvan/helmet-detection, cần KAGGLE_API_TOKEN)
# + frame 640x640 (HF eiglesiasr/nohelmetnet, công khai) cho các clip test/train đã chọn.
set -e; cd "$(dirname "$0")/.."
mkdir -p data/helmet/ann data/helmet/frames
TEST="${TEST_CLIPS:-Bago_highway_10 Bago_highway_20 Bago_rural_3 Bago_rural_49 Bago_urban_3 Bago_urban_50}"
TRAIN="${TRAIN_CLIPS:-Bago_highway_11 Bago_highway_12 Bago_highway_13 Bago_rural_16 Bago_rural_2 Bago_rural_23 Bago_urban_12 Bago_urban_13 Bago_urban_15 Mandalay_1_100}"
echo $TRAIN | tr ' ' '\n' > data/helmet/train_clips.txt
kaggle datasets download hozngvan/helmet-detection -f data_split.csv -p data/helmet >/dev/null
for c in $TEST $TRAIN; do kaggle datasets download hozngvan/helmet-detection -f annotation/annotation/$c.csv -p data/helmet/ann >/dev/null; sleep 1; done
curl -s https://huggingface.co/api/datasets/eiglesiasr/nohelmetnet | python3 -c "import json,sys;[print(s['rfilename']) for s in json.load(sys.stdin)['siblings'] if s['rfilename'].startswith('images/')]" > data/nohelmetnet_files.txt
dl(){ c=$1; step=$2; l=$(echo $c | tr A-Z a-z); mkdir -p data/helmet/frames/$c
  grep "/${l}_[0-9]*.jpg" data/nohelmetnet_files.txt | awk -v s=$step 'NR%s==1||s==1' | xargs -P 16 -I{} sh -c 'curl -sL -o data/helmet/frames/'$c'/$(basename {} | sed "s/.*_//") https://huggingface.co/datasets/eiglesiasr/nohelmetnet/resolve/main/{}'; }
for c in $TEST; do dl $c 1; done
for c in $TRAIN; do dl $c 5; done
echo "data ok"
