#!/bin/sh

python3 detect_to_csv.py \
  --model yolov3.weights \
  --config model/yolov3.cfg \
  --classes model/coco_classes.txt \
  --input_video test.mp4 \
  --output_csv test.csv

# python3 detect_to_csv.py \
#   --model yolov3.weights \
#   --config model/yolov3.cfg \
#   --classes model/coco_classes.txt \
#   --input_video test.mp4 \
#   --output_video out/sample_output.mp4 \
#   --output_csv test.csv
