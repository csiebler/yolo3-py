# YOLOv3 Python to CSV

A simple tool for running video or camera data through [YOLOv3](https://pjreddie.com/darknet/yolo/) and outputting predictions to a CSV file.

## Usage

Firstly, load weights for YOLO:

```
$ ./prepare.sh
```

Next, feed your data into the script using regular `YOLOv3`:

```
python3 detect_to_json.py \
  --model yolov3.weights \
  --config model/yolov3.cfg \
  --classes model/coco_classes.txt \
  --input_video test.mp4 \
  --output_video out/sample_output.mp4 \
  --output_csv test.csv
```

Alternatively you can use `YOLOv3-tiny`:

```
python3 detect_to_json.py \
  --model yolov3-tiny.weights \
  --config model/yolov3-tiny.cfg \
  --classes model/coco_classes.txt \
  --input_video test.mp4 \
  --output_video out/sample_output.mp4 \
  --output_csv test.csv
```

If output to CSV is set, the following format will be outputted:

```
# frame,label,confidence,x,y,w,h
0,person,0.734311044216156,858,311,18,29
0,person,0.7461023330688477,714,355,17,26
0,person,0.46452274918556213,960,349,19,34
...
```