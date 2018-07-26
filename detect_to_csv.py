import cv2 as cv
import numpy as np
import argparse
from detector import yolo_detector

def draw_detections(frame, detections):    
    for o in detections:
        print(o)
        text = '%s: %.2f' % (o[1], o[2])
        p1 = (o[3], o[4])
        p2 = (o[3]+o[5], o[4]+o[6])
        cv.rectangle(frame, p1, p2, (0, 255, 0))
        labelSize, baseLine = cv.getTextSize(
            text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(o[4], labelSize[1])
        cv.rectangle(frame, (o[3], o[4] - labelSize[1]),
                     (o[3] + labelSize[0], o[4] + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, text, p1, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


def keep_detecting(detector, input_file, output_video_file, output_csv_file):
    stream = cv.VideoCapture(input_file if input_file else 0)
    window_name = "Detections"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_AUTOSIZE, cv.WINDOW_AUTOSIZE)
    cv.moveWindow(window_name, 16, 16)

    all_detections = np.array([]).reshape((-1, 7))

    if output_video_file:
        _, test_frame = stream.read()
        height = test_frame.shape[0]
        width = test_frame.shape[1]
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_output = cv.VideoWriter(
            output_video_file, fourcc, 20.0, (width, height))

    while stream.isOpened():
        timer = cv.getTickCount()

        # Get next frame
        success, frame = stream.read()
        if not success:
            break

        # Detect objects
        detections = detector.predict(frame)
        print(detections)

        if output_csv_file:
         all_detections = np.vstack((all_detections, detections))

        # Draw predictions and log FPS
        #draw_detections(frame, detections)
        print("FPS: " + str(cv.getTickFrequency() / (cv.getTickCount() - timer)))

        # Display result
        cv.imshow(window_name, frame)

        # Write to output file if required
        if output_video_file:
            video_output.write(frame)

        # Check if user pressed ESC
        k = cv.waitKey(1) & 0xff
        if k == 27:
            break

    stream.release()
    if output_video_file:
        video_output.release()
    cv.destroyAllWindows()

    if output_csv_file:
        np.savetxt(output_csv_file, all_detections, fmt='%s', delimiter=',', header='frame,label,confidence,x,y,w,h')


def main():
    p = argparse.ArgumentParser(
        description='Detect objects in videos and output detections to CSV')
    p.add_argument('--input_video',
                   help='Input video file. Skip to capture from camera')
    p.add_argument('--output_video',
                   help='(Optional) Output video file with detection boxes')
    p.add_argument('--output_csv',
                   help='(Optional) CSV output file')
    p.add_argument('--model', required=True,
                   help='Path to a darknet .weights file containing trained weights')
    p.add_argument('--config', required=True,
                   help='Path to a darknet .cfg file containing network configuration')
    p.add_argument('--classes', required=True,
                   help='Path to text file with class names for detected objects')
    p.add_argument('--threshold', type=float, default=0.35,
                   help='Confidence threshold for detection')
    args = p.parse_args()

    detector = yolo_detector(args.model, args.config,
                             args.classes, args.threshold)
    keep_detecting(detector, args.input_video, args.output_video, args.output_csv)


if __name__ == '__main__':
    main()
