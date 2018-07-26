import cv2 as cv
import numpy as np
import sys

class yolo_detector:

    def __init__(self, model_file, config_file, class_file, threshold):
        self.model_file = model_file
        self.config_file = config_file
        self.class_file = class_file
        self.threshold = threshold
        self.frame = 0
        self.load_model_file()

    def load_model_file(self):
        with open(self.class_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        self.net = cv.dnn.readNetFromDarknet(self.config_file, self.model_file)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


    def predict(self, frame):
        # convert frame into darknet format and run model
        blob = cv.dnn.blobFromImage(cv.resize(frame, (416, 416)), 0.003921, (416, 416), (0,0,0), swapRB=True,  crop=False)
        self.net.setInput(blob)
        raw_predictions = self.net.forward()

        # filter data
        result = []
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]
        for prediction in raw_predictions:
            confidences = prediction[5:]  # first 4 are location, then probabilties
            classId = np.argmax(confidences)
            confidence = float(confidences[classId])
            if confidence > self.threshold:
                w = int(prediction[2] * frame_w)
                h = int(prediction[3] * frame_h)
                x = int(prediction[0] * frame_w - (w / 2))
                y = int(prediction[1] * frame_h - (h / 2))
                label = str(self.classes[classId])
                result.extend([self.frame, label, confidence, x, y, w, h])
        self.frame += 1
        return np.array(result).reshape((-1, 7))
