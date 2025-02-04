import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from centroidtracker import CentroidTracker
import pymongo
from datetime import datetime

# Connect to MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB connection string
db = client["test"]  # Update with your MongoDB database name
collection = db["metrics"]  # Update with your MongoDB collection name
# Get today's date in YYYY-MM-DD format
today_date = datetime.now().strftime("%Y-%m-%d  %H:%M")
document_id = "Tiban_" + today_date


class VideoStream:
    """Camera object that controls video streaming from a video file"""

    def __init__(self, video_path, resolution=(1920, 1080), framerate=30):
        # Initialize the video stream with the video file
        self.stream = cv2.VideoCapture(video_path)  # Open the video file
        if not self.stream.isOpened():
            print("Error: Couldn't open video file.")
            sys.exit(1)

        # Get the original resolution of the video
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set resolution if needed (optional)
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the video is stopped, stop the thread
            if self.stopped:
                # Close video resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the video stream
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                print("End of video reached.")
                self.stopped = True
                break

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the video stream should be stopped
        self.stopped = True


# compare the co-ordinates for dictionaries of interest
def DictDiff(dict1, dict2):
    dict3 = {**dict1}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = np.subtract(dict2[key], dict1[key])
    return dict3


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.80)
parser.add_argument('--resolution',
                    help='Desired video resolution in WxH. If the resolution is not supported by the video, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter

    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter

    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del (labels[0])

# Load the Tensorflow Lite model.
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if ('StatefulPartitionedCall' in outname):  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Set the video input path
video_path = 'video/vlogpyth.mp4'  # Update this with your video path

# Initialize video stream from the file
videostream = VideoStream(video_path=video_path, resolution=(imW, imH), framerate=60).start()
time.sleep(1)
ct = CentroidTracker(expiration_time=1.0)  # Set expiration to 1 second

objects = {}
old_objects = {}
prev_total = 0
unique_object = set()
object_counts = {
    'look': 0,
    'no look': 0,
    'car': 0,
    'bike': 0,
}
prev_counts = {}
detected_objects = set()

while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    old_objects.update(objects)

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    # Reset object count dictionary for the current frame
    rects = []

    # Loop over all detections
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            box = np.array([xmin, ymin, xmax, ymax])

            rects.append(box.astype("int"))

            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (10, 255, 0), 2)

            label = '%s: %d%%' % (labels[int(classes[i])], int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # update the centroid for the objects
    objects = ct.update(rects)

    # calculate the difference between this and the previous frame
    x = DictDiff(objects, old_objects)

    for (objectID, centroid) in objects.items():
        if objectID not in detected_objects:
            detected_objects.add(objectID)
            object_name = label.split(':')[0].strip()
            print(object_name)
            if object_name == 'look':
                object_counts['look'] += 1
            elif object_name == 'no look':
                object_counts['no look'] += 1
            elif object_name == 'car':
                object_counts['car'] += 1
            elif object_name == 'bike':
                object_counts['bike'] += 1

        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    for object_name, count in object_counts.items():
        text = "Count {}: {}".format(object_name, count)
        cv2.putText(frame, text, (10, 60 + 30 * list(object_counts.keys()).index(object_name)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    collection.replace_one({"_id": document_id}, object_counts, upsert=True)

    # Draw framerate in corner of frame
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255),
                2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close all windows
cv2.destroyAllWindows()
videostream.stop()
