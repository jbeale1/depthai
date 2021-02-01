#!/usr/bin/env python3

# Read in video file, detect items, draw bounding boxes
# description of original video-input mobilenet code:
# https://docs.luxonis.com/projects/api/en/gen2_develop/samples/17_video_mobilenet/ 

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np

# ----------------
# infile="./construction_vest.mp4"
infile="./walk-720p.mp4"


minConfidence = 0.75 # nn confidence theshold

# available mobilenet labels  9=chair 11=table 15=person
labels = ["background", "aeroplane", "bicycle", "bird", "boat", 
  "bottle",   "bus", "car", "cat", "chair", "cow", "diningtable", 
  "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
   "sofa", "train", "tvmonitor"]


# Get argument first
mobilenet_path = str((Path(__file__).parent / Path('models/mobilenet.blob')).resolve().absolute())
if len(sys.argv) > 1:
    mobilenet_path = sys.argv[1]

# Start defining a pipeline
pipeline = dai.Pipeline()


# Create neural network input
xin_nn = pipeline.createXLinkIn()
xin_nn.setStreamName("in_nn")

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(mobilenet_path)
xin_nn.out.link(detection_nn.input)

# Create output
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_in = device.getInputQueue(name="in_nn")
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

frame = None
bboxes = []


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


cap = cv2.VideoCapture(str(Path(infile).resolve().absolute()))
while cap.isOpened():
    read_correctly, frame = cap.read()
    if not read_correctly:
        break

    nn_data = dai.NNData()
    nn_data.setLayer("data", to_planar(frame, (300, 300)))
    q_in.send(nn_data)


    in_nn = q_nn.tryGet()

    if in_nn is not None:
        # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
        bboxes = np.array(in_nn.getFirstLayerFp16())
        # take only the results before -1 digit
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        # transform the 1D array into Nx7 matrix
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        
        
        # filter out the results which confidence less than a defined threshold
        # also, select out just the x1,y1,x2,y2 coords
        bboxes = bboxes[bboxes[:, 1] == 15]  # just people
        bboxes = bboxes[bboxes[:, 2] > minConfidence]
        for box in bboxes:
            print("%04.2f" % (box[2]))  # DEBUG  show confidence level
        bboxes = bboxes[:, 3:7]  # select just the x1,y1 x2,y2 coords
                

    if frame is not None:
        # if the frame is available, draw bounding boxes on it and show the frame
        for raw_bbox in bboxes:
            bbox = frame_norm(frame, raw_bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("rgb", frame)

    if cv2.waitKey(1) == ord('q'):
        break
        
