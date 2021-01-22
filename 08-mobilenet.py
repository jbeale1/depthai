#!/usr/bin/env python3

# https://github.com/luxonis/depthai-python/tree/gen2_develop/examples
# mods J.Beale 2021-01-21

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
# ----------------
minConfidence = 0.75 # nn confidence theshold

# available mobilenet labels  9=chair 11=table 15=person
labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# Get argument first
mobilenet_path = str((Path(__file__).parent / Path('models/mobilenet-ssd.blob')).resolve().absolute())
if len(sys.argv) > 1:
    mobilenet_path = sys.argv[1]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.setImageOrientation(dai.CameraImageOrientation.NORMAL) # USB up


# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(mobilenet_path)
cam_rgb.preview.link(detection_nn.input)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

frame = None
bboxes = []


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


while True:
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_rgb = q_rgb.tryGet()
    in_nn = q_nn.tryGet()

    if in_rgb is not None:
        # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_nn is not None:
        # Detection values: [image_id, label, confidence, x_min, y_min, x_max, y_max]
        # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
        bboxes = np.array(in_nn.getFirstLayerFp16())
        # take only the results before -1 digit
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        # transform the 1D array into Nx7 matrix
        bboxes = bboxes.reshape((bboxes.size // 7, 7))

        # filter out the results which confidence less than a defined threshold
        # also, select out just the x1,y1,x2,y2 coords
        bboxes = bboxes[bboxes[:, 1] == 15]  # just people
        bboxes = bboxes[bboxes[:, 2] > minConfidence][:, 3:7]

    if frame is not None:
        # if the frame is available, draw bounding boxes on it and show the frame
        for raw_bbox in bboxes:
            bbox = frame_norm(frame, raw_bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("rgb", frame)

    if cv2.waitKey(1) == ord('q'):
        break
