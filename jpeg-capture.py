#!/usr/bin/env python3

#./depthai/depthai-experiments/gen2_examples

import subprocess
import depthai as dai

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

# Create an encoder, consuming the frames and encoding them using H.265 encoding
videoEncoder = pipeline.createVideoEncoder()
videoEncoder.setDefaultProfilePreset(3840, 2160, 30, dai.VideoEncoderProperties.Profile.MJPEG)
cam.video.link(videoEncoder.input)

# Create output
videoOut = pipeline.createXLinkOut()
videoOut.setStreamName('jpeg')
videoEncoder.bitstream.link(videoOut.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

q = device.getOutputQueue(name="jpeg", maxSize=30, blocking=True)

i=0

while True:
    jpegPacket = q.get()  # blocking call, will wait until a new data has arrived
    print("Saving image %d" % i)
    fname = ("test_%03d.jpg" % i)
    jpegPacket.getData().tofile(fname)  # appends the packet data to the opened file
    i += 1
