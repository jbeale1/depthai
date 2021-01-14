#!/usr/bin/env python3

# This file runs on 'OAK-1' hardware from OpenCV DepthAI 
# highly experimental code by J.Beale 2021-01-13

# based on https://docs.luxonis.com/en/latest/pages/samples/object_tracker/
# create sh12cmx12NCE1 blob with: 
#  "python3 depthai_demo.py -sh 12 -cmx 12 -nce 1"

import cv2
import depthai
import datetime  # enable timestamp with time/date 


device = depthai.Device('', False)

p = device.create_pipeline(config={
    "streams": ["previewout", "object_tracker"],
    "ai": {
        #blob compiled for maximum 12 shaves
        "blob_file": "/home/pi/depthai/depthai/resources/nn/mobilenet-ssd/mobilenet-ssd.blob.sh12cmx12NCE1",
        "blob_file_config": "/home/pi/depthai/depthai/resources/nn/mobilenet-ssd/mobilenet-ssd.json",
        "shaves" : 12,
        "cmx_slices" : 12,
        "NN_engines" : 1,
    },
    'ot': {
        'max_tracklets': 20,
        'confidence_threshold': 0.9,
    },
})

if p is None:
    raise RuntimeError("Error initializing pipelne")

labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
tracklets = None

idleTime = 0 # how many cycles since we've tracked something interesting
startTime = datetime.datetime.now()

while True:
    for packet in p.get_available_data_packets():
        if packet.stream_name == 'object_tracker':
            tracklets = packet.getObjectTracker()
        elif packet.stream_name == 'previewout':
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            traklets_nr = tracklets.getNrTracklets() if tracklets is not None else 0

            idleTime += 1  # assume we didn't see anything this cycle...

            for i in range(traklets_nr):
              tracklet = tracklets.getTracklet(i)
              tlabel = labels[tracklet.getLabel()]
              status = tracklet.getStatus()  # TRACKED / LOST
              if (tlabel == "person") and (status != "LOST"):
                left = tracklet.getLeftCoord()
                top = tracklet.getTopCoord()
                right = tracklet.getRightCoord()
                bottom = tracklet.getBottomCoord()
                area = (right - left) * (bottom - top)  # pixels inside box
                cx = (right + left)/2
                cy = (top + bottom)/2

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0))

                middle_pt = (int(left + (right - left) / 2), int(top + (bottom - top) / 2))
                cv2.circle(frame, middle_pt, 0, (255, 0, 0), -1)
                cv2.putText(frame, f"ID {tracklet.getId()}", middle_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.putText(frame, tlabel, (left, bottom - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, status, (left, bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # print("%d: %s (%d,%d) %d" % (i,tlabel,cx,cy,area),end="") # JPB DEBUG
                if (idleTime == 1):  # we were tracking motion last cycle too
                    # print("")
                    pass
                else:
                    startTime = datetime.datetime.now()
                    print("Start: %d %s %s" % (i,tlabel,startTime)) # new motion seen after some gap
                    
                idleTime = 0  # if we did see something, reset idle time

            cv2.imshow('previewout', frame)
            if (idleTime == 1):
                endTime = datetime.datetime.now()
                duration = (endTime - startTime).total_seconds() # seconds
                print("          End: %d" % duration)


    if cv2.waitKey(1) == ord('q'):
        break

del p
del device
