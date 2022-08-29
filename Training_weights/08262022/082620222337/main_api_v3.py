#!/usr/bin/env python3
"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
We add parsing from JSON files that contain configuration
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import blobconverter
#from time import monotonic

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='yolov4_tiny_coco_416x416', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='json/yolov4-tiny.json', type=str)
parser.add_argument('-t', '--threshold', help="Minimum distance object parses to be considered a real movement",
                    default=0.25, type=float)
parser.add_argument('-spi', '--spi', action='store_true', 
                    default=False, help="Send tracklets to the MCU via SPI")
#parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI RGB camera for inference (conflicts with -vid)")

args = parser.parse_args()

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 2, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.color, 2, self.line_type)

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 6, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(W, H)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)



# List of labels
labelMap = ["vehicle","person"]

# Create and configure the object tracker
objectTracker = pipeline.create(dai.node.ObjectTracker)
objectTracker.setDetectionLabelsToTrack([0])  # Track vehicle
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# Take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)



# Linking
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
# Link detection networks outputs to the object tracker
detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)

detectionNetwork.out.link(objectTracker.inputDetections)
detectionNetwork.out.link(nnOut.input)

script = pipeline.create(dai.node.Script)
objectTracker.out.link(script.inputs['tracklets'])

with open("script.py", "r") as f:
    s = f.read()
    s = s.replace("THRESH_DIST_DELTA", str(args.threshold))
    script.setScript(s)

# Send tracklets to the host
trackerOut = pipeline.create(dai.node.XLinkOut)
trackerOut.setStreamName("out")
script.outputs['out'].link(trackerOut.input)

# Send send RGB preview frames to the host
xlinkOut = pipeline.create(dai.node.XLinkOut)
xlinkOut.setStreamName("preview")
objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

if args.spi:
    # Send tracklets via SPI to the MCU
    spiOut = pipeline.create(dai.node.SPIOut)
    spiOut.setStreamName("tracklets")
    spiOut.setBusId(0)
    objectTracker.out.link(spiOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    outQ = device.getOutputQueue("out")
    previewQ = device.getOutputQueue("preview")

    counters = None
    frames = None
    text = TextHelper()
    
    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame, detections):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame , detection.xmax, detection.ymin, detection.ymax
        cv2.imshow(name, frame)
    checkd = 0
    
    def update():
        global counters, frames, text
        if previewQ.has():
            frames = previewQ.get().getCvFrame()

        if outQ.has():
            jsonText = str(outQ.get().getData(), 'utf-8')
            counters = json.loads(jsonText)
            print(counters)

        if counters is not None:
            text.putText(frames, f"Car In: {counters['In']}, Box: {counters['DelX']}, Angle: {counters['Theta']}", (3, 30))
        if frames is not None:
            cv2.imshow("frames", frames)
    
    while True:
        update()
                
        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb is not None:
            #print('R1')
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            #print('R2')
            detections = inDet.detections
            c_old = counter
            counter += 1
            #for detection in detections:
            

        if frame is not None:
            #print('R3')
            displayFrame("rgb", frame, detections)

        if cv2.waitKey(1) == ord('q'):
            #print('R4')
            break