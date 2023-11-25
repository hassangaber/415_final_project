#!/usr/bin/env/ python3
import cv2
import numpy as np
from typing import Tuple, List

MODEL=cv2.dnn.readNetFromDarknet(cfgFile= "config/yolov3.cfg",darknetModel="config/yolov3.weights")
MODEL.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

def yolo_v3_pipeline(img:np.ndarray)->Tuple[np.ndarray,List[str],List[float]]:
    classes = open('config/coco.names').read().strip().split('\n')
    np.random.seed(1337)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False) # blob for darknet
    MODEL.setInput(blob)
    ln = MODEL.getUnconnectedOutLayersNames()
    outs = MODEL.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    # Detect objects from pre-defined list and extract likelihoods
    for output in outs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Draw the boxes alongside classname, likelihoods on the image
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    labels=[]
    prob=[]

    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in colors[classIDs[i]]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
        labels.append(classes[classIDs[i]])
        prob.append(confidences[i])
        print(f'P({classes[classIDs[i]]}|Image) = {confidences[i]}')

        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # cv2.imshow(img)

    return (img,labels,prob)
