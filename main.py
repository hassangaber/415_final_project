#!/usr/bin/env/ python3
import time
import numpy as np
from typing import Tuple, Dict, List
from scipy import spatial

import cv2

"""
Script for object detection of Ste Catherine drive which should be run twice, alternating the `black_out_sides` boolean.
"""
np.random.seed(42)
INPUT_VIDEO='data/st-catherines_drive.mp4'
LABELS=open('config/coco.names').read().strip().split('\n')
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
FRAMES_BEFORE_CURRENT = 30
USE_GPU=0

list_of_vehicles = ["car"]
inputWidth, inputHeight = 256,256
pedestrian_positions = {}
trackers = {}
vehicle_position_history = {}
distance_threshold = 10  # Adjust the value as needed
unique_id_counter = 0  # Initialize a global counter for unique IDs
weightsPath='config/yolov3.weights'
configPath='config/yolov3.cfg'
outputVideoPath='data/out3.avi'
preDefinedConfidence=0.7
preDefinedThreshold=0.5
counted_vehicle_ids = set()
counted_parked_vehicle_ids = set()
counted_people_ids = set()

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if USE_GPU:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

videoStream = cv2.VideoCapture(INPUT_VIDEO)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

x1_line = 0
y1_line = video_height//2
x2_line = video_width
y2_line = video_height//2

previous_frame_detections = [{} for _ in range(FRAMES_BEFORE_CURRENT)]
black_out_sides = True  # Set to False to black out the middle instead
visible_width, visible_height = 300, video_height
num_frames, vehicle_count, people_count, parked_vehicle_count = 0, 0, 0,0



print("[INFO] Running Detection...")


def displayVehicleCount(frame: np.ndarray, vehicle_count: int) -> None:
    """
    Displays the count of detected vehicles on the frame.

    Args:
    frame (np.ndarray): The frame on which the count is displayed.
    vehicle_count (int): The number of vehicles detected.
    """
    cv2.putText(frame, 'Detected Vehicles: ' + str(vehicle_count), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)

def displayPedestrianCount(frame: np.ndarray, pedestrian_count: int) -> None:
    """
    Displays the count of detected pedestrians on the frame.

    Args:
    frame (np.ndarray): The frame on which the count is displayed.
    pedestrian_count (int): The number of pedestrians detected.
    """
    cv2.putText(frame, 'Detected Pedestrians: ' + str(pedestrian_count), (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)

def drawDetectionBoxes(idxs: List[int], boxes: List[Tuple[int, int, int, int]], classIDs: List[int], confidences: List[float], frame: np.ndarray) -> None:
    """
    Draws detection boxes with labels on the frame for each detection.

    Args:
    idxs (List[int]): Indices of detections after applying non-maxima suppression.
    boxes (List[Tuple[int, int, int, int]]): Bounding boxes of detections.
    classIDs (List[int]): Class IDs of detections.
    confidences (List[float]): Confidence scores of detections.
    frame (np.ndarray): The frame on which to draw the detection boxes.
    """
    for i in idxs:
        (x, y, w, h) = boxes[i]
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (x + (w // 2), y + (h // 2)), 2, (0, 0xFF, 0), thickness=2)

def initializeVideoWriter(video_width: int, video_height: int, videoStream: cv2.VideoCapture) -> cv2.VideoWriter:
    """
    Initializes the video writer object with the same FPS, width, and height as the source video.

    Args:
    video_width (int): Width of the source video.
    video_height (int): Height of the source video.
    videoStream (cv2.VideoCapture): Video stream object of the source video.

    Returns:
    cv2.VideoWriter: Initialized video writer object.
    """
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps, (video_width, video_height), True)

def boxInPreviousFrames(previous_frame_detections: Dict[Tuple[int, int, int, int], int], current_box: Tuple[int, int, int, int], current_detections: Dict[Tuple[int, int, int, int], int], k: int) -> Tuple[bool, int]:
    """
    Checks if the current box was present in the previous frames.

    Args:
    previous_frame_detections (Dict[Tuple[int, int, int, int], int]): Detections of the previous frames.
    current_box (Tuple[int, int, int, int]): The coordinates of the current box.
    current_detections (Dict[Tuple[int, int, int, int], int]): Current frame detections.
    k (int): Additional parameter for detection identification.

    Returns:
    Tuple[bool, int]: A tuple indicating if the box was found in previous frames, and the found ID.
    """
    centerX, centerY, width, height = current_box
    dist = np.inf
    found_id = None

    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:
            continue
        temp_dist, index = spatial.KDTree(coordinate_list).query([(k, centerX, centerY)])
        if temp_dist < dist:
            dist = temp_dist
            coord = coordinate_list[index[0]]
            found_id = previous_frame_detections[i][coord]

    if dist <= (max(width, height) / 2) and found_id is not None:
        current_detections[(k, centerX, centerY)] = found_id
        return True, found_id
    else:
        return False, None

def checkBoundingBoxIntersection(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
    """
    Checks if two bounding boxes intersect.

    Args:
    box1 (Tuple[int, int, int, int]): Coordinates of the first bounding box.
    box2 (Tuple[int, int, int, int]): Coordinates of the second bounding box.

    Returns:
    bool: True if the bounding boxes intersect, False otherwise.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if (x1 < x2 + w2) and (x1 + w1 > x2) and (y1 < y2 + h2) and (y1 + h1 > y2):
        return True
    else:
        return False

def count_vehicles(idxs: List[int], boxes: List[Tuple[int, int, int, int]], classIDs: List[int], vehicle_count: int, people_count: int, previous_frame_detections: Dict[Tuple[int, int, int, int], int], frame: np.ndarray) -> Tuple[int, int, Dict[Tuple[int, int, int, int], int]]:
    """
    Counts and tracks vehicles and pedestrians in the given frame.

    Args:
    idxs (List[int]): Indices of detections after applying non-maxima suppression.
    boxes (List[Tuple[int, int, int, int]]): Bounding boxes of detections.
    classIDs (List[int]): Class IDs of detections.
    vehicle_count (int): Current count of vehicles.
    people_count (int): Current count of pedestrians.
    previous_frame_detections (Dict[Tuple[int, int, int, int], int]): Detections from previous frames.
    frame (np.ndarray): The frame on which detections are drawn.

    Returns:
    Tuple[int, int, Dict[Tuple[int, int, int, int], int]]: Updated vehicle and pedestrian counts, and current detections.
    """
    current_detections = {}

    for i in idxs:
        (x, y, w, h) = boxes[i]
        centerX = x + (w // 2)
        centerY = y + (h // 2)
        if LABELS[classIDs[i]] in list_of_vehicles:
            k = 1
            if not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections, k)[0]:
                ID = vehicle_count
                current_detections[(1, centerX, centerY)] = ID
                if ID not in counted_vehicle_ids:
                    counted_vehicle_ids.add(ID)
                    vehicle_count += 1
            else:
                ID = current_detections.get((1, centerX, centerY), vehicle_count)
            cv2.putText(frame, str(ID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
        elif LABELS[classIDs[i]] == 'person':
            k = 0
            if not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections, k)[0]:
                ID = people_count
                current_detections[(0, centerX, centerY)] = ID
                if ID not in counted_people_ids:
                    counted_people_ids.add(ID)
                    people_count += 1
            else:
                ID = current_detections.get((0, centerX, centerY), people_count)
            if ID in pedestrian_positions:
                prev_box = pedestrian_positions[ID]
                current_box = (centerX, centerY, w, h)
                if checkBoundingBoxIntersection(prev_box, current_box):
                    continue
                pedestrian_positions[ID] = (0, centerX, centerY)
                cv2.putText(frame, str(ID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
    return vehicle_count, people_count, current_detections

if __name__ == "__main__":
    writer = initializeVideoWriter(video_width, video_height, videoStream)

    while True:
        print("================NEW FRAME================")
        num_frames+= 1
        print("FRAME:\t", num_frames)
        boxes, confidences, classIDs = [], [], [] 
        vehicle_crossed_line_flag = False 

        (grabbed, frame) = videoStream.read()

        if not grabbed:
            break

        if black_out_sides:
            center_start = (video_width - visible_width) // 2
            center_end = center_start + visible_width
            frame[:, :center_start] = 0  # Black out the left side
            frame[:, center_end-50:] = 0  # Black out the right side
        else:
            side_width = (video_width - visible_width) // 2
            frame[:, side_width:side_width + visible_width] = 0  # Black out the center

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        for output in layerOutputs:
            # loop over each of the detections
            for i, detection in enumerate(output):
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > preDefinedConfidence:
                    box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,preDefinedThreshold)
        drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)
        
        vehicle_count, people_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count, people_count, previous_frame_detections, frame)
        
        displayPedestrianCount(frame, people_count)
        displayVehicleCount(frame, vehicle_count)

        # write the output frame to disk
        writer.write(frame)

        if True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   
        
        previous_frame_detections.pop(0) #Removing the first frame from the list
        previous_frame_detections.append(current_detections)

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    videoStream.release()