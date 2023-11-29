#!/usr/bin/env/ python3
import numpy as np
import imutils
import time
from scipy import spatial
import cv2
import os
#from input_retrieval import *

#All these classes will be counted as 'vehicles'
list_of_vehicles = ["bicycle","car","motorbike","bus"]
# Setting the threshold for the number of frames to search a vehicle for
FRAMES_BEFORE_CURRENT = 30
inputWidth, inputHeight = 256, 256

#Parse command line arguments and extract the values required
# LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
#   preDefinedConfidence, preDefinedThreshold, USE_GPU= parseCommandLineArguments()
pedestrian_positions = {}
vehicle_position_history = {}
distance_threshold = 10  # Adjust the value as needed
unique_id_counter = 0  # Initialize a global counter for unique IDs

def checkBoundingBoxDistance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    center_x1, center_y1 = x1 + (w1 // 2), y1 + (h1 // 2)
    center_x2, center_y2 = x2 + (w2 // 2), y2 + (h2 // 2)

    # Calculate the Euclidean distance between the centers of the two bounding boxes
    distance = ((center_x1 - center_x2) * 2 + (center_y1 - center_y2) * 2) ** 0.5

    # Check if the distance is below the threshold
    if distance < distance_threshold:
        return True
    else:
        return False

LABELS=open('config/coco.names').read().strip().split('\n')
weightsPath='config/yolov3.weights'
configPath='config/yolov3.cfg'
inputVideoPath='data/st-catherines_drive.mp4'
# inputVideoPath='data/mcgill_drive.mp4'
outputVideoPath='data/out2.avi'
preDefinedConfidence=0.4
preDefinedThreshold=0.15
USE_GPU=0

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# PURPOSE: Displays the vehicle count on the top-left corner of the frame
# PARAMETERS: Frame on which the count is displayed, the count number of vehicles 
# RETURN: N/A
def displayVehicleCount(frame, vehicle_count):
    cv2.putText(
        frame, #Image
        'Detected Vehicles: ' + str(vehicle_count), #Label
        (40, 60), #Position
        cv2.FONT_HERSHEY_SIMPLEX, #Font
        2, #Size
        (0, 0, 255), #Color
        2, #Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        )

def displayParkedVehicleCount(frame, parked_vehicle_count):
    cv2.putText(
        frame, #Image
        'Detected Parked Vehicles: ' + str(parked_vehicle_count), #Label
        (40, 120), #Position
        cv2.FONT_HERSHEY_SIMPLEX, #Font
        2, #Size
        (0, 0, 255), #Color
        2, #Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        )

def displayPedestrianCount(frame, pedestrian_count):
    cv2.putText(
        frame, #Image
        'Detected Pedestrians: ' + str(pedestrian_count), #Label
        (40, 200), #Position
        cv2.FONT_HERSHEY_SIMPLEX, #Font
        2, #Size
        (0, 0, 255), #Color
        2, #Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        )
    
# PURPOSE: Determining if the box-mid point cross the line or are within the range of 5 units
# from the line
# PARAMETERS: X Mid-Point of the box, Y mid-point of the box, Coordinates of the line 
# RETURN: 
# - True if the midpoint of the box overlaps with the line within a threshold of 5 units 
# - False if the midpoint of the box lies outside the line and threshold
def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
    x1_line, y1_line, x2_line, y2_line = line_coordinates #Unpacking

    if (x_mid_point >= x1_line and x_mid_point <= x2_line+5) and\
        (y_mid_point >= y1_line and y_mid_point <= y2_line+5):
        return True
    return False

# PURPOSE: Displaying the FPS of the detected video
# PARAMETERS: Start time of the frame, number of frames within the same second
# RETURN: New start time, new number of frames 
def displayFPS(start_time, num_frames):
    current_time = int(time.time())
    if(current_time > start_time):
        os.system('clear') # Equivalent of CTRL+L on the terminal
        print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames

# PURPOSE: Draw all the detection boxes with a green dot at the center
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            #Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)

# PURPOSE: Initializing the video writer with the output video path and the same number
# of fps, width and height as the source video 
# PARAMETERS: Width of the source video, Height of the source video, the video stream
# RETURN: The initialized video writer
def initializeVideoWriter(video_width, video_height, videoStream):
    # Getting the fps of the source video
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps, (video_width, video_height), True)

def find_nearest_key(dictionary, target_key, max_distance=20):
    nearest_key = None
    smallest_distance = np.inf
    for key in dictionary.keys():
        distance = np.linalg.norm(np.array(key) - np.array(target_key))
        if distance < smallest_distance:
            smallest_distance = distance
            nearest_key = key
    if smallest_distance <= max_distance:
        return nearest_key
    else:
        return None
    
def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, width, height = current_box
    dist = np.inf  # Initializing the minimum distance
    found_id = None  # Initialize the variable to store the found ID

    # Iterating through all the previous frames
    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:  # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
        if temp_dist < dist:
            dist = temp_dist
            coord = coordinate_list[index[0]]
            found_id = previous_frame_detections[i][coord]  # Retrieve the ID

    if dist <= (max(width, height) / 2) and found_id is not None:
        current_detections[(centerX, centerY)] = found_id  # Update the current detection with the found ID
        return True, found_id
    else:
        return False, None

# Global sets to keep track of counted IDs
counted_vehicle_ids = set()
counted_parked_vehicle_ids = set()
counted_people_ids = set()

def checkBoundingBoxIntersection(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # Calculate the coordinates of the top-left and bottom-right corners
    x1_tl, y1_tl, x1_br, y1_br = x1, y1, x1 + w1, y1 + h1
    x2_tl, y2_tl, x2_br, y2_br = x2, y2, x2 + w2, y2 + h2
    # Check for intersection
    if (
        x1_tl < x2_br
        and x1_br > x2_tl
        and y1_tl < y2_br
        and y1_br > y2_tl
    ):
        return True
    else:
        return False

def count_parked_vehicles(idxs, boxes, classIDs, parked_vehicle_count, previous_frame_detections, frame, movement_threshold=20, required_stationary_frames=10, debug=False):
    current_detections = {}
    global unique_id_counter  # Ensure to declare the global counter

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centerX = x + (w // 2)
            centerY = y + (h // 2)

            if LABELS[classIDs[i]] in list_of_vehicles:
                found_in_previous, ID = boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)

                if not found_in_previous:
                    ID = unique_id_counter
                    unique_id_counter += 1
                current_detections[(centerX, centerY)] = ID

                if ID not in vehicle_position_history:
                    vehicle_position_history[ID] = []

                vehicle_position_history[ID].append((centerX, centerY))

                # Only consider the last 'required_stationary_frames' positions
                if len(vehicle_position_history[ID]) > required_stationary_frames:
                    vehicle_position_history[ID].pop(0)  # Remove the oldest position

                # Determine if the vehicle is stationary
                is_stationary = True
                for pos in vehicle_position_history[ID]:
                    if np.linalg.norm(np.array(pos) - np.array(vehicle_position_history[ID][0])) > movement_threshold:
                        is_stationary = False
                        break

                if is_stationary and ID not in counted_parked_vehicle_ids:
                    counted_parked_vehicle_ids.add(ID)
                    parked_vehicle_count += 1

                if debug:
                    cv2.putText(frame, str(ID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
                    for pos in vehicle_position_history[ID]:
                        cv2.circle(frame, pos, 2, [0, 255, 0], -1)  # Draw green dots for position history

    return parked_vehicle_count

# def count_vehicles(idxs, boxes, classIDs, vehicle_count, people_count, previous_frame_detections, frame):
#     current_detections = {}
#     global unique_id_counter  # Assuming this is declared globally

#     if len(idxs) > 0:
#         for i in idxs.flatten():
#             (x, y) = (boxes[i][0], boxes[i][1])
#             (w, h) = (boxes[i][2], boxes[i][3])
#             centerX = x + (w // 2)
#             centerY = y + (h // 2)

#             # Check if the detected object is a vehicle or person
#             if LABELS[classIDs[i]] in list_of_vehicles:
#                 found_in_previous, found_id = boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)
                
#                 if not found_in_previous:
#                     # Assign a new ID if this is a new detection
#                     unique_id_counter += 1
#                     current_detections[(centerX, centerY)] = unique_id_counter
#                     counted_vehicle_ids.add(unique_id_counter)
#                     vehicle_count += 1
#                 else:
#                     # Use existing ID for this detection
#                     current_detections[(centerX, centerY)] = found_id if found_id is not None else unique_id_counter
                
#                 cv2.putText(frame, f"Vehicle ID: {current_detections[(centerX, centerY)]}", (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
                        
#             elif LABELS[classIDs[i]] == 'person':
#                 # Similar logic for pedestrians
#                 found_in_previous, found_id = boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)
#                 if not found_in_previous:
#                     people_count += 1
#                     current_detections[(centerX, centerY)] = people_count
#                 else:
#                     current_detections[(centerX, centerY)] = found_id if found_id is not None else people_count

#                 cv2.putText(frame, f"Pedestrian ID: {current_detections[(centerX, centerY)]}", (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

#     return vehicle_count, people_count, current_detections


def count_vehicles(idxs, boxes, classIDs, vehicle_count, people_count, previous_frame_detections, frame):
    current_detections = {}

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centerX = x + (w // 2)
            centerY = y + (h // 2)

            # Check if the detected object is a vehicle or person
            if LABELS[classIDs[i]] in list_of_vehicles:
                if not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)[0]:
                    # Assign a new ID if this is a new detection
                    ID = vehicle_count
                    current_detections[(centerX, centerY)] = ID
                    if ID not in counted_vehicle_ids:
                        counted_vehicle_ids.add(ID)
                        vehicle_count += 1
                else:
                    # Use existing ID for this detection
                    ID = current_detections.get((centerX, centerY), vehicle_count)
                cv2.putText(frame, str(ID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
                        
            elif LABELS[classIDs[i]] == 'person':
                if not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)[0]:
        # Assign a new ID if this is a new detection
                    ID = people_count
                    current_detections[(centerX, centerY)] = ID

                    if ID not in counted_people_ids:
                        counted_people_ids.add(ID)
                        people_count += 1
                else:
                    # Use existing ID for this detection
                    ID = current_detections.get((centerX, centerY), people_count)

                # Check if the pedestrian ID already exists and if their bounding box intersects
                if ID in pedestrian_positions:
                    prev_box = pedestrian_positions[ID]
                    current_box = (centerX, centerY, w, h)

                    # Check if the bounding boxes intersect
                    if checkBoundingBoxIntersection(prev_box, current_box):
                        continue

                    # Update the pedestrian's position
                    pedestrian_positions[ID] = (centerX, centerY)
                    cv2.putText(frame, str(ID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
    return vehicle_count, people_count, current_detections



# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the output layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Using GPU if flag is passed
if USE_GPU:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Specifying coordinates for a default line 
x1_line = 0
y1_line = video_height//2
x2_line = video_width
y2_line = video_height//2

#Initialization
previous_frame_detections = [{} for _ in range(FRAMES_BEFORE_CURRENT)]

# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count, people_count, parked_vehicle_count = 0, 0, 0,0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())
# loop over frames from the video file stream
while True:
    print("================NEW FRAME================")
    num_frames+= 1
    print("FRAME:\t", num_frames)
    # Initialization for each iteration
    boxes, confidences, classIDs = [], [], [] 
    vehicle_crossed_line_flag = False 

    #Calculating fps each second
    start_time, num_frames = displayFPS(start_time, num_frames)
    # read the next frame from the file
    (grabbed, frame) = videoStream.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for i, detection in enumerate(output):
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > preDefinedConfidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                            
                #Printing the info of the detection
                #print('\nName:\t', LABELS[classID],
                    #'\t|\tBOX:\t', x,y)

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,preDefinedThreshold)

    # Draw detection box 
    drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

    vehicle_count, people_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count, people_count, previous_frame_detections, frame)
    parked_vehicle_count = count_parked_vehicles(idxs, boxes, classIDs, parked_vehicle_count, previous_frame_detections, frame)
    # Display Vehicle Count if a vehicle has passed the line 
    
    displayPedestrianCount(frame, people_count)
    displayParkedVehicleCount(frame, parked_vehicle_count)
    displayVehicleCount(frame, vehicle_count)

    # write the output frame to disk
    writer.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
    
    # Updating with the current frame detections
    previous_frame_detections.pop(0) #Removing the first frame from the list
    # previous_frame_detections.append(spatial.KDTree(current_detections))
    previous_frame_detections.append(current_detections)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
videoStream.release()
