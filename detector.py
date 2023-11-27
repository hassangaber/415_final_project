import numpy as np
import cv2
import time
from scipy import spatial
from typing import List, Dict, Tuple

class Detector:
    def __init__(self, inputVideoPath:str='data/st-catherines_drive.mp4'):
        self.list_of_vehicles = ["bicycle", "car", "motorbike", "bus"]
        self.FRAMES_BEFORE_CURRENT = 30
        self.inputWidth, self.inputHeight = 256, 256

        self.LABELS = open('config/coco.names').read().strip().split('\n')
        self.weightsPath = 'config/yolov3.weights'
        self.configPath = 'config/yolov3.cfg'
        self.inputVideoPath = inputVideoPath
        self.outputVideoPath = 'data/out.avi'
        self.preDefinedConfidence = 0.5
        self.preDefinedThreshold = 0.3
        self.USE_GPU = 0

        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        if self.USE_GPU:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        self.ln = [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.videoStream = cv2.VideoCapture(self.inputVideoPath)
        self.video_width = int(self.videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.x1_line, self.y1_line = 0, self.video_height // 2
        self.x2_line, self.y2_line = self.video_width, self.video_height // 2

        self.previous_frame_detections = [{} for _ in range(self.FRAMES_BEFORE_CURRENT)]
        self.num_frames, self.vehicle_count, self.people_count = 0, 0, 0
        self.writer = self.initializeVideoWriter()

        self.counted_vehicle_ids=set()
        self.counted_pedestrian_ids=set()

        self.start_time = int(time.time())

    def initializeVideoWriter(self):
        sourceVideofps = self.videoStream.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        return cv2.VideoWriter(self.outputVideoPath, fourcc, sourceVideofps, (self.video_width, self.video_height), True)

    def displayVehicleCount(self, frame, vehicle_count):
        cv2.putText(frame, 'Detected Vehicles: ' + str(vehicle_count), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0xFF, 0), 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)

    def displayPedestrianCount(self, frame, pedestrian_count):
        cv2.putText(frame, 'Detected Pedestrians: ' + str(pedestrian_count), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0x7F, 0), 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)

    def drawDetectionBoxes(self, idxs, boxes, classIDs, confidences, frame):
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (x + (w//2), y + (h//2)), 2, (0, 0xFF, 0), thickness=2)

    def displayFPS(self, start_time, num_frames):
        current_time = int(time.time())
        if(current_time > start_time):
            print("FPS:", num_frames)
            num_frames = 0
            start_time = current_time
        return start_time, num_frames

    def find_nearest_key(self, dictionary: Dict, target_key: Tuple[int, int], max_distance: int = 20):
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

    def boxInPreviousFrames(self, previous_frame_detections: List[Dict], current_box: Tuple[int, int, int, int], current_detections: Dict):
        centerX, centerY, width, height = current_box
        dist = np.inf

        for i in range(self.FRAMES_BEFORE_CURRENT):
            coordinate_list = list(previous_frame_detections[i].keys())
            if len(coordinate_list) == 0:
                continue
            temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
            if temp_dist < dist:
                dist = temp_dist
                frame_num = i
                coord = coordinate_list[index[0]]

        if dist > (max(width, height) / 2):
            return False

        current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
        return True

    def count_vehicles(self, idxs, boxes, classIDs, frame):
        current_detections = {}

        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centerX = x + (w // 2)
            centerY = y + (h // 2)

            if self.LABELS[classIDs[i]] in self.list_of_vehicles:
                if not self.boxInPreviousFrames(self.previous_frame_detections, (centerX, centerY, w, h), current_detections):
                    ID = self.vehicle_count
                    current_detections[(centerX, centerY)] = ID
                    self.vehicle_count += 1
                cv2.putText(frame, str(ID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
            
            # Check if the detected object is a vehicle or person
            if self.LABELS[classIDs[i]] in self.list_of_vehicles:
                if not self.boxInPreviousFrames(self.previous_frame_detections, (centerX, centerY, w, h), current_detections):
                    # Assign a new ID if this is a new detection
                    ID = self.vehicle_count
                    current_detections[(centerX, centerY)] = ID

                    if ID not in self.counted_vehicle_ids:
                        self.counted_vehicle_ids.add(ID)
                        self.vehicle_count += 1

                else:
                    # Use existing ID for this detection
                    ID = current_detections.get((centerX, centerY), self.vehicle_count)

                cv2.putText(frame, str(ID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

        return current_detections

    def fit(self):
        while True:
            print("================NEW FRAME================")
            self.num_frames += 1
            print("FRAME:\t", self.num_frames)

            self.start_time, self.num_frames = self.displayFPS(self.start_time, self.num_frames)
            (grabbed, frame) = self.videoStream.read()

            if not grabbed:
                break

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.inputWidth, self.inputHeight), swapRB=True, crop=False)
            self.net.setInput(blob)
            start = time.time()
            layerOutputs = self.net.forward(self.ln)
            end = time.time()

            boxes, confidences, classIDs = [], [], []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > self.preDefinedConfidence:
                        box = detection[0:4] * np.array([self.video_width, self.video_height, self.video_width, self.video_height])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.preDefinedConfidence, self.preDefinedThreshold)
            self.drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)
            current_detections = self.count_vehicles(idxs, boxes, classIDs, frame)

            self.displayVehicleCount(frame, self.vehicle_count)

            self.writer.write(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.previous_frame_detections.pop(0)
            self.previous_frame_detections.append(current_detections)

        print("[INFO] cleaning up...")
        self.writer.release()
        self.videoStream.release()
