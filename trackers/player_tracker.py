# from ultralytics import YOLO
# import cv2
# import pickle
# import sys
# sys.path.append("../")
# from utils import measure_distance, get_center_of_bbox
# class PlayerTracker:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)

#     # Make later
#     def choose_and_filter_players(self, court_keypoints, player_detections):
#         player_detections_first_frame = player_detections[0]
#         chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
#         filtered_player_detections = []
#         for player_dict in player_detections:
#             filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
#             filtered_player_detections.append(filtered_player_dict)
#         return filtered_player_detections

#     #Make later with above
#     def choose_players(self, court_keypoints, player_dict):
#         distances = []
#         for track_id, bbox in player_dict.items():
#             player_center = get_center_of_bbox(bbox)

#             min_distance = float('inf')
#             for i in range(0,len(court_keypoints),2):
#                 court_keypoint = (court_keypoints[i], court_keypoints[i+1])
#                 distance = measure_distance(player_center, court_keypoint)
#                 if distance < min_distance:
#                     min_distance = distance
#             distances.append((track_id, min_distance))
        
#         # sorrt the distances in ascending order
#         distances.sort(key = lambda x: x[1])
#         # Choose the first 2 tracks
#         chosen_players = [distances[0][0], distances[1][0]]
#         return chosen_players
        
#     def detect_frames(self, frames, read_from_stub=False, stub_path=None):
#         player_detection = []

#         if read_from_stub and stub_path is not None:
#             with open(stub_path, 'rb') as f:
#                 player_detection = pickle.load(f)
#             return player_detection
        
#         for frame in frames:
#             player_dict = self.detect_frame(frame)
#             player_detection.append(player_dict)
            
#         if stub_path is not None:
#             with open(stub_path, 'wb') as f:
#                 pickle.dump(player_detection, f)
            
#         return player_detection
    
#     def detect_frame(self, frame):
#         results =self.model.track(frame, persist=True)[0]
#         id_name_dict = results.names
        
#         player_dict = {}
#         for box in results.boxes:
#             track_id = int(box.id.tolist()[0])
#             result = box.xyxy.tolist()[0]
#             object_cls_id = box.cls.tolist()[0]
#             object_cls_name = id_name_dict[object_cls_id]
#             if object_cls_name == "person":
#                 player_dict[track_id] = result
                
#         return player_dict 
    
#     def draw_bboxes(self, video_frames, player_detections):
#         output_video_frames = []
#         for frame, player_dict in zip(video_frames, player_detections):
#             # Draw bounding boxes
#             for track_id, bbox in player_dict.items():
#                 x1, y1, x2, y2 = bbox
#                 cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # -10 means the text is below the bounding box buffer
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) #2 means outside border, the (0, 255, 0) is the color of the rectangle
#             output_video_frames.append(frame)
#         return output_video_frames

from ultralytics import YOLO
import cv2
import pickle
import sys
import numpy as np
from collections import defaultdict, deque

sys.path.append("../")
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.player_history = defaultdict(lambda: deque(maxlen=90))  # Store history for last 20 frames
        self.player_presence = defaultdict(int)
        self.court_boundaries = None
        self.frame_queue = deque(maxlen=90)  # Store last 20 frames of player detections

    def set_court_boundaries(self, court_keypoints):
        self.court_boundaries = np.array(court_keypoints).reshape((-1, 2))

    def choose_and_filter_players(self, player_detections):
        self.update_frame_queue(player_detections)
        consistent_players = self.get_consistent_players()
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in consistent_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def update_frame_queue(self, player_detections):
        self.frame_queue.append(player_detections)
        self.update_player_presence()

    def update_player_presence(self):
        self.player_presence.clear()
        for frame_detections in self.frame_queue:
            for player_dict in frame_detections:
                for track_id, bbox in player_dict.items():
                    player_center = get_center_of_bbox(bbox)
                    if self.is_inside_court(player_center):
                        self.player_presence[track_id] += 1

    def get_consistent_players(self):
        # Filter out players based on their presence in frames
        consistent_players = sorted(self.player_presence.items(), key=lambda item: item[1], reverse=True)[:2]
        consistent_player_ids = [player[0] for player in consistent_players]
        return consistent_player_ids

    def is_inside_court(self, point):
        if self.court_boundaries is None:
            return True
        return cv2.pointPolygonTest(self.court_boundaries, point, False) >= 0

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detection = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detection = pickle.load(f)
            return player_detection
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detection.append(player_dict)
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detection, f)
            
        return player_detection

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
                
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # -10 means the text is below the bounding box buffer
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 2 means outside border, the (0, 255, 0) is the color of the rectangle
            output_video_frames.append(frame)
        return output_video_frames

