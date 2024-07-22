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

import cv2
import pickle
import sys
import numpy as np
from ultralytics import YOLO
sys.path.append("../")
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections, init_frames=5):
        # Use the first few frames to initialize player selection
        initial_detections = player_detections[:init_frames]
        chosen_players = self.choose_players(court_keypoints, initial_detections)
        
        # Filter detections based on chosen players
        filtered_player_detections = [
            {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            for player_dict in player_detections
        ]
        return filtered_player_detections

    def choose_players(self, court_keypoints, initial_detections):
        court_center = np.mean(np.array(court_keypoints).reshape(-1, 2), axis=0)
        
        # Aggregate player positions across initial frames
        player_positions = {}
        for frame_detections in initial_detections:
            for track_id, bbox in frame_detections.items():
                center = get_center_of_bbox(bbox)
                if track_id in player_positions:
                    player_positions[track_id].append(center)
                else:
                    player_positions[track_id] = [center]
        
        # Calculate average position and distance to court center
        avg_player_positions = {track_id: np.mean(positions, axis=0) for track_id, positions in player_positions.items()}
        distances = [(track_id, np.linalg.norm(avg_position - court_center)) for track_id, avg_position in avg_player_positions.items()]
        
        # Sort by distance and select closest players
        distances.sort(key=lambda x: x[1])
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        player_detection = [self.detect_frame(frame) for frame in frames]
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detection, f)
                
        return player_detection
    
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names
        
        player_dict = {
            int(box.id.tolist()[0]): box.xyxy.tolist()[0]
            for box in results.boxes if id_name_dict[box.cls.tolist()[0]] == "person"
        }
        
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Player ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            output_video_frames.append(frame)
        return output_video_frames
