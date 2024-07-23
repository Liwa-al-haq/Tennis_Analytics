from utils import (read_video,
                   save_video,
                   measure_distance, draw_player_stats, convert_meters_to_pixel_distance, convert_pixel_distance_to_meters,draw_ball_stat)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court import MiniCourt
from copy import deepcopy
import pandas as pd
def main():
    #Reading Video
    input_video_path = "input_videos/input_video2.mp4"
    video_frames = read_video(input_video_path)
    
    #Detect the players and ball
    player_tracker = PlayerTracker(model_path='yolov10x.pt')
    ball_tracker = BallTracker('models/yolo5_last.pt')
    
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections2.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections2.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    #Draw the output
    
    #Court Line Detection
    court_line_detector = CourtLineDetector("models/keypoints_model.pt")
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    # Make later to filter only to 2 players 
    player_detections = player_tracker.choose_and_filter_players(player_detections)
    # player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    
    #Make later Inititalize Mini Court
    mini_court = MiniCourt(video_frames[0])
    
    #Detect ball shot make after minicourt
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    #Draw player bounding boxes bbox
    #Later Convert position to mini court position
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)
    
    #Calc ball shot speed
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,
        'player_1_distance_covered':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
        'player_2_distance_covered':0
    } ]
    
    player_stats_data = []
    player_1_total_distance = 0
    player_2_total_distance = 0

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1]
        )
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # Player who shot the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id],
                ball_mini_court_detections[start_frame][1]
            )
        )

        # Opponent player
        opponent_player_id = 1 if player_shot_ball == 2 else 2

        # Distance covered by the player who shot the ball
        distance_covered_by_shot_player_pixels = measure_distance(
            player_mini_court_detections[start_frame][player_shot_ball],
            player_mini_court_detections[end_frame][player_shot_ball]
        )
        distance_covered_by_shot_player_meters = convert_pixel_distance_to_meters(
            distance_covered_by_shot_player_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )

        # Distance covered by the opponent player
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id]
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )

        # Update total distances covered
        if player_shot_ball == 1:
            player_1_total_distance += distance_covered_by_shot_player_meters
            player_2_total_distance += distance_covered_by_opponent_meters
        else:
            player_1_total_distance += distance_covered_by_opponent_meters
            player_2_total_distance += distance_covered_by_shot_player_meters

        # Speed of the opponent
        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        # Update player stats
        current_player_stats = deepcopy(player_stats_data[-1]) if player_stats_data else {
            'frame_num': start_frame,
            'player_1_number_of_shots': 0,
            'player_1_total_shot_speed': 0,
            'player_1_last_shot_speed': 0,
            'player_1_distance_covered': 0,
            'player_1_total_player_speed': 0,
            'player_1_last_player_speed': 0,
            'player_2_number_of_shots': 0,
            'player_2_total_shot_speed': 0,
            'player_2_last_shot_speed': 0,
            'player_2_distance_covered': 0,
            'player_2_total_player_speed': 0,
            'player_2_last_player_speed': 0
        }

        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_distance_covered'] = player_1_total_distance if opponent_player_id == 1 else player_2_total_distance
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']

    
    print(f"ball_shot_frames: {ball_shot_frames}")
        
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
    
    ## Draw court KeyPoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    
    #Later Draw MiniCourt 
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color = (0,255,255))
    
    #Later Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)
    
    output_video_frames = draw_ball_stat(output_video_frames, ball_shot_frames)
    #Draw frame no on top left corner 
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame No: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        output_video_frames[i] = frame
    
    save_video(output_video_frames, "output_videos/output_video5.avi")
if __name__ == "__main__":
    main()
    
## Todo
# 1. Add number of shots
# 2. How fast player is moving when he dosent have ball
# 3. How fast ball for winning shot
# 4. Check if ball is in the court or not
# 5. Check if player is in the court or not
# 6. Net Logic
