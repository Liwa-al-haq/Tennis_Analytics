from ultralytics import YOLO

model = YOLO('yolov10x')

result = model.track("input_videos/input_video5.mp4",conf = 0.2, save = True)

# print("boxes:")

# for box in result[0].boxes:
#     print(box)