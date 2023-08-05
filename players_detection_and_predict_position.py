# python detect.py --weights runs/train/exp3/weights/best.pt --conf 0.4 --source {dataset.location}/test/images --hide-labels --line-thickness 2

import os
print("CPU_Core:", os.cpu_count())  # コア数

import torch
import cv2

from pathlib import Path

import numpy as np

# import importlib.util

# spec = importlib.util.spec_from_file_location('main', 'camera_pose_estimation_package/camera_pose_estimation/main.py')
# main = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(main)

import sys
sys.path.append('/Users/sugimuraminori/local_soccer_tracker/camera_pose_estimation_pkg')
# from common import draw_point, yolobbox2bbox
from camera_pose_estimation.main import calibrate_from_image

from team_assigner_with_ball_tracking import basic_ball_tracker, yolobbox2bbox

from team_assigner_with_ball_tracking import DominantColors


# 学習済みモデルをアップロード 
model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'best.pt')
# 信頼度の閾値を0.4に変更
model.conf = 0.25

# 検出できる物体を表示する
print(model.names)

# filename = 'images/image_001.png'
# img = cv2.imread(Path(filename))

# results = model(filename)  # Predict
# objects = results.pandas().xyxy[0]  # 検出結果を取得してobjectに格納
# # objectに格納したデータ
# # => バウンディングBOX左上のx座標とy座標、信頼度、クラスラベル、物体名


# # 画像の幅 高さ
# img_width = img.shape[1]
# img_height = img.shape[0]

#  # 物体検出結果を出力するためのtxtファイルを作成
# with open('detect_labels/detection_Result.txt', 'w') as f:

#     for i in range(len(objects)):
#         if (objects.name[i] != "refree"):
#             name = objects.name[i]
#             # 検出されたオブジェクトのバウンディングボックスの座標
#             xmin = objects.xmin[i]
#             ymin = objects.ymin[i]
#             width = objects.xmax[i] - objects.xmin[i]
#             height = objects.ymax[i] - objects.ymin[i]


#             # 中心点の座標を正規化
#             norm_x = ((objects.xmax[i] + objects.xmin[i]) / 2) / img_width
#             norm_y = ((objects.ymax[i] + objects.ymin[i]) / 2) / img_height
            
#             # 幅、高さを正規化
#             normalized_width = width / img_width
#             normalized_height = height / img_height

#             # print(f"{class_id}, 座標x:{xmin}, 座標y:{ymin}, 幅:{width}, 高さ:{height}")
#             # csvファイルにバウンディングBOX情報を出力
#             if objects.name[i] == "ball":
#                 id = 0
#             elif objects.name[i] == "goalkeeper":
#                 id = 1
#             else:
#                 id = 2
#             print(f"{id} {norm_x} {norm_y} {normalized_width} {normalized_height}", file=f)

# results.show()  # 検出した物体の表示




# ---------------------------------------------------------------------------------------------------------
# Copied by main.py

"""
Main script to use the whole algorithms.
First detect key points on the soccer pitch and find the
camera intrinsic/extrinsic calibration matrices
Then use yolov5 detector to locate players and balls.
Finally reproject those detections to a top view image of the soccer field
"""



def from_world_to_field(x, z):
    """Map world coordinates to the soccer_field.png image"""

    center_field = [826, 520]
    scale = (1585 - 68) / 105  # 105 meters

    u = int(center_field[0] + x * scale)
    v = int(center_field[1] - z * scale)
    return (u, v)


def unproject_to_ground(K, to_device_from_world, u, v):
    """
    Unproject pixel point (u, v) and find its world coordinates,
    assuming that its altitude is zero
    """
    # First to K^-1 * (u, v, 1)
    homog_uv = np.ones((3, 1))
    homog_uv[0, 0] = u
    homog_uv[1, 0] = v

    K_inv_uv = np.dot(np.linalg.inv(K), homog_uv)
    alpha = K_inv_uv[0, 0]
    beta = K_inv_uv[1, 0]

    b = np.zeros((2, 1))
    r00 = to_device_from_world[0, 0]
    r02 = to_device_from_world[0, 2]
    r10 = to_device_from_world[1, 0]
    r12 = to_device_from_world[1, 2]
    r20 = to_device_from_world[2, 0]
    r22 = to_device_from_world[2, 2]
    tx = to_device_from_world[0, 3]
    ty = to_device_from_world[1, 3]
    tz = to_device_from_world[2, 3]
    b[0, 0] = alpha * tz - tx
    b[1, 0] = beta * tz - ty

    M = np.zeros((2, 2))
    M[0, 0] = r00 - alpha * r20
    M[0, 1] = r02 - alpha * r22
    M[1, 0] = r10 - beta * r20
    M[1, 1] = r12 - beta * r22

    final = np.dot(np.linalg.inv(M), b)
    X = final[0, 0]
    Z = final[1, 0]

    return X, Z


def is_in_field(X, Z):
    """Check if point (X, Z) is located inside the soccer field boundaries"""
    return -52.5 < X < 52.5 and -34 < Z < 34



# field = cv2.imread("soccer_field.png")
# field_copy = field.copy()

# ball_positions = basic_ball_tracker(Path('detect_labels'))

# # Default value of our focal length to start with
# # Dont forget to change this value if you are using different sizes of images
# guess_fx = 2000
# guess_rot = np.array([[0.25, 0, 0]])
# guess_trans = (0, 0, 80)



# # Find intrinsic/extrinsic camera matrices
# K, to_device_from_world, rot, trans, img = calibrate_from_image(
#     img, guess_fx, guess_rot, guess_trans
# )


# # Load yolov5 inference results
# filename = Path(filename)
# yolo_file = Path("detect_labels/detection_Result.txt")

# data = np.loadtxt(yolo_file, encoding = "UTF-8")

# players_pos = []
# for k in range(data.shape[0]):
#     # Skip ball with class id 0 & Skip referee with class id 3
#     if data[k, 0] == 0 or data[k, 0] == 3:
#         continue

#     # Skip detections with ridiculous values
#     # if data[k, 3] < 0.01:
#     #     continue

#     # Transform yolo bounding box to opencv convention
#     pt1, pt2 = yolobbox2bbox(
#         data[k, 1], data[k, 2], data[k, 3], data[k, 4], img_width, img_height
#     )
#     # Make a sub image and find shirt color
#     sub_img = img[pt1[1] : pt2[1], pt1[0] : pt2[0]]


#     dc = DominantColors(sub_img, 2)
#     colors = dc.dominant_colors()
#     color = (int(colors[0][2]), int(colors[0][1]), int(colors[0][0]))


#     foot_point = int((pt1[0] + pt2[0]) / 2)

#     X, Z = unproject_to_ground(K, to_device_from_world, foot_point, pt2[1])

#     if not is_in_field(X, Z):
#         print('is not in field.')
    
#     # Mark the player with a circle on the soccer pitch
#     field = cv2.circle(
#         field,
#         from_world_to_field(X, Z),
#         radius=15,
#         color=color,
#         thickness=3,
#     )
#     field_player_pos = [round(X+52.5, 2), round(-Z+34, 2)]
#     players_pos.append(field_player_pos)

# ball_bl, ball_tr = yolobbox2bbox(
#     ball_positions[0, 0],
#     ball_positions[0, 1],
#     ball_positions[0, 2],
#     ball_positions[0, 3],
#     img_width,
#     img_height,
# )
# if to_device_from_world is not None:
#     foot_point = int((ball_bl[0] + ball_tr[0]) / 2)

#     X, Z = unproject_to_ground(K, to_device_from_world, foot_point, ball_tr[1])
#     # Mark the ball with green-blue color
#     if is_in_field(X, Z):
#         field = cv2.circle(
#             field,
#             from_world_to_field(X, Z),
#             radius=15,
#             color=(0, 0, 0),
#             thickness=3,
#         )
#         field_ball_pos = [round(X+52.5, 2), round(-Z+34, 2)]


# # Modify current value of calibration matrices to get benefit
# # of this computation for next image
# guess_rot = (
#     rot if to_device_from_world is not None else np.array([[0.25, 0, 0]])
# )
# guess_trans = trans if to_device_from_world is not None else (0, 0, 80)
# guess_fx = K[0, 0]

# Display result
# cv2.imshow("field", field)
# cv2.imshow("img", img)
# k = cv2.waitKey(0)

# out_path = Path("out")
# print(f"Writing image to {str(out_path)}")
# cv2.imwrite(str(out_path), field)

# with open('top_view_labels/Each_Players_position.csv', 'w') as f:
#     print(f"class_id, x, y", file=f)

#     for i in range(len(players_pos)):
#         print(f'2, {players_pos[i][0]}, {(players_pos[i][1])}', file = f)
#     print(f'0, {field_ball_pos[0]}, {(field_ball_pos[1])}', file = f)


# from video--------------------------------------------------------------------------------------------
print('input video, and output video.')
HOME = '/Users/sugimuraminori'
print('HOME:', HOME)

from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime as dt

SOURCE_VIDEO_PATH = f"{HOME}/local_soccer_tracker/video/video2.mp4"

tdatetime = dt.now()
tstr = tdatetime.strftime('%Y:%m:%d_%H:%M:%S')
TARGET_VIDEO_PATH = f"{HOME}/local_soccer_tracker/out/top_view/top_view.mp4"


@dataclass(frozen=True)
class VideoConfig:
    fps: float
    width: int
    height: int

def generate_frames(video_file: str):
    video = cv2.VideoCapture(video_file)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        yield frame

    video.release()

# create cv2.VideoWriter object that we can use to save output video
def get_video_writer(target_video_path: str, video_config: VideoConfig) -> cv2.VideoWriter:
    video_target_dir = os.path.dirname(os.path.abspath(target_video_path))
    os.makedirs(video_target_dir, exist_ok=True)
    return cv2.VideoWriter(
        target_video_path, 
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"), 
        fps=video_config.fps, 
        frameSize=(video_config.width, video_config.height), 
        isColor=True
    )

frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))


cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS:", fps)

# initiate video writer
video_config = VideoConfig(
    fps = fps, 
    width = video_width, 
    height= video_height)

video_writer = get_video_writer(
    target_video_path=TARGET_VIDEO_PATH, 
    video_config=video_config)

print("players detection...")

# loop over frames
j = 0
for frame in tqdm(frame_iterator, total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    j += 1
    # 物体検出結果を出力するためのtxtファイルを作成
    num = '{0:03d}'.format(j)
    with open(f'detect_labels/detection_Result{num}.txt', 'w', encoding='utf-8') as f:
        # run detector
        results = model(frame, size = 1280)
        
        objects = results.pandas().xyxy[0]  # 検出結果を取得してobjectに格納

        # data = []

        for i in range(len(objects)):
            if (objects.name[i] != "refree"):
                name = objects.name[i]
                # 検出されたオブジェクトのバウンディングボックスの座標
                xmin = objects.xmin[i]
                ymin = objects.ymin[i]
                width = objects.xmax[i] - objects.xmin[i]
                height = objects.ymax[i] - objects.ymin[i]

                # 中心点の座標を正規化
                norm_x = ((objects.xmax[i] + objects.xmin[i]) / 2) / video_width
                norm_y = ((objects.ymax[i] + objects.ymin[i]) / 2) / video_height
                
                # 幅、高さを正規化
                normalized_width = width / video_width
                normalized_height = height / video_height

                # print(f"{class_id}, 座標x:{xmin}, 座標y:{ymin}, 幅:{width}, 高さ:{height}")
                # csvファイルにバウンディングBOX情報を出力
                if objects.name[i] == "ball":
                    id = 0
                elif objects.name[i] == "goalkeeper":
                    id = 1
                else:
                    id = 2
                
                # data.append([id, norm_x, norm_y, normalized_width, normalized_height])
                print(f"{id},{round(norm_x, 5)},{round(norm_y, 5)},{round(normalized_width, 5)},{round(normalized_height, 5)}", file=f)

print("players_detection done!")

ball_positions = basic_ball_tracker(Path('detect_labels'), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

j = 0
for frame in tqdm(frame_iterator, total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    j += 1
    filename = "detection_Result"+'{0:03d}'.format(j)+".txt"
    yolo_file = "./detect_labels/"+filename
    print(yolo_file)
    data = np.loadtxt(yolo_file, encoding="utf-8")
    field = cv2.imread("soccer_field.png")

    

    # Default value of our focal length to start with
    # Dont forget to change this value if you are using different sizes of images
    guess_fx = 2000
    guess_rot = np.array([[0.25, 0, 0]])
    guess_trans = (0, 0, 80)


    # Find intrinsic/extrinsic camera matrices
    K, to_device_from_world, rot, trans, img = calibrate_from_image(
        frame, guess_fx, guess_rot, guess_trans
    )

    # players_pos = []
    for k in range(len(data)):
        # Skip ball with class id 0 & Skip referee with class id 3
        if data[k, 0] == 0.0:
            continue

        # Skip detections with ridiculous values
        # if data[k][3] < 0.01:
        #     continue

        # Transform yolo bounding box to opencv convention
        pt1, pt2 = yolobbox2bbox(
            data[k, 1], data[k, 2], data[k, 3], data[k, 4], video_width, video_height
        )
        # Make a sub image and find shirt color
        sub_img = img[pt1[1] : pt2[1], pt1[0] : pt2[0]]


        dc = DominantColors(sub_img, 2)
        colors = dc.dominant_colors()
        color = (int(colors[0][2]), int(colors[0][1]), int(colors[0][0]))


        foot_point = int((pt1[0] + pt2[0]) / 2)

        X, Z = unproject_to_ground(K, to_device_from_world, foot_point, pt2[1])

        if not is_in_field(X, Z):
            print('is not in field.')
        
        # Mark the player with a circle on the soccer pitch
        field = cv2.circle(
            field,
            from_world_to_field(X, Z),
            radius=15,
            color=color,
            thickness=3,
        )
        # field_player_pos = [round(X+52.5, 2), round(-Z+34, 2)]
        # players_pos.append(field_player_pos)

    ball_bl, ball_tr = yolobbox2bbox(
        ball_positions[0, 0],
        ball_positions[0, 1],
        ball_positions[0, 2],
        ball_positions[0, 3],
        video_width,
        video_height,
    )
    if to_device_from_world is not None:
        foot_point = int((ball_bl[0] + ball_tr[0]) / 2)

        X, Z = unproject_to_ground(K, to_device_from_world, foot_point, ball_tr[1])
        # Mark the ball with green-blue color
        if is_in_field(X, Z):
            field = cv2.circle(
                field,
                from_world_to_field(X, Z),
                radius=15,
                color=(0, 0, 0),
                thickness=3,
            )
            # field_ball_pos = [round(X+52.5, 2), round(-Z+34, 2)]


    # Modify current value of calibration matrices to get benefit
    # of this computation for next image
    guess_rot = (
        rot if to_device_from_world is not None else np.array([[0.25, 0, 0]])
    )
    guess_trans = trans if to_device_from_world is not None else (0, 0, 80)
    guess_fx = K[0, 0]

    # Display result
    # cv2.imshow("field", field)
    # k = cv2.waitKey(0)

    # save video frame
    video_writer.write(field)

# close output video
video_writer.release()
