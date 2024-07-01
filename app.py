from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from openpose import pyopenpose as op

app = Flask(__name__)

# 加载训练好的YOLOv8模型
model = YOLO('/Users/liangweixian/Downloads/yolov8n.pt')

# OpenPose初始化
params = dict()
params["model_folder"] = "/Users/liangweixian/Downloads/openpose/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def estimate_pose(image):
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])
    return datum.cvOutputData, datum.poseKeypoints

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    results = model(img)
    pose_image, keypoints = estimate_pose(img)
    
    # 结合识别结果和姿态估计
    detection_results = {
        "objects": results.pandas().xyxy[0].to_dict(),
        "pose": keypoints.tolist()
    }
    
    return jsonify(detection_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
