import torch
import torchvision
import numpy as np
import cv2
from ultralytics import YOLO

def train():
    model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 model
    model.train(data='coco.yaml', epochs=50, batch=16, imgsz=640)  # Training the model

if __name__ == "__main__":
    train()
