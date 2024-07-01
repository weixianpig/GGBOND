from ultralytics import YOLO

# 加载预训练的YOLOv8模型
model = YOLO('yolov8n.pt')

# 训练模型
model.train(data='path/to/dataset.yaml', epochs=50, imgsz=640)

# 评估模型
results = model.val()
print(results)
