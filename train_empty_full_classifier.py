from ultralytics import YOLO

# Load pretrained YOLO classification model
model = YOLO("yolov8n-cls.pt")

# Train model
model.train(
    data=r"D:\Door_Detection",   # parent folder containing train/ and val/
    epochs=25,
    imgsz=224,
    batch=16
)