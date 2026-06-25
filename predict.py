import cv2
import os
from ultralytics import YOLO

# Load trained model
model = YOLO(r"runs/classify/train/weights/best.pt")
video_path = r"D:\Door_Detection\video2.mp4"

# Output folder to save detected frame
save_folder = r"D:\Door_Detection\detected_frames"
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

empty_count = 0
threshold = 5   # trigger if empty appears in 5 consecutive frames
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Predict on current frame
    results = model.predict(source=frame, verbose=False)

    probs = results[0].probs
    class_id = probs.top1
    class_name = results[0].names[class_id]
    confidence = probs.top1conf.item()

    # If predicted empty, increase counter
    if class_name.lower() == "empty":
        empty_count += 1
    else:
        empty_count = 0

    # Show prediction on video
    cv2.putText(
        frame,
        f"{class_name} ({confidence:.2f})",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Truck State Detection", frame)

    # Trigger if empty detected for threshold frames
    if empty_count >= threshold:
        print(f"Truck door is open at frame {frame_number}")

        # Save detected frame
        save_path = os.path.join(save_folder, f"open_detected_frame_{frame_number}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Frame saved at: {save_path}")

        # Show this frame for 5 seconds
        cv2.imshow("Detected Open Frame", frame)
        cv2.waitKey(5000)   # 5000 ms = 5 sec
        break

    # Press q to quit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()