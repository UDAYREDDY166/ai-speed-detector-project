from ultralytics import YOLO
import cv2
import math

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("traffic.mp4")

# Store previous positions
vehicle_positions = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Track vehicles
    results = model.track(frame, persist=True)

    # Get boxes
    boxes = results[0].boxes

    if boxes.id is not None:

        for box, track_id in zip(boxes.xyxy, boxes.id):

            x1, y1, x2, y2 = map(int, box)

            track_id = int(track_id)

            # Find center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Default speed
            speed = 0

            # Check previous position
            if track_id in vehicle_positions:

                prev_x, prev_y = vehicle_positions[track_id]

                # Calculate movement distance
                distance = math.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)

                # Fake speed conversion
                speed = int(distance * 2)

            # Store current position
            vehicle_positions[track_id] = (cx, cy)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Show speed
            cv2.putText(
                frame,
                f"ID {track_id} Speed: {speed} km/h",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    # Show output
    cv2.imshow("Speed Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()