import cv2
import face_recognition
import pickle

def main(model_path, threshold=0.6):
    # Load the known faces and embeddings
    print("[INFO] Loading encodings...")
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    print("[INFO] Starting video stream...")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and face encodings
        boxes = face_recognition.face_locations(rgb_frame, model="hog")
        encodings = face_recognition.face_encodings(rgb_frame, boxes)

        names = []
        distances_display = []

        for encoding in encodings:
            # Compare against known encodings
            face_distances = face_recognition.face_distance(data["encodings"], encoding)

            # Initially, mark as Unknown and distance as None
            name = "Unknown"
            distance_val = None

            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                best_distance = face_distances[best_match_index]
                if best_distance <= threshold:
                    name = data["names"][best_match_index]
                    distance_val = best_distance

            names.append(name)
            distances_display.append(distance_val)

        # Display results
        for ((top, right, bottom, left), name, dist) in zip(boxes, names, distances_display):
            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Prepare label with name and optionally similarity score
            label = name
            if dist is not None:
                label += f" ({dist:.2f})"

            # Label background rectangle
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
            # Label text
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Real-time Face Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Directly set the model encodings file here
    model_path = "encodings.pickle"  # <-- Set your serialized encodings file path here

    # You can adjust the threshold here if needed (default 0.6 is typical)
    threshold = 0.4

    main(model_path, threshold)
