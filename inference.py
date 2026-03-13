import cv2
import pickle
import numpy as np
from datetime import datetime
import os

def run_inference():
    # Load trained model and student info
    try:
        with open('trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            knn = model_data['model']
            student_info = model_data['info']
    except FileNotFoundError:
        print("Trained model not found. Run train.py first.")
        return
        
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    # Create attendance directory if it doesn't exist
    if not os.path.exists('attendance'):
        os.makedirs('attendance')
        
    # Track attendance for current session
    attendance_recorded = set()
    
    print("Starting inference. Press 'q' to stop.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            face_flatten = face_roi.flatten().reshape(1, -1)
            
            # Predict student
            prediction = knn.predict(face_flatten)[0]
            confidence = knn.predict_proba(face_flatten)[0][prediction]
            
            # Threshold for unknown
            if confidence > 0.6:
                name, sid = student_info[prediction]
                status = "Present"
                color = (0, 255, 0) # Green for registered
                
                # Record attendance
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")
                
                if (name, sid, date_str) not in attendance_recorded:
                    attendance_recorded.add((name, sid, date_str))
                    with open(f'attendance/attendance_{date_str}.csv', 'a') as f:
                        f.write(f"{name},{sid},{date_str},{time_str}\n")
            else:
                name = "Unknown"
                sid = "N/A"
                status = "Not Registered"
                color = (0, 0, 255) # Red for unknown
                
            # Draw bounding box and caption
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display info
            now = datetime.now()
            date_display = now.strftime("%Y-%m-%d %H:%M:%S")
            
            cv2.putText(frame, f"Name: {name}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"ID: {sid}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Status: {status}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Date: {date_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        cv2.imshow('Attendance Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()
