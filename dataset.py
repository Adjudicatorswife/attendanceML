import cv2
import os
import numpy as np
import pickle

def prepare_dataset():
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    data = []
    labels = []
    student_info = {} # Mapping of label to (name, id)
    
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        print("Dataset directory not found.")
        return
        
    label_count = 0
    for student_dir in os.listdir(dataset_path):
        student_path = os.path.join(dataset_path, student_dir)
        if not os.path.isdir(student_path):
            continue
            
        name, sid = student_dir.split('_')
        student_info[label_count] = (name, sid)
        
        print(f"Processing images for {name} ({sid})...")
        
        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                data.append(face_roi.flatten())
                labels.append(label_count)
                
        label_count += 1
        
    if not data:
        print("No face data found in dataset.")
        return
        
    # Save processed data
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump({'data': np.array(data), 'labels': np.array(labels), 'info': student_info}, f)
        
    print("Dataset preparation complete. Saved to processed_data.pkl")

if __name__ == "__main__":
    prepare_dataset()
