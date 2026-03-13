import cv2
import os
import sys

def collect_images(student_name, student_id):
    # Create directory for the student
    directory = f"dataset/{student_name}_{student_id}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    count = 0
    print(f"Starting collection for {student_name}. Press 'q' to stop or wait for 50 images.")
    
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display the frame
        cv2.putText(frame, f"Collecting: {count}/50", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Collecting Images', frame)
        
        # Save image
        img_path = os.path.join(directory, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Successfully collected {count} images for {student_name}.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 collect_imgs.py <student_name> <student_id>")
    else:
        name = sys.argv[1]
        sid = sys.argv[2]
        collect_images(name, sid)
