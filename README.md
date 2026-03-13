# Face Recognition Attendance Tracking System

This Python-based system allows you to collect student face data, train a machine learning model, and perform real-time attendance tracking using a live camera feed.

## Project Structure

- `collect_imgs.py`: Script to capture face images of students.
- `dataset.py`: Script to extract face features and prepare the dataset.
- `train.py`: Script to train a K-Nearest Neighbors (KNN) classifier.
- `inference.py`: Script for real-time face recognition and attendance logging.
- `requirements.txt`: Lists all Python dependencies.
- `dataset/`: Directory where raw student images are stored.
- `attendance/`: Directory where attendance logs (CSV) are saved.

## Prerequisites

### General Setup
Ensure you have Python 3.10.0 (or a compatible Python 3.x version) installed. It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source venv/bin/activate

# Install required libraries from requirements.txt
pip install -r requirements.txt
```

### Windows Specific Notes

For Windows users, ensure you have Python added to your system's PATH during installation. If you encounter issues with `opencv-python`, you might try installing `opencv-contrib-python` instead. You can modify the `requirements.txt` file to `opencv-contrib-python` if needed.

This project uses `opencv`'s Haar Cascades for face detection and a custom K-Nearest Neighbors (KNN) classifier for face recognition. It does not directly use the `face-recognition` library which has complex dependencies like `dlib` and `cmake` that can be challenging to install on Windows. If you wish to explore more advanced face recognition libraries in the future, you might need to install CMake and Visual C++ Build Tools (available with Visual Studio).

## How to Use

### 1. Collect Student Data
Run the collection script for each student you want to register. Replace `<student_name>` and `<student_id>` with the actual details.
```bash
python collect_imgs.py <student_name> <student_id>
```
The script will capture 50 images from your camera. Ensure good lighting and different angles. Press `q` to stop early.

### 2. Prepare the Dataset
After collecting data for all students, run the dataset preparation script to extract face features:
```bash
python dataset.py
```
This will create a `processed_data.pkl` file in the project root.

### 3. Train the Model
Train the machine learning model using the processed data:
```bash
python train.py
```
This will create a `trained_model.pkl` file in the project root.

### 4. Run Attendance Tracking
Start the real-time inference script:
```bash
python inference.py
```
- **Registered Students**: A green box will appear around their face with their name, ID, and "Present" status. Attendance will be logged in `attendance/attendance_YYYY-MM-DD.csv`.
- **Unregistered Students**: A red box will appear with the label "Unknown".

## Features
- **Real-time Detection**: Uses Haar Cascades for fast face detection.
- **Machine Learning**: Uses a KNN classifier for face recognition.
- **Automatic Logging**: Records student name, ID, date, and time in a CSV file.
- **Visual Feedback**: Displays name, ID, date, and status directly on the camera feed.
