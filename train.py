import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    # Load processed data
    try:
        with open('processed_data.pkl', 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print("Processed data not found. Run dataset.py first.")
        return
        
    X = dataset['data']
    y = dataset['labels']
    info = dataset['info']
    
    if len(X) == 0:
        print("No data to train on.")
        return
        
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model and student info
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump({'model': knn, 'info': info}, f)
        
    print("Trained model saved to trained_model.pkl")

if __name__ == "__main__":
    train_model()
