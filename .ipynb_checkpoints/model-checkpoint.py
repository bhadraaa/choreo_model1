import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset from CSV
csv_file = "hand_landmarks.csv"
label_file = "labels.csv"

df = pd.read_csv(csv_file)
labels_df = pd.read_csv(label_file)

# Extract landmarks and labels
X = df.iloc[:, 1:].values  # All columns except index
y = df.iloc[:, 0].values  # First column as labels

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Output layer (number of gestures)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save trained model 
model.save("gesture_model.h5")
print("Model trained and saved as 'gesture_model.h5'")
