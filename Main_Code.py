import cv2
import os
import numpy as np
import pandas as pd
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 2: Data Preprocessing and Feature Extraction
from skimage.feature import local_binary_pattern

def preprocess_and_extract_features(image_path):
    image = cv2.imread(image_path)
    desired_size = (224, 224)
    image_resized = cv2.resize(image, desired_size)
    denoised_image = cv2.fastNlMeansDenoisingColored(image_resized, None, 10, 10, 7, 21)
    grayscale_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Feature extraction - Calculate Local Binary Patterns (LBP) features using scikit-image
    radius = 1
    points = 8 * radius
    lbp_image = local_binary_pattern(grayscale_image, points, radius, method="uniform")

    return lbp_image.flatten()  # Flatten the LBP image before returning it



# Step 3: Data Collection
dataset_path ="/content/drive/MyDrive/MP 5 Dataset"  # Specify the path to the folder containing your images

# Ensure the specified folder exists
if not os.path.exists(dataset_path):
    raise ValueError(f"The specified folder '{dataset_path}' does not exist. Please provide the correct path.")

# Collect image paths from all subfolders
image_paths = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('jpeg', 'jpg', 'png')):
            image_paths.append(os.path.join(root, file))

# Ensure at least one image is found
if not image_paths:
    raise ValueError(f"No images found in the specified folder '{dataset_path}'. Please check your dataset.")

# Extract features for each image in the dataset
feature_vectors = []
for path in image_paths:
    feature = preprocess_and_extract_features(path)
    feature_vectors.append(feature)

feature_vectors = []
for path in image_paths:
    print("Processing image:", path)
    feature = preprocess_and_extract_features(path)
    feature_vectors.append(feature)

# Add this print statement
print("Number of features extracted:", len(feature_vectors))

# Step 4: Create a DataFrame
feature_df = pd.DataFrame(feature_vectors)


# Print information about the DataFrame
print("DataFrame Info:")
print(feature_df.info())

# Check for missing values
print("\nMissing Values:")
print(feature_df.isnull().sum())

# Step 5: Optional - Feature Scaling
scaler = StandardScaler()

# Handle missing or infinite values by replacing them with zeros
feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
feature_df.fillna(0, inplace=True)

# Convert the DataFrame to a NumPy array before scaling
scaled_features = scaler.fit_transform(feature_df.values)

# Continue with the rest of your code...
# Step 6: Optional - Labels (if available)
# Replace 'labels' with your actual labels or classes
# Example labels for the new dataset
labels = [0, 1, 1, 0]  # Example labels for the new dataset

# Ensure the number of labels matches the number of samples
if len(labels) != len(scaled_features):
    raise ValueError("Number of labels does not match the number of samples.")

label_df = pd.DataFrame({'label': labels})

# Step 7: Split Data (if you have labels)
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# Step 8: Model Training (Use the model of your choice)
model = SVC(kernel='linear')  # Example: Support Vector Machine
model.fit(X_train, y_train)

# Step 9: Make Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
