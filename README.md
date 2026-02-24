# EmotionDetection

🎭 Emotion Detection using CNN
This project proposes a Real-time Facial Emotion Recognition system using Convolutional Neural Networks (CNN). The proposed system will classify human facial expressions into various categories such as Happy, Sad, Angry, Surprise, and so on.

🚀 Technology Stack
The project utilizes the following libraries:

TensorFlow/Keras: For designing, compiling, and training the deep learning model.

OpenCV (cv2): For image preprocessing, resizing, and real-time face detection.

Matplotlib: For data visualization and plotting training history.

NumPy: For efficient numerical computations and array manipulations.

Dataset
The proposed system will be trained on the FER-2013 dataset (Facial Expression Recognition).
🔗 Dataset Link: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

🛠️ Installation & Setup
Clone the repository:

Scores
Accuracy: 0.41850097520200613
mean absolute error: 1.4937308442463082

Install dependencies:

Model Architecture
The proposed system will have an architecture comprising multiple Convolutional layers designed to extract spatial features from facial images.

Conv2D Layers: Used to extract features like edges, shapes, and textures.

MaxPooling2D: Used for spatial down-sampling to reduce computational load.

Dropout: Included to prevent overfitting during training.

Flatten & Dense: Converts feature maps into a 1D vector to perform final classification using a Softmax activation.

Implementation Details
The images will be processed using a pipeline comprising grayscale conversion and resizing:

Performance
After 20+ epochs, the proposed system will have the following performance:

Author
Arno Emeksuzyan
