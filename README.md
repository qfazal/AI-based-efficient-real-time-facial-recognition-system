# AI-based Efficient Real-Time Facial Recognition System

This project implements a real-time facial recognition system using **Facenet-PyTorch**, designed for efficiency, reliability, security, and accuracy. The project commenced in February 2025 and is ongoing. The following tasks have been completed so far:

\- ✅ Literature review and identification of limitations and research gap  

\- ✅ Implementation of baseline model(script: `src/face_recognition.py`)

\- ✅ Inception Resnet (V1) model pretrained on VGGFace2, downloaded and loaded offline correctly using an alternative approach, thereby enhancing security, speed and efficiency by eliminating network latency. Reliability is also increased as the facerecognition system works without a stable internet connection. 

\- ✅ Multiple faces in stored videos detected correctly using mtcnn 

\- ✅ Data of detected faces stored in a database file using pickle

\- ✅ Fast and reliable face recognition of known single faces in terms of speed


\- ⚠️ Current accuracy has limitations; recognition performance decreases under certain conditions (e.g., low light, occlusion).

\- ⚠️ Current implementation performs recognition of a single known face.

Tasks currently in progress:  

\- ⏳ Development and implementation of statistical adaptive thresholds for improved similarity matching and recognition

\- ⏳ Custom Webcam feed-based dataset creation and expansion

\- ⏳ Designing a GUI for the recognition system 

\- ⏳ Accuracy benchmarking  


Planned next steps:  

\- 🔜 Implement and test the pipeline in real-time using a webcam feed

\- 🔜 Extended evaluation metrics and benchmarking  

\- 🔜 Documentation and publishing of results


How to run:

Step 1: Run the download_model.py file on Google Colab for downloading 'inception_resnet_v1_vggface2.pth'

Step 2: Download the 'inception_resnet_v1_vggface2.pth' file and copy it in the directory where the face_recognition.py file is placed

Step 3: Run face_recognition.py


