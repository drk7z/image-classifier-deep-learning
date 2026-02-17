# 👋 Portfolio Project for Recruiters — Pet Image Classifier (DL + MLOps + Security)

## 🎯 Project Summary
This repository showcases an end-to-end AI product focused on **image classification** (Cat vs Dog) using **Deep Learning** with a production-minded approach.

It is not only a model training notebook — it includes:
- A deployable Streamlit application
- Secure file upload handling
- CI pipeline with dependency vulnerability audit
- Dockerized runtime for consistent environments

## 💼 Why this project matters
This project demonstrates practical skills recruiters often look for:
- **Machine Learning Engineering**: model training, evaluation, inference
- **Software Engineering**: modular Python architecture and reusable components
- **Product Thinking**: user-facing app with intuitive UX
- **Security & DevOps**: CI checks, vulnerability scanning, container hardening

## 🧠 What the app does
The app receives an uploaded image and predicts whether it is:
- 🐱 Gato
- 🐶 Cachorro

It uses **Transfer Learning (MobileNetV2)** to leverage pre-trained visual features and returns:
- Predicted class
- Confidence score
- Confidence chart for both classes

## 🏗️ Technical Stack
- Python
- TensorFlow / Keras
- Streamlit
- NumPy / Pillow / Matplotlib
- GitHub Actions (CI)
- Docker

## 🔐 Security & Reliability Highlights
- File type and MIME validation for uploads
- Image integrity validation before prediction
- Upload size limits
- Safe temporary file handling
- CI with `pip-audit` to detect known CVEs in dependencies
- Non-root container execution

## 📁 Architecture Overview
- `src/model.py`: model definitions
- `src/train.py`: training pipeline
- `src/evaluate.py`: model evaluation
- `src/predict.py`: inference logic
- `app.py`: Streamlit frontend
- `.github/workflows/ci.yml`: CI security and quality checks
- `Dockerfile`: production-ready containerization

## 🚀 Quick Run
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start the app:
   - `streamlit run app.py`
3. Open in browser:
   - `http://localhost:8501`

## 📌 Recruiter Notes
If you are evaluating this repository for hiring:
- This project is designed to show both **ML capability** and **software delivery maturity**.
- The codebase reflects concern for **readability, maintainability, and secure defaults**.
- It can be extended to multi-class classification and cloud deployment.

## 📫 Contact
If useful, I can provide a guided walkthrough of:
- model training decisions
- performance trade-offs
- production hardening choices
