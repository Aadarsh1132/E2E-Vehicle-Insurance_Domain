# 🚀 MLOps Project: End-to-End Machine Learning Pipeline

## 🌟 Project Overview
This project implements a full-fledged **MLOps pipeline** that automates the training, validation, deployment, and monitoring of a machine learning model. The pipeline is powered by **CI/CD workflows, AWS, MongoDB, MLflow, and Docker** to ensure seamless model versioning, deployment, and scalability.

## 🏗️ Key Features
✅ **Automated CI/CD pipeline** using GitHub Actions and AWS EC2 for deployment  
✅ **Experiment tracking with MLflow** to manage and compare model performance  
✅ **Data Ingestion & Processing** with MongoDB Atlas as a scalable NoSQL database  
✅ **Model Training & Evaluation** using an LSTM-based neural network  
✅ **AWS S3 Integration** for model storage and versioning  
✅ **Self-Hosted Runner** on AWS EC2 to automate the deployment process  
✅ **Dockerized Application** for containerized and scalable deployment  
✅ **Live Web App** running on an EC2 instance accessible via a public IP  

---
## 📌 Technologies & Services Used
- **Programming Language**: Python 3.10
- **Frameworks & Libraries**: TensorFlow/Keras, Pandas, Scikit-learn, Flask, FastAPI
- **MLOps Tools**: MLflow, Docker, GitHub Actions, AWS EC2, AWS S3, MongoDB Atlas
- **Database**: MongoDB Atlas (for storing raw and processed data)
- **Cloud Services**: AWS IAM, AWS EC2, AWS S3, AWS ECR
- **CI/CD**: GitHub Actions, AWS IAM, AWS ECR, EC2 Self-Hosted Runner

---
## 📂 Project Structure
```
📦 mlops-project
├── 📂 src
│   ├── 📂 configuration  # AWS & MongoDB Configurations
│   ├── 📂 components     # ML pipeline components (data ingestion, transformation, training)
│   ├── 📂 entity         # Data classes for schema validation
│   ├── 📂 pipelines      # Training and inference pipeline
│   ├── 📂 aws_storage    # Model storage management on AWS S3
│   ├── 📜 app.py         # Main API server for predictions
│   ├── 📜 model.py       # Model training script
│   ├── 📜 utils.py       # Helper functions
├── 📂 notebooks          # Jupyter notebooks for EDA & Feature Engineering
├── 📂 static             # Static web files (CSS, images, etc.)
├── 📂 templates          # HTML templates for the web UI
├── 📜 requirements.txt   # Dependencies
├── 📜 setup.py           # Package setup
├── 📜 Dockerfile         # Docker image definition
├── 📜 .github/workflows  # CI/CD pipeline configuration
└── 📜 README.md          # Project documentation
```

---
## 🔥 How to Set Up and Run the Project
### 1️⃣ **Setup the Environment**
```sh
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```
### 2️⃣ **Configure MongoDB Atlas**
- Create a free MongoDB Atlas cluster
- Add a new database and collection
- Get the MongoDB connection string
- Set up environment variables:
```sh
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/db_name"
```

### 3️⃣ **Train the Model**
Run the training pipeline:
```sh
python src/pipelines/train_pipeline.py
```

### 4️⃣ **Deploy the Web App**
```sh
flask run --host=0.0.0.0 --port=5000
```

---
## 📡 Deployment Pipeline
This project uses **GitHub Actions + AWS EC2 + Docker** for automated deployment.
### **CI/CD Workflow**
1. **Push to GitHub** → Triggers GitHub Actions
2. **Docker Build & Push** → Builds and pushes Docker image to AWS ECR
3. **Deploy to AWS EC2** → Runs the containerized model on the cloud

### **AWS Setup for Deployment**
- **IAM Setup**: Create IAM user and attach `AdministratorAccess`
- **S3 Setup**: Store trained models in S3 bucket
- **EC2 Setup**: Launch an instance & install Docker
- **Self-Hosted Runner**: Connect GitHub Actions to EC2

---
## 🌍 Accessing the Live Application
Once deployed, the web app will be accessible at:
```sh
http://54.167.32.143:5000
```
### **Model Training Endpoint**
```sh
http://54.167.32.143:5000/training
```

---
## 🛠️ Future Enhancements
🔹 Model drift detection and retraining pipeline  
🔹 CI/CD extension for multi-cloud deployment (Azure, GCP)  
🔹 Advanced hyperparameter tuning with Optuna  

---
## 🤝 Contributing
Feel free to submit issues and pull requests.

**Star ⭐ this repo if you find it useful!**

