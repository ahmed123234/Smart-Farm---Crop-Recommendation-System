# 🌾 SmartFarm Crop Recommendation System

## Project Overview

The SmartFarm Crop Recommendation System is an end-to-end machine learning project designed to advise farmers on the most suitable crop to plant based on soil nutrient profiles ($\text{N, P, K, pH}$) and localized climatic conditions ($\text{temperature, humidity, rainfall}$). This classification solution aims to optimize agricultural output, reduce resource waste, and mitigate planting risk, contributing to the field of Precision Agriculture.

The project follows a full MLOps lifecycle, moving from data analysis and model selection to containerization and deployment.

The core of the application is a pre-trained Random Forest Classifier wrapped within a Scikit-learn Pipeline, ensuring reliable, high-performance predictions accessible via a simple RESTful endpoint. The entire service is containerized using Docker for seamless deployment across various environments, including local Minikube clusters and cloud platforms like Render Run.

## 🌟 Key Features

* **Data Preparation & EDA**: Comprehensive analysis of the dataset, including feature distribution, balancing checks, and necessary data scaling.

* **Multi-Model Training**: Compares the performance of three classic classification algorithms: Logistic Regression, K-Nearest Neighbors (KNN), Random Forests, Decison Trees and XGBoost.

* **Hyperparameter Tuning**: Utilizes GridSearchCV on the best-performing model (Random Forest) to find the optimal parameter set, maximizing predictive accuracy.

* **Feature Importance**: Analyzes the contribution of each environmental factor, providing interpretability for the final model's decisions.

* **Model Export**: Saves the final, tuned model and the data scaler using pickle for easy integration into the $\text{API}$.

* **Web Service Deployment**: Serves real-time predictions via a lightweight Flask $\text{API}$ powered by Gunicorn or waitress.

* **Containerization**: Provides a production-ready **Docker** setup for easy, reproducible local deployment.
  

## 🛠️ Prerequisites

To run this project, you need the following installed:

1. Primary Language: **Python 3.10+**

2. **pip** (Python package installer)
3. Data Science: **Pandas**, **NumPy**
4. Machine Learning: **Scikit-learn** (Version 1.5.1+)
5. API Framework: **Flask**
6. Production Server: **Waitress** or **Gunicorn**
7. Environment Management: **Pipenv** (or standard pip using requirements.txt)
8. Deployment: **Docker**, **Kubernetes** (via Minikube), **Render** for cloud deployment


## 📥 Setup and Installation

### Step 1: Download the Dataset

Download the **Crop Recommendation Dataset** (Crop_recommendation.csv) from Kaggle and place it directly into the project root directory.

### Step 2: Ensure Project Files Exist

Verify that the following files are present in your project directory (these will be generated throughout our process):

**README.md**(this file)

**Pipfile**

**train.py**

**predict.py**

**Dockerfile**

### Step 3: Install Dependencies

You will need the dependencies defined in the Pipfile file (generated previously):

```bash
# install pipenv
pip install pipenv

#install the needed dependencies
pipenv install 
```

## 🚀 Execution Guide (End-to-End)

The project execution is divided into two primary phases: Model Training and $\text{API}$ Deployment.

### Phase 1: Train, Tune, and Export the Model (Run train.py)

This is the first step and is necessary to create the required model files.

Run the ML Pipeline:

```bash
python train.py
```

#### Pipeline Actions:

- Loads and preprocesses data.

- Trains and evaluates Logistic Regression, KNN, Decesion Trees, XGBoost and Random Forest.

- Selects Random Forest as the best base model.

- Tunes the Random Forest model using GridSearchCV.

- Prints the final metrics and feature importance.

#### Critical Output Files:
After successful execution, the script will create two critical files in the root directory:

**crop_recommendation_model_pipeline.pkl**: The saved, highly-tuned Random Forest Classifier.

**scaler.pkl**: The fitted StandardScaler object, crucial for transforming incoming $\text{API}$ data.

### Phase 2: Deploy the Prediction Service with Docker (Run $\text{predict.py}$)

Once the model files (crop_recommendation_model_pipeline.pkl and scaler.pkl) are created, you can containerize and run the prediction $\text{API}$.

#### Build the Docker Image:

```bash
docker build -t smartfarm-api .
```

#### Run the Container:

This command runs the image in the background (-d) and maps the container's port 9040 to your local machine's port 9040 (-p).

```bash
docker run -d -p 9040:9040 --name smartfarm-service smartfarm-api
```

#### Verification (Optional):

Check the Docker logs to ensure the Gunicorn server successfully started and loaded the models:

```bash
docker logs smartfarm-service
```

#### 💻 Making Predictions (Using the Deployed API)

The prediction service is now running locally at http://localhost:9040. The API endpoint is **/predict** and accepts a **JSON payload**.

#### Endpoint Details

Method: **POST**

URL: **http://localhost:9040/predict**

Content-Type: **application/json**

Request Body (**JSON**)


Start the production server locally, listening on port 9040:

```bash
waitress-serve --listen=0.0.0.0:9040 predict:app
```

Input JSON Format: The payload must be a list containing a single $\text{JSON}$ object with the seven required feature keys (names must match exactly):

```bash
[
  {
    "N": 90,
    "P": 45,
    "K": 40,
    "temperature": 25.5,
    "humidity": 85.0,
    "ph": 6.5,
    "rainfall": 220.0
  }
]
```

#### Example cURL Request (Predicting Rice):

```bash
curl -X POST http://localhost:9040/predict -H "Content-Type: application/json" -d '
[
  {
    "N": 90,
    "P": 45,
    "K": 40,
    "temperature": 25.5,
    "humidity": 85.0,
    "ph": 6.5,
    "rainfall": 220.0
  }
]'
```

#### Example Response:

```bash
{
  "confidence_score": 0.99,
  "recommended_crop": "rice"
}
```

#### 🛑 Stopping the Service

To stop and remove the running Docker container:

```bash
docker stop smartfarm-service
docker rm smartfarm-service
```

### 🐳 Deployment (Docker & Kubernetes)

The application is built for containerized deployment, ensuring a consistent environment from development to production.

Docker image was built in the previous step 

#### Kubernetes (Minikube Example)

The repository includes standard Kubernetes manifests (**deployment.yaml** and **service.yaml**) for deployment into a local Minikube cluster:

```bash
# Ensure **Minikube** is started and connected to Docker:
eval $(minikube docker-env)

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Get the URL to test the service
minikube service smartfarm-service --url
```
### Cloud Deployment (Render)
use **https://smartfarm-predictor.onrender.com/** (POST) to access the predict service

Example accessing the predict service using https://smartfarm-predictor.onrender.com/ 

```bash
curl -X POST https://smartfarm-predictor.onrender.com/: 19, "temperature": 27.3179, "humidity": 51.6692, "ph": 6.0052, "rainfall": 32.5591}'

```
