#🌾 SmartFarm Crop Recommendation System

## Project Overview

The SmartFarm Crop Recommendation System is an end-to-end machine learning project designed to advise farmers on the most suitable crop to plant based on soil nutrient profiles ($\text{N, P, K, pH}$) and localized climatic conditions ($\text{temperature, humidity, rainfall}$). This classification solution aims to optimize agricultural output, reduce resource waste, and mitigate planting risk, contributing to the field of Precision Agriculture.

The project follows a full MLOps lifecycle, moving from data analysis and model selection to containerization and local deployment using Docker.

## 🌟 Key Features

* **Data Preparation & $\text{EDA}$**: Comprehensive analysis of the dataset, including feature distribution, balancing checks, and necessary data scaling.

* **Multi-Model Training**: Compares the performance of three classic classification algorithms: Logistic Regression, K-Nearest Neighbors ($\text{KNN}$), and Random Forest.

* **Hyperparameter Tuning**: Utilizes GridSearchCV on the best-performing model (Random Forest) to find the optimal parameter set, maximizing predictive accuracy.

* **Feature Importance**: Analyzes the contribution of each environmental factor, providing interpretability for the final model's decisions.

* **Model Export**: Saves the final, tuned model and the data scaler using pickle for easy integration into the $\text{API}$.

* **Web Service Deployment**: Serves real-time predictions via a lightweight Flask $\text{API}$ powered by Gunicorn.

* **Containerization**: Provides a production-ready **Docker** setup for easy, reproducible local deployment.

## 🛠️ Prerequisites

To run this project, you need the following installed:

1. **Python 3.8+**

2. **pip** (Python package installer)

3. **Docker** (for deploying the web service)

## 📥 Setup and Installation

### Step 1: Download the Dataset

Download the **Crop Recommendation Dataset** (Crop_recommendation.csv) from Kaggle and place it directly into the project root directory.

### Step 2: Ensure Project Files Exist

Verify that the following files are present in your project directory (these will be generated throughout our process):

**'README.md**(this file)

**requirements.txt**

**ml_pipeline.py**

**app.py**

**Dockerfile**

### Step 3: Install Dependencies

You will need the dependencies defined in the requirements.txt file (generated previously):

```bash
pip install -r requirements.txt
```

** 🚀 Execution Guide (End-to-End)

The project execution is divided into two primary phases: Model Training and $\text{API}$ Deployment.

### Phase 1: Train, Tune, and Export the Model (Run ml_pipeline.py)

This is the first step and is necessary to create the required model files.

Run the ML Pipeline:

```bash
python ml_pipeline.py
```

#### Pipeline Actions:

- Loads and preprocesses data.

- Trains and evaluates Logistic Regression, KNN, Decesion Trees and Random Forest.

- Selects Random Forest as the best base model.

- Tunes the Random Forest model using GridSearchCV.

- Prints the final metrics and feature importance.

#### Critical Output Files:
After successful execution, the script will create two critical files in the root directory:

**best_model.pkl**: The saved, highly-tuned Random Forest Classifier.

**scaler.pkl**: The fitted StandardScaler object, crucial for transforming incoming $\text{API}$ data.

### Phase 2: Deploy the Prediction Service with Docker (Run $\text{app.py}$)

Once the model files (best_model.pkl and scaler.pkl) are created, you can containerize and run the prediction $\text{API}$.

#### Build the Docker Image:

```bash
docker build -t smartfarm-api .
```

#### Run the Container:

This command runs the image in the background (-d) and maps the container's port 8080 to your local machine's port 8080 (-p).

```bash
docker run -d -p 8080:8080 --name smartfarm-service smartfarm-api
```

#### Verification (Optional):

Check the Docker logs to ensure the Gunicorn server successfully started and loaded the models:

```bash
docker logs smartfarm-service
```

#### 💻 Making Predictions (Using the Deployed API)

The prediction service is now running locally at http://localhost:8080. The API endpoint is **/predict** and accepts a **JSON payload**.

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
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '
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
  "prediction": "rice",
  "confidence_score": 0.99
}
```

#### 🛑 Stopping the Service

To stop and remove the running Docker container:

```bash
docker stop smartfarm-service
docker rm smartfarm-service
```
