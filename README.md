# 🌾 SmartFarm Crop Recommendation System

## Project Overview

The SmartFarm Crop Recommendation System is an end-to-end machine learning project designed to advise farmers on the most suitable crop to plant based on soil nutrient profiles ($\text{N, P, K, pH}$) and localized climatic conditions ($\text{temperature, humidity, rainfall}$). This solution uses a trained classification model to optimize agricultural output, reduce resource waste, and mitigate planting risk.

The project follows a full MLOps lifecycle, moving from data analysis and model selection to containerization and local deployment using Docker.

## 🌟 Key Features

* **Multi-Model Training:** Compares performance of Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest.
* **Hyperparameter Tuning:** Uses `GridSearchCV` on the best model to maximize accuracy.
* **Feature Importance:** Analyzes the contribution of each environmental factor ($\text{N, P, K}$, $\text{pH}$, etc.).
* **Web Service Deployment:** Serves the model via a lightweight **Flask $\text{API}$** powered by Gunicorn.
* **Containerization:** Provides a production-ready **Docker** setup for easy local deployment and scaling.

---

## 🛠️ Prerequisites

To run this project, you need the following installed:

1.  **Python 3.8+**
2.  **pip** (Python package installer)
3.  **Docker** (for deploying the web service)

---

## 📥 Setup and Installation

### Step 1: Download the Dataset

Download the **Crop Recommendation Dataset** (`Crop_recommendation.csv`) from Kaggle and place it directly into the project root directory.

### Step 2: Install Dependencies

You will need the dependencies defined in the `requirements.txt` file (generated previously):

```bash
pip install -r requirements.txt
