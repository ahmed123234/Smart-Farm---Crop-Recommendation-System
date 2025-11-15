import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


# --- 1. Data Loading and Preparation Functions ---
def load_data(): 
  DATA_URL = "https://raw.githubusercontent.com/ahmed123234/Smart-Farm---Crop-Recommendation-System/refs/heads/main/Crop_recommendation.csv"
    
  # Load data
  try:
    df = pd.read_csv(DATA_URL, na_values=['?'], index_col=0)
    print(f"Data loaded successfully. Total records: {len(df)}")
  except FileNotFoundError:
    print(f"Error: {DATA_URL} not found. Please ensure the dataset is in the same directory.")
  exit()

  return df


def load_and_preprocess_data():
    """Loads, cleans, and separates the data."""
    
    df = load_data()
  
    # 1. Cleaning and Target Definition
    print("\n--- 2.2. Checking for Missing Values ---")
    print(df.isnull().sum())
    print("-" * 30)
    if df.isnull().sum().any():
        print("Addressing Missing Values")
        df.fillna(0, inplace=True)
        print("Missing values replaced with 0.")
    else:
        print("No missing values found. Data quality is high.")
    
    print("\n--- 2.3. Fix Data Types ---")
    print(df.dtypes)
    print("-" * 30)
    
    print("\n--- 2.4 Identify and remove exact duplicate rows that represent the same entity ---")
    df.drop_duplicates(inplace=True)
    print("Duplicate rows are equal to {}".format(df.duplicated().sum()))
    print("-" * 30)

    print("\n--- 2.5. Target Variable (Crop) Distribution ---")
    # Check how many samples we have for each crop
    print(df['label'].value_counts())
    print("-" * 30)
    # Insight: A balanced dataset is key for classification.
    if df['label'].value_counts().std() < 5: # Small threshold for standard deviation
        print("Dataset is perfectly balanced (100 records per crop), ideal for training.")
    else:
        print("CAUTION: Dataset is imbalanced. May require techniques like oversampling/undersampling.")
    
    # Feature set definition
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[features]
    y = df['label']
    
    print("\n--- 2.6. Feature Analysis (Statistical Summary) ---")
    # Statistics reveal range, mean, and potential outliers
    print(X.describe().T)
    print("-" * 30)
  
    return X, y


# --- 2. Main Execution ---
if __name__ == "__main__":
    print("Starting Model Training for Readmission Risk Prediction...")
    
    # 2.1. Load and prepare data
    X, y = load_and_preprocess_data()
    
    # Split data for final testing (though the full pipeline will be trained on X/y for maximum data)
    # We use a test set to confirm final performance before saving.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling features (Essential for distance-based models like KNN and helpful for LR)
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
  
    # 2.2 Model Definition
    # Random Forest with parameters tuned
    RF_model = RandomForestClassifier(
        random_state=42,
        criterion='gini',
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=10,
        n_estimators=300
    )
    print("Training the Random Forest model...")
    
    # Train the pipeline (preprocessing steps are fitted on X_train first)
    RF_model.fit(X_train, y_train)

    # 2.3. Evaluate on the Test set
    y_pred = RF_model.predict(X_test)
    final_f1 = f1_score(y_test, y_pred)
    
    print(f"\nTraining Complete.")
    print(f"Model Performance (F1-Score on Test Set): {final_f1:.4f}")
    
    # 2.4. Save the trained pipeline
    MODEL_FILENAME = 'crop_recommendation_model_pipeline.pkl'
    with open(MODEL_FILENAME, 'wb') as file:
        pickle.dump(RF_model, file)
        
    print(f"Successfully saved the model pipeline to '{MODEL_FILENAME}'")
