import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


# --- 1. Data Loading and Preparation Functions ---
def load_data(): 
    """Loads the dataset from the specified URL."""
    DATA_URL = "https://raw.githubusercontent.com/ahmed123234/Smart-Farm---Crop-Recommendation-System/refs/heads/main/Crop_recommendation.csv"
    
    try:
        # Load data
        df = pd.read_csv(DATA_URL)
        print(f"Data loaded successfully. Total records: {len(df)}")
    except Exception as e:
        print(f"Error loading data: {e}. Exiting.")
        exit()

    return df


def load_and_preprocess_data():
    """Loads, cleans, and separates the data."""
    
    df = load_data()
    print("\n--- Initial Data Overview ---")
    print(df.head())
    print("-" * 30)
  
    # 1. Cleaning and Target Definition
    print("\n--- Checking for Missing Values ---")
    print(df.isnull().sum())
    print("-" * 30)
    if df.isnull().sum().any():
        print("Addressing Missing Values")
        df.fillna(0, inplace=True)
        print("Missing values replaced with 0.")
    else:
        print("No missing values found. Data quality is high.")
    
    print("\n--- Fix Data Types ---")
    print(df.dtypes)
    print("-" * 30)
    
    print("\n--- Identify and remove exact duplicate rows that represent the same entity ---")
    df.drop_duplicates(inplace=True)
    print("Duplicate rows are equal to {}".format(df.duplicated().sum()))
    print("-" * 30)

    print("\n--- Target Variable (Crop) Distribution ---")
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
    
    print("\n--- Feature Analysis (Statistical Summary) ---")
    # Statistics reveal range, mean, and potential outliers
    print(X.describe().T)
    print("-" * 30)
  
    return X, y

def create_pipeline(X):
    """
    Defines the full end-to-end pipeline: Preprocessing (Scaling) + Model (RandomForest).
    """
    
    # 1. Define Features to be scaled
    numerical_features = X.columns.tolist() 

    # 2. Preprocessing Pipeline: Scaling
    # We only have numerical features, so we define a simple ColumnTransformer for scaling.
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='passthrough' # Ensure no columns are dropped
    )
    
    # 3. Model Definition (using the user's chosen parameters)
    RF_model = RandomForestClassifier(
        random_state=42,
        criterion='gini',
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=10,
        n_estimators=300
    )

    # 4. Final Pipeline: Preprocessor -> Model
    # This Pipeline ensures that when 'predict' is called later, the raw data is scaled first.
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RF_model)
    ])
    
    return full_pipeline

# --- 2. Main Execution ---
if __name__ == "__main__":
    print("Starting Crop Recommendation Model Training...")
    
    # 2.1. Load and prepare data
    X, y = load_and_preprocess_data()
    
    # Split data for training/testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2.2. Create and train the full pipeline
    full_pipeline = create_pipeline(X)
    
    print("Training the full Random Forest Pipeline...")
    full_pipeline.fit(X_train, y_train)

    # 2.3. Evaluate on the Test set
    y_pred = full_pipeline.predict(X_test)
    
    # Calculate F1-Score for multi-class classification
    final_f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = full_pipeline.score(X_test, y_test)

    print(f"\nTraining Complete.")
    print(f"Model Performance (Accuracy on Test Set): {accuracy:.4f}")
    print(f"Model Performance (Weighted F1-Score): {final_f1:.4f}")
    
    # 2.4. Save the trained pipeline
    MODEL_FILENAME = 'crop_recommendation_model_pipeline.pkl'
    with open(MODEL_FILENAME, 'wb') as file:
        pickle.dump(full_pipeline, file)
        
    print(f"\nSuccessfully saved the COMPLETE PIPELINE to '{MODEL_FILENAME}'")
