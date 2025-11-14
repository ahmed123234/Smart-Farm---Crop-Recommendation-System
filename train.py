import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

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
        print("✅ Missing values replaced with 0.")
    else:
        print("✅ No missing values found. Data quality is high.")
    
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
        print("✅ Dataset is perfectly balanced (100 records per crop), ideal for training.")
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


    # Simple Label Encoding for binary features
    # le = LabelEncoder()
    # for col in ['change', 'diabetesMed']:
    #     df[col] = le.fit_transform(df[col])
    
    # df = df[df['gender'] != 'Unknown/Invalid']
    # df['gender'] = le.fit_transform(df['gender'])

    # y = df['READMIT_30_DAYS']
    # X = df.drop('READMIT_30_DAYS', axis=1)
    
    return X, y

def create_pipeline(X, y):
    """
    Defines the full end-to-end pipeline: Preprocessing + Model.
    """
    # Separate feature types
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # 1. Preprocessing Pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column Transformer to apply transformations to correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 2. Model Definition
    # XGBoost with parameters tuned to handle class imbalance (scale_pos_weight)
    xgb_model = XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss', 
        scale_pos_weight=9, # Approx ratio of negatives to positives (0s/1s)
        n_estimators=200,    # From light tuning/default choice
        max_depth=5,         # From light tuning/default choice
        learning_rate=0.05
    )

    # 3. Final Pipeline: Preprocessor -> Model
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])
    
    return full_pipeline

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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
  
    # 2.2. Create and train the full pipeline
    # full_pipeline = create_pipeline(X, y)
    model = RandomForestClassifier(random_state=42, criterion='gini', max_depth=10, max_features='sqrt', n_estimators=300)
    
    print("Training the full XGBoost Pipeline...")
    # Train the pipeline (preprocessing steps are fitted on X_train first)
    model.fit(X_train, y_train)

    # 2.3. Evaluate on the Test set
    y_pred = full_pipeline.predict(X_test)
    final_f1 = f1_score(y_test, y_pred)
    
    print(f"\nTraining Complete.")
    print(f"Model Performance (F1-Score on Test Set): {final_f1:.4f}")
    
    # 2.4. Save the trained pipeline
    MODEL_FILENAME = 'crop_recommendation_model_pipeline.pkl'
    with open(MODEL_FILENAME, 'wb') as file:
        pickle.dump(full_pipeline, file)
        
    print(f"Successfully saved the model pipeline to '{MODEL_FILENAME}'")
    print("This file contains both the preprocessor and the trained XGBoost model.")
