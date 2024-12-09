import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

def transform_data(dataset, encoders=None, scaler=None, is_training=True):
    """
    Prepare and transform the dataset by creating features, encoding categorical values,
    and scaling numerical columns.
    """
    # Feature generation: Date-time components
    dataset['transaction_date'] = pd.to_datetime(dataset['unix_time'], unit='s')
    dataset['transaction_hour'] = dataset['transaction_date'].dt.hour
    dataset['weekday'] = dataset['transaction_date'].dt.weekday
    dataset['day_of_year'] = dataset['transaction_date'].dt.dayofyear
    dataset['is_weekend'] = dataset['transaction_date'].dt.weekday >= 5

    # Derive user age in years
    dataset['dob'] = pd.to_datetime(dataset['dob'])
    dataset['user_age'] = (dataset['transaction_date'] - dataset['dob']).dt.days // 365

    # User-level aggregated features
    dataset['total_transactions'] = dataset.groupby('id')['id'].transform('count')
    dataset['mean_transaction_amt'] = dataset.groupby('id')['amt'].transform('mean')
    dataset['amt_residual'] = dataset['amt'] - dataset['mean_transaction_amt']
    dataset['amt_scaled'] = dataset['amt'] * dataset['total_transactions']

    # Drop redundant columns
    dataset.drop(columns=['unix_time', 'transaction_date'], inplace=True, errors='ignore')

    # Handle categorical fields
    categorical_columns = ['category', 'gender', 'state', 'job', 'merchant']
    if is_training:
        encoders = {col: LabelEncoder() for col in categorical_columns}
        for col, encoder in encoders.items():
            dataset[col] = encoder.fit_transform(dataset[col])
    else:
        for col, encoder in encoders.items():
            dataset[col] = encoder.transform(dataset[col])

    # Normalize numeric features
    numeric_columns = ['amt', 'mean_transaction_amt', 'amt_residual', 'amt_scaled']
    if is_training:
        scaler = MinMaxScaler()
        dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])
    else:
        dataset[numeric_columns] = scaler.transform(dataset[numeric_columns])

    # Features to use for training and prediction
    features = [
        'category', 'amt', 'transaction_hour', 'weekday', 'gender', 'state',
        'city_pop', 'job', 'day_of_year', 'is_weekend', 'total_transactions',
        'user_age', 'mean_transaction_amt', 'amt_residual', 'amt_scaled'
    ]

    return dataset[features], encoders, scaler


# Load the training dataset
training_data = pd.read_csv('train.csv')
X_train, categorical_encoders, feature_scaler = transform_data(training_data, is_training=True)
y_train = training_data['is_fraud']

# Define the optimized XGBoost model with GPU acceleration
xgb_model = xgb.XGBClassifier(
    n_estimators=500,              # Increased number of boosting rounds
    max_depth=8,                   # Increased tree depth
    learning_rate=0.05,            # Reduced learning rate for finer updates
    subsample=0.85,                # Slightly increased subsample ratio
    colsample_bytree=0.9,          # Increased feature sampling ratio per tree
    gamma=1.0,                     # Minimum loss reduction to make a split
    reg_alpha=0.5,                 # L1 regularization term
    reg_lambda=1.0,                # L2 regularization term
    objective='binary:logistic',   # Binary classification objective
    tree_method='gpu_hist',        # GPU-accelerated histogram algorithm
    predictor='gpu_predictor',     # GPU for predictions
    random_state=40                # Fixed random state for reproducibility
)

# Split training data into train and validation subsets
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
    X_train, y_train, test_size=0.3, stratify=y_train, random_state=40
)

# Train the model
xgb_model.fit(X_train_split, y_train_split)

# Evaluate on validation set
val_predictions = xgb_model.predict(X_valid_split)
validation_accuracy = accuracy_score(y_valid_split, val_predictions)
validation_f1 = f1_score(y_valid_split, val_predictions)
validation_confusion = confusion_matrix(y_valid_split, val_predictions)
validation_report = classification_report(y_valid_split, val_predictions)

print("Validation Accuracy:", validation_accuracy)
print("F1 Score:", validation_f1)
print("Confusion Matrix:\n", validation_confusion)
print("Classification Report:\n", validation_report)

# Load test dataset
test_data = pd.read_csv('test.csv')
X_test, _, _ = transform_data(test_data, encoders=categorical_encoders, scaler=feature_scaler, is_training=False)

# Make predictions for test data
test_data['is_fraud'] = xgb_model.predict(X_test)

# Save results to CSV
test_data[['id', 'is_fraud']].to_csv('fraud_predictions.csv', index=False)

print("Predictions exported to: fraud_predictions.csv")
