from xgboost import XGBClassifier
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv('../data/synthetic_financial_data.csv')

X = df.drop('label', axis=1)
y = df['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train_encoded)

model_directory = '../trained_models'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model_path = os.path.join(model_directory, 'xgboost_model.bin')
joblib.dump(model, model_path)
label_encoder_path = os.path.join(model_directory, 'label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_path)

preprocessor_path = os.path.join(model_directory, 'preprocessor.pkl')
joblib.dump(preprocessor, preprocessor_path)
