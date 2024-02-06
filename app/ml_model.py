from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

class MLModel:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def predict(self, X):
        return self.model.predict(X)
