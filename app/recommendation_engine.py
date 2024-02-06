import pandas as pd
from .ml_model import MLModel
from .data_preprocessing import preprocess_data

class RecommendationEngine:
    def __init__(self, model_path: str):
        self.ml_model = MLModel()
        self.ml_model.load(model_path)

    def generate_recommendations(self, user_data: pd.DataFrame) -> list:
        preprocessed_data = preprocess_data(user_data)
        predictions = self.ml_model.predict(preprocessed_data)
        return self.map_predictions_to_recommendations(predictions)

    def map_predictions_to_recommendations(self, predictions: pd.Series) -> list:
        # Aquí mapearías las predicciones a recomendaciones textuales
        # Ejemplo muy simplificado:
        recommendations_map = {
            0: "Considera aumentar tus ahorros.",
            1: "Puede ser un buen momento para reducir tus gastos.",
            2: "Explora opciones de inversión para maximizar tus activos.",
            # ... más mapeos según tus categorías
        }
        return [recommendations_map.get(prediction, "Recomendación no disponible.") for prediction in predictions]
