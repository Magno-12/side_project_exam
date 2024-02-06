from fastapi import FastAPI, HTTPException
import pandas as pd
from app.data_preprocessing import preprocess_data
from app.recommendation_engine import RecommendationEngine
from app.models import UserInputSchema

app = FastAPI()
recommendation_engine = RecommendationEngine(model_path='./trained_models/xgboost_model.bin')

@app.post("/recommendation/")
async def get_recommendation(user_input: UserInputSchema):
    user_data_df = pd.DataFrame([dict(user_input)])
    try:
        preprocessed_data = preprocess_data(user_data_df)
        recommendations = recommendation_engine.generate_recommendations(preprocessed_data)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
