
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import uvicorn

app = FastAPI(title="Return Shield AI API")

# Загружаем модель
model = xgb.XGBClassifier()
model.load_model("return_shield_ai_v2.json")

class OrderInput(BaseModel):
    customer_return_rate: float
    product_return_rate: float
    time_on_page: float
    items_in_cart: int
    size_mismatch: int
    price: float
    category: str
    season: str

@app.post("/predict")
def predict_return(order: OrderInput):
    data = pd.DataFrame([order.dict()])
    data = pd.get_dummies(data, columns=["category", "season"], drop_first=True)

    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in data.columns:
            data[col] = 0
    data = data[model_features]

    risk = model.predict_proba(data)[0][1]
    risk_percent = round(risk * 100, 1)

    if risk < 0.40:
        recommendation = "Низкий риск — можно отправлять заказ"
        action = "normal"
    elif risk < 0.65:
        recommendation = "Средний риск — показать рекомендации по размеру"
        action = "recommend"
    else:
        recommendation = "Высокий риск — предложить скидку или Store Credit"
        action = "discount"

    return {
        "risk_percent": risk_percent,
        "recommendation": recommendation,
        "suggested_action": action,
        "estimated_savings_rub": round(risk * 1250, 0),
        "co2_savings_kg": round(risk * 28.5, 1)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
