
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Return Shield AI API")

MODEL_PATH = "return_shield_ai_v2.json"

# Создаём модель, если её нет
if not os.path.exists(MODEL_PATH):
    print("Создаём модель...")
    np.random.seed(42)
    n = 8000

    df = pd.DataFrame({
        'customer_return_rate': np.random.beta(2, 5, n),
        'product_return_rate': np.random.beta(3, 7, n),
        'time_on_page': np.random.exponential(130, n),
        'items_in_cart': np.random.poisson(2.8, n),
        'size_mismatch': np.random.binomial(1, 0.4, n),
        'price': np.random.uniform(1000, 15000, n),
    })

    categories = ['футболки', 'джинсы', 'платья', 'обувь', 'верхняя одежда']
    seasons = ['зима', 'лето']

    df['category'] = np.random.choice(categories, n)
    df['season'] = np.random.choice(seasons, n)

    df = pd.get_dummies(df, columns=['category', 'season'])

    # Простая целевая переменная
    df['return'] = (df['customer_return_rate'] > 0.35).astype(int)

    X = df.drop('return', axis=1)
    y = df['return']

    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='logloss')
    model.fit(X, y)
    model.save_model(MODEL_PATH)
    print("Модель создана")
else:
    print("Модель загружена")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

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
    try:
        data = pd.DataFrame([order.dict()])
        data = pd.get_dummies(data, columns=['category', 'season'])

        # Приводим к колонкам модели
        model_features = model.get_booster().feature_names
        for col in model_features:
            if col not in data.columns:
                data[col] = 0
        data = data[model_features]

        risk = float(model.predict_proba(data)[0][1])
        risk_percent = round(risk * 100, 1)

        if risk < 0.4:
            rec = "Низкий риск — отправлять обычным способом"
        elif risk < 0.65:
            rec = "Средний риск — показать рекомендации по размеру"
        else:
            rec = "Высокий риск — предложить скидку или store credit"

        return {
            "risk_percent": risk_percent,
            "recommendation": rec,
            "estimated_savings_rub": round(risk * 1250),
            "co2_savings_kg": round(risk * 28.5, 1)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
