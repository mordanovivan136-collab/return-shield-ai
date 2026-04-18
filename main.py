
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Return Shield AI API")

# Автоматическое создание модели, если файл не существует
MODEL_PATH = "return_shield_ai_v2.json"

if not os.path.exists(MODEL_PATH):
    print("Модель не найдена. Создаём новую...")
    np.random.seed(42)
    n = 10000

    data = {
        'customer_return_rate': np.random.beta(2, 5, n),
        'product_return_rate': np.random.beta(3, 7, n),
        'time_on_page': np.random.exponential(130, n),
        'items_in_cart': np.random.poisson(2.8, n),
        'size_mismatch': np.random.binomial(1, 0.38, n),
        'price': np.random.lognormal(4.6, 0.75, n),
    }

    df = pd.DataFrame(data)
    categories = ['джинсы', 'платья', 'верхняя одежда', 'футболки', 'обувь', 'аксессуары', 'брюки', 'юбки', 'свитеры', 'пальто']
    seasons = ['зима', 'весна', 'лето', 'осень']

    df['category'] = np.random.choice(categories, n, p=[0.18,0.14,0.15,0.12,0.10,0.08,0.08,0.07,0.05,0.03])
    df['season'] = np.random.choice(seasons, n)

    df = pd.get_dummies(df, columns=['category', 'season'], drop_first=True)

    logit = -5.0 + 6.0*df['customer_return_rate'] + 4.5*df['product_return_rate'] -0.006*df['time_on_page'] +             0.7*df['items_in_cart'] + 3.0*df['size_mismatch'] -0.002*df['price']

    prob = 1 / (1 + np.exp(-logit))
    df['return'] = (np.random.rand(n) < prob).astype(int)

    X = df.drop('return', axis=1)
    y = df['return']

    model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, eval_metric='logloss')
    model.fit(X, y)
    model.save_model(MODEL_PATH)
    print("Модель успешно создана и сохранена")
else:
    print("Модель загружена из файла")

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
    data = pd.DataFrame([order.dict()])
    data = pd.get_dummies(data, columns=['category', 'season'], drop_first=True)

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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
