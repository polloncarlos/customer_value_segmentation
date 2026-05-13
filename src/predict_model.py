import os
import joblib

from dotenv import load_dotenv

from src.data_loader import load_data
from src.feature_engineering import transform_data
from src.deploy import save_to_rds


# ==============================
# CONFIG
# ==============================

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATA_PATH = os.getenv("S3_PATH")

ENDPOINT = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'


# ==============================
# MAIN
# ==============================

def main():

    print("🔹 [1/6] Carregando dados...")
    df = load_data(DATA_PATH)

    print("🔹 [2/6] Feature engineering...")
    df = transform_data(df)

    features = [
        'revenue_velocity',
        'total_items',
        'items_velocity',
        'return_orders',
        'product_loyalty'
    ]

    X = df[features].copy()

    print("🔹 [3/6] Carregando modelos...")
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans.pkl")

    print("🔹 [4/6] Scaling...")
    X_scaled = scaler.transform(X)

    print("🔹 [5/6] Gerando clusters...")
    df['cluster'] = kmeans.predict(X_scaled)

    print("🔹 [6/6] Salvando no RDS...")
    save_to_rds(df, ENDPOINT)

    print("✅ Predição finalizada com sucesso!")


if __name__ == "__main__":
    main()