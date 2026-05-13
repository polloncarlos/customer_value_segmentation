import os
import joblib

from dotenv import load_dotenv

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.feature_engineering import transform_data


# ==============================
# CONFIG
# ==============================

load_dotenv()

DATA_PATH = os.getenv("S3_PATH")


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

    print("🔹 [3/6] Scaling...")
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("🔹 [4/6] Treinando KMeans...")
    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=50
    )

    kmeans.fit(X_scaled)

    print("🔹 [5/6] Salvando modelos...")

    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(kmeans, "models/kmeans.pkl")

    print("🔹 [6/6] Gerando clusters finais...")

    df['cluster'] = kmeans.predict(X_scaled)

    print("✅ Treino finalizado com sucesso!")


if __name__ == "__main__":
    main()