import joblib
import os

from dotenv import load_dotenv
from src.data_loader import load_data
from src.feature_engineering import transform_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

import umap


# ==============================
# CONFIG
# ==============================

load_dotenv()

DATA_PATH = os.getenv("S3_PATH")


# ==============================
# MAIN
# ==============================
def main():

    print("🔹 [1/8] Carregando dados...")
    df = load_data(DATA_PATH)

    print("🔹 [2/8] Feature engineering...")
    df = transform_data(df)

    features = [
        'revenue_velocity',
        'total_items',
        'items_velocity',
        'return_orders',
        'product_loyalty'
    ]

    X = df[features].copy()
    y = df['gross_revenue']

    print("🔹 [3/8] Treinando RandomForest...")

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X, y)

    print("🔹 [4/8] Gerando embedding...")
    leafs = rf.apply(X)

    print("🔹 [5/8] OneHotEncoder...")

    ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore',
        max_categories=50
    )

    embedding = ohe.fit_transform(leafs)

    print("🔹 [6/8] UMAP...")

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        random_state=42
    )

    embedding_reduced = reducer.fit_transform(embedding)

    print("🔹 [7/8] KMeans...")

    kmeans = KMeans(
        n_clusters=5,
        n_init=50,
        max_iter=300,
        random_state=42
    )

    kmeans.fit(embedding_reduced)

    print("🔹 [8/8] Salvando modelos...")

    os.makedirs("models", exist_ok=True)

    # ✅ SALVA SOMENTE MODELOS LEVES
    joblib.dump(rf, "models/rf.pkl")
    joblib.dump(ohe, "models/ohe.pkl")
    joblib.dump(kmeans, "models/kmeans.pkl")

    print("✅ Treino finalizado!")
    print("✅ Modelos salvos em /models")


if __name__ == "__main__":
    main()