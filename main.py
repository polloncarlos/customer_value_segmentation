import sys

from src.train_model import main as train_pipeline
from src.predict_model import main as predict_pipeline


# ==============================
# MAIN ORCHESTRATOR
# ==============================
def main():

    if len(sys.argv) < 2:

        print("\n❌ Nenhum comando informado.")
        print("\nUse:")
        print("python main.py train")
        print("python main.py predict\n")

        return

    command = sys.argv[1].lower()

    # ==========================
    # TRAIN
    # ==========================
    if command == "train":

        print("\n🚀 EXECUTANDO PIPELINE DE TREINO\n")

        train_pipeline()

    # ==========================
    # PREDICT
    # ==========================
    elif command == "predict":

        print("\n🚀 EXECUTANDO PIPELINE DE PREDIÇÃO\n")

        predict_pipeline()

    # ==========================
    # INVALID COMMAND
    # ==========================
    else:

        print(f"\n❌ Comando inválido: {command}")

        print("\nUse:")
        print("python main.py train")
        print("python main.py predict\n")


# ==============================
# EXECUTION
# ==============================
if __name__ == "__main__":
    main()