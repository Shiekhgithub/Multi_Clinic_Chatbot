"""
ingest.py
---------
One-time script: loads the three CSVs, converts them to Documents,
and builds the ChromaDB vector stores on disk.

Run:
    python ingest.py
"""

import os
from dotenv import load_dotenv
from data_ingestion import load_heart_disease, load_dermatology, load_diabetes
from vector_store import build_all_stores

load_dotenv()


def main():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    heart_csv   = os.getenv("HEART_DISEASE_CSV",  "./data/heart.csv")
    derm_csv    = os.getenv("DERMATOLOGY_CSV",     "./data/dermatology.csv")
    diab_csv    = os.getenv("DIABETES_CSV",        "./data/diabetes.csv")

    # Validate file paths
    for path in [heart_csv, derm_csv, diab_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CSV not found: {path}\n"
                "Please download the datasets from Kaggle and place them in ./data/"
            )

    print("=" * 60)
    print("  Healthcare RAG — Data Ingestion")
    print("=" * 60)

    print("\n[1/3] Loading Heart Disease dataset …")
    heart_docs = load_heart_disease(heart_csv)
    print(f"      → {len(heart_docs)} documents created.")

    print("\n[2/3] Loading Dermatology dataset …")
    derm_docs = load_dermatology(derm_csv)
    print(f"      → {len(derm_docs)} documents created.")

    print("\n[3/3] Loading Pakistani Diabetes dataset …")
    diab_docs = load_diabetes(diab_csv)
    print(f"      → {len(diab_docs)} documents created.")

    print("\n[Building vector stores …]")
    build_all_stores(persist_dir, heart_docs, derm_docs, diab_docs)

    print("\n✅  All vector stores built successfully!")
    print(f"    Stored in: {os.path.abspath(persist_dir)}")
    print("\nYou can now run the chatbot:")
    print("    streamlit run app.py       ← Streamlit UI")
    print("    python cli_chat.py         ← Terminal chat")


if __name__ == "__main__":
    main()
