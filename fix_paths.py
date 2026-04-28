import pandas as pd
import os

def fix():
    csv_path = "data/processed/registered_pairs.csv"
    if not os.path.exists(csv_path):
        print("Error: CSV not found")
        return
        
    df = pd.read_csv(csv_path)
    # Ensure we use the non-normalized, registered stained images
    # because they proved to be the most stable.
    df_final = pd.DataFrame({
        "unstained": df["unstained"],
        "stained": df["stained"]
    })
    df_final.to_csv("data/processed/final_training_pairs.csv", index=False)
    print("Metadata fixed: Now pointing to stable Registered Raw images.")

if __name__ == "__main__":
    fix()
