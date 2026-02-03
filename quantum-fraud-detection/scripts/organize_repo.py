
import os
import shutil

def organize_repo():
    # 1. Configs
    if os.path.exists("configs/config_test_qml.yaml"):
        print("Removing config_test_qml.yaml")
        os.remove("configs/config_test_qml.yaml")
    
    if os.path.exists("configs/env_template.txt"):
        print("Renaming env_template.txt to .env.example")
        os.rename("configs/env_template.txt", "configs/.env.example")

    # 2. Notebooks
    if not os.path.exists("notebooks/archive"):
        os.makedirs("notebooks/archive")
    
    nb_to_archive = [
        "IBMQiskit.ipynb", "newfraud.ipynb", "quantum_fraud_detection_colab.ipynb"
    ]
    for nb in nb_to_archive:
        src = os.path.join("notebooks", nb)
        dst = os.path.join("notebooks/archive", nb)
        if os.path.exists(src):
            print(f"Archiving {nb}...")
            shutil.move(src, dst)
            
    print("✅ Organization Complete.")

if __name__ == "__main__":
    organize_repo()
