import os

# --- Répertoire de destination ---
download_path = "speech_commands_data"

# Créer le dossier si n'existe pas
os.makedirs(download_path, exist_ok=True)

# --- Téléchargement du dataset Kaggle ---
print("Téléchargement en cours...")

os.system(f'kaggle datasets download -d yashdogra/speech-commands -p {download_path} --unzip')

print("Téléchargement terminé !")
