import os
import zipfile
import urllib.request

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-ar-mgb2-0.4.zip"
MODEL_ZIP = "vosk-model-ar-mgb2-0.4.zip"
MODEL_DIR = "model-ar"

def download_model():
    if os.path.exists(MODEL_DIR):
        print(f"'{MODEL_DIR}' already exists. No download needed.")
        return
    print("Downloading Vosk Arabic model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_ZIP)
    print("Extracting model...")
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(".")
    for name in os.listdir("."):
        if name.startswith("vosk-model-ar-mgb2"):
            os.rename(name, MODEL_DIR)
            break
    os.remove(MODEL_ZIP)
    print(f"Model downloaded and extracted as '{MODEL_DIR}'.")

if __name__ == "__main__":
    download_model()