import gdown
import zipfile
import os

# --- Download ---
url = "https://drive.google.com/uc?id=1Jb7yltxh7rwNcrCRycWYSIfercQ35Mov"
zip_path = "pole_dataset.zip"

gdown.download(url, zip_path, quiet=False)

# --- Create folder for extraction ---
extract_folder = "data/pole_dataset"
os.makedirs(extract_folder, exist_ok=True)

# --- Unzip ---
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print(f"Dataset extracted to: {extract_folder}")

# --- Remove zip file ---
os.remove(zip_path)
print("Zip file removed")