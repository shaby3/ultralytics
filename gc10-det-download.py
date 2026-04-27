import kagglehub
import os

# Download latest version
os.environ["KAGGLEHUB_CACHE"] = r"C:\Users\SSAFY\ultralytics\gc10"
path = kagglehub.dataset_download("alex000kim/gc10det")

print("Path to dataset files:", path)