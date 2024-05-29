import os
from roboflow import Roboflow

# Roboflow API anahtarınızla Roboflow'u başlatın
rf = Roboflow(api_key="FTKEGatiDsdWWrRoJ2aQ")
project = rf.workspace("alper-fbc0d").project("tumor-detection2-rscmi")
model = project.version(2).model




# Görüntü dosyalarının bulunduğu klasör
input_folder = "trainyes2"
# Tahminlerin kaydedileceği klasör
output_folder = "pred"

# Eğer çıktı klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Klasördeki tüm dosyaları işle
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Sadece jpg ve png dosyalarını işle
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, "predicted_" + filename)

        # Tahmini yap ve kaydet
        model.predict(input_path, confidence=40, overlap=30).save(output_path)
        print(f"{filename} işlendi ve {output_path} olarak kaydedildi.")


