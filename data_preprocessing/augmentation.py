# =======================
# Блок 2: Аугментация изображений
# =======================
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import shutil

# Определяем список аугментаций
augmentations = [
    {"name": "horizontal_flip", "transform": A.HorizontalFlip(p=1.0)},
    {"name": "shear_left", "transform": A.Affine(shear={'x': (-15, -15), 'y': (0, 0)}, p=1.0)},
    {"name": "shear_right", "transform": A.Affine(shear={'x': (15, 15), 'y': (0, 0)}, p=1.0)},
    {"name": "brightness_increase", "transform": A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3), contrast_limit=0, p=1.0)},
    {"name": "brightness_decrease", "transform": A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.2), contrast_limit=0, p=1.0)},
    {"name": "gauss_noise", "transform": A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)},
]
augmentation_prob = 0.5  # вероятность аугментации для каждой исходной пары

# Создаем локальные папки для аугментированных данных
aug_images_dir = "/content/augmented_images"
aug_labels_dir = "/content/augmented_labels"
os.makedirs(aug_images_dir, exist_ok=True)
os.makedirs(aug_labels_dir, exist_ok=True)

augmented_pairs = []  # сюда будем сохранять новые пары (путь к аугментированному изображению, путь к аугментированному лейблу)

for orig_img_path, orig_lbl_path in tqdm(pairs, desc="Augmenting images"):
    # Загружаем изображение
    image = cv2.imread(orig_img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base_name = os.path.basename(orig_img_path)
    file_name, ext = os.path.splitext(base_name)
    
    # Применяем аугментацию с заданной вероятностью
    if random.random() < augmentation_prob:
        for aug in augmentations:
            augmented = aug["transform"](image=image)
            aug_image = augmented['image']
            aug_filename = f"{aug['name']}_{file_name}{ext}"
            aug_image_path = os.path.join(aug_images_dir, aug_filename)
            # Сохраняем аугментированное изображение (конвертируем обратно в BGR для cv2.imwrite)
            cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            # Для лейблов просто копируем исходный файл с новым именем
            aug_label_filename = f"{aug['name']}_{file_name}.txt"
            aug_label_path = os.path.join(aug_labels_dir, aug_label_filename)
            shutil.copy(orig_lbl_path, aug_label_path)
            augmented_pairs.append((aug_image_path, aug_label_path))
            
# Объединяем исходные пары с аугментированными
all_pairs = pairs + augmented_pairs
print(f"Всего пар после аугментации: {len(all_pairs)}")


