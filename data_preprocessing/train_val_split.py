# =======================
# Блок 3: Разбиение на train и val
# =======================
random.seed(42)
random.shuffle(all_pairs)
n_total = len(all_pairs)
n_train = int(n_total * 0.8)
train_pairs = all_pairs[:n_train]
val_pairs = all_pairs[n_train:]

print(f"Train пар: {len(train_pairs)}, Validation пар: {len(val_pairs)}")

# Извлекаем списки путей
train_images = [img for img, lbl in train_pairs]
train_labels = [lbl for img, lbl in train_pairs]
val_images = [img for img, lbl in val_pairs]
val_labels = [lbl for img, lbl in val_pairs]

# Формируем итоговые словари с данными для передачи в модель
train = {
    "images": train_images,
    "labels": train_labels,
    "classes": classes
}
val = {
    "images": val_images,
    "labels": val_labels,
    "classes": classes
}

print("Первые 3 train изображения:", train["images"][:3])
print("Первые 3 train аннотации:", train["labels"][:3])
print("Классы:", train["classes"])
