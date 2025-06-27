import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from train_model import Food101Dataset

# Конфигурация
FOOD_101_ROOT_PATH = os.getenv("FOOD_101_ROOT_PATH")
DB_PATH = os.path.join(FOOD_101_ROOT_PATH, "food_101_metadata.db")
MODEL_PATH = os.path.join(os.path.dirname(FOOD_101_ROOT_PATH), "models", "best_efficientnet_b0_final.pth")

BATCH_SIZE = 256
IMAGE_SIZE = 224

# Трансформации данных
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Функция для построения матрицы ошибок
def plot_confusion_matrix(cm, class_names, save_path, top_n_classes=None):
    if top_n_classes and top_n_classes < len(class_names):
        # Выбираем топ-N классов по количеству ошибок
        errors = np.sum(cm - np.diag(np.diag(cm)), axis=1)
        top_indices = np.argsort(errors)[-top_n_classes:]
        cm = cm[top_indices][:, top_indices]
        class_names = [class_names[i] for i in top_indices]
    
    plt.figure(figsize=(max(len(class_names) * 0.5, 10), max(len(class_names) * 0.5, 10)))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Добавляем аннотации
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        if cm[i, j] > 0:  # Показываем только ненулевые значения
            plt.text(j, i, f"{cm[i, j]:d}",
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Матрица ошибок сохранена по пути: {save_path}")

# Функция для вывода топ-N ошибок
def print_top_n_errors(cm, class_names, n=10):
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append((cm[i, j], class_names[i], class_names[j]))
    errors = sorted(errors, reverse=True)[:n]
    print(f"\nТоп-{n} ошибок:")
    for count, true, pred in errors:
        print(f"Истинный: {true}, Предсказанный: {pred}, Кол-во: {count}")

# Функция для отображения примеров предсказаний
def plot_predictions(dataset, model, device, class_names, num_examples=10, correct=True, save_path=None):
    fig = plt.figure(figsize=(15, 8))
    plt.suptitle(f"Примеры {'правильных' if correct else 'неправильных'} предсказаний", fontsize=16)

    model.eval()
    examples_shown = 0
    with torch.no_grad():
        indices = np.random.permutation(len(dataset))  # Случайный порядок для разнообразия
        for i in indices:
            if examples_shown >= num_examples:
                break

            image, true_label_idx = dataset[i]
            if image is None or true_label_idx is None:
                continue
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            _, predicted_label_idx = torch.max(output, 1)

            is_correct = (predicted_label_idx.item() == true_label_idx)

            if (correct and is_correct) or (not correct and not is_correct):
                ax = fig.add_subplot(2, num_examples // 2, examples_shown + 1, xticks=[], yticks=[])
                img_display = image.cpu().numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_display = std * img_display + mean
                img_display = np.clip(img_display, 0, 1)

                ax.imshow(img_display)
                title_color = "green" if is_correct else "red"
                ax.set_title(f"Ист: {class_names[true_label_idx]}\nПред: {class_names[predicted_label_idx.item()]}",
                             color=title_color)
                examples_shown += 1
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
        print(f"Примеры предсказаний сохранены по пути: {save_path}")
    plt.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Загрузка тестового датасета
    print("Загрузка тестового датасета...")
    test_dataset = Food101Dataset(DB_PATH, "test", val_test_transforms, selected_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = test_dataset.idx_to_class
    num_classes = len(class_names)

    # Инициализация модели
    print("Инициализация модели EfficientNet-B0...")
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Сбор предсказаний
    print("Сбор предсказаний...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs is None or labels is None:
                continue
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Конвертация в numpy массивы
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Создание папки для графиков
    plots_dir = os.path.join(os.path.dirname(FOOD_101_ROOT_PATH), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Построение матрицы ошибок
    print("Построение матрицы ошибок...")
    cm = confusion_matrix(all_labels, all_preds)
    # Полная матрица (101x101)
    plot_confusion_matrix(cm, class_names, os.path.join(plots_dir, "confusion_matrix_full.png"))
    # Уменьшенная матрица для топ-20 классов
    plot_confusion_matrix(cm, class_names, os.path.join(plots_dir, "confusion_matrix_top20.png"), top_n_classes=20)

    # 2. Вывод топ-10 ошибок
    print_top_n_errors(cm, class_names, n=10)

    # 3. Отображение примеров правильных и неправильных предсказаний
    print("Отображение примеров предсказаний...")
    plot_predictions(test_dataset, model, device, class_names, num_examples=10, correct=True,
                     save_path=os.path.join(plots_dir, "correct_predictions.png"))
    plot_predictions(test_dataset, model, device, class_names, num_examples=10, correct=False,
                     save_path=os.path.join(plots_dir, "incorrect_predictions.png"))

    print("Анализ производительности завершен. Проверьте папку 'plots' для графиков.")