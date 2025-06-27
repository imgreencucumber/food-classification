from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Конфигурация (в .env)
FOOD_101_ROOT_PATH = os.getenv("FOOD_101_ROOT_PATH")

# Путь к файлу базы данных SQLite
DB_PATH = os.path.join(FOOD_101_ROOT_PATH, "food_101_metadata.db")

# Выбранные классы для тестового обучения
TEST_CLASSES = ["pizza", "sushi", "donuts"]

# Параметры обучения
BATCH_SIZE = 256
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# Размер изображения для модели
IMAGE_SIZE = 224

# Кастомный Dataset для Food-101
class Food101Dataset(Dataset):
    def __init__(self, db_path, dataset_type, transform=None, selected_classes=None):
        self.db_path = db_path
        self.dataset_type = dataset_type
        self.transform = transform
        self.data = []
        self.class_to_idx = {}
        self.idx_to_class = []

        self._load_data(selected_classes)

    def _load_data(self, selected_classes):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = f"SELECT image_path, label FROM food_101_metadata WHERE dataset_type = \'{self.dataset_type}\'"
        if selected_classes:
            class_list_str = ", ".join([f"\'{c}\'" for c in selected_classes])
            query += f" AND label IN ({class_list_str})"

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        # Собираем уникальные классы и создаем маппинг
        unique_labels = sorted(list(set([row[1] for row in rows])))
        self.idx_to_class = unique_labels
        self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}

        # Формируем список данных (путь к изображению, индекс класса)
        for img_path, label in rows:
            if label in self.class_to_idx:
                self.data.append((img_path, self.class_to_idx[label]))

        print(f"Загружено {len(self.data)} записей для {self.dataset_type} датасета.")
        print(f"Количество классов: {len(self.idx_to_class)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_idx = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_idx

# Трансформации данных
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def plot_metrics(train_losses, val_losses, train_accs, val_accs, num_epochs, save_dir):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    plots_save_path = os.path.join(save_dir, "test_MobileNetV3-Small_5epochs.png")
    plt.savefig(plots_save_path)
    plt.close()
    print(f"Графики метрик сохранены по пути: {plots_save_path}")

if __name__ == '__main__':
    # Инициализация датасетов и DataLoader'ов
    print("Инициализация датасетов...")
    train_dataset = Food101Dataset(DB_PATH, "train", train_transforms, selected_classes=TEST_CLASSES)
    val_dataset = Food101Dataset(DB_PATH, "test", val_test_transforms, selected_classes=TEST_CLASSES)

    # Проверка, что классы совпадают
    if train_dataset.idx_to_class != val_dataset.idx_to_class:
        print("ВНИМАНИЕ: Классы в обучающей и валидационной выборках не совпадают")
        print("Обучающие классы:", train_dataset.idx_to_class)
        print("Валидационные классы:", val_dataset.idx_to_class)

    num_classes = len(train_dataset.idx_to_class)
    print(f"Количество классов для обучения: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Инициализация модели MobileNetV3-Small
    print("Инициализация модели MobileNetV3-Small...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Замораживаем все параметры, кроме последнего слоя
    for param in model.parameters():
        param.requires_grad = False

    # Заменяем последний слой классификатора
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    model = model.to(device)

    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[3].parameters(), lr=LEARNING_RATE)

    # Обучение модели
    print("Начало обучения...")
    start_time = time.time()

    # Списки для хранения метрик
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_train / total_train
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f}")
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Валидация
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_loader_tqdm.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct_val / total_val
        print(f"Epoch {epoch+1} Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.4f}")
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

    end_time = time.time()
    print(f"Обучение завершено за {(end_time - start_time):.2f} секунд.")

    # Сохранение модели
    PROJECT_ROOT = os.getenv('PROJECT_ROOT')
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "test_MobileNetV3-Small_5epochs.pth")
    if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH))
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Модель сохранена по пути: {MODEL_SAVE_PATH}")

    # Визуализация метрик
    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plot_metrics(train_losses, val_losses, train_accs, val_accs, NUM_EPOCHS, plots_dir)

    # Вывод метрик для анализа
    print("\n--- Метрики для анализа ---")
    print(f"train_losses = {train_losses}")
    print(f"train_accs = {train_accs}")
    print(f"val_losses = {val_losses}")
    print(f"val_accs = {val_accs}")