from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

# Конфигурация (в .env)
FOOD_101_ROOT_PATH = os.getenv("FOOD_101_ROOT_PATH")

# Путь к папке с изображениями
IMAGES_PATH = os.path.join(FOOD_101_ROOT_PATH, "images")

# Путь к папке с метаданными (train.txt, test.txt)
META_PATH = os.path.join(FOOD_101_ROOT_PATH, "meta")

# Путь к файлу базы данных SQLite
DB_PATH = os.path.join(FOOD_101_ROOT_PATH, "food_101_metadata.db")

# Явная настройка PySpark
python_exe_path = os.getenv("PYSPARK_PYTHON")
os.environ["PYSPARK_PYTHON"] = python_exe_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe_path

# Инициализация Spark Session
print("Инициализация Spark Session...")
spark = SparkSession.builder \
    .appName("Food101MetadataPreparation") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
print("Spark Session инициализирована.")

# Вспомогательная функция для парсинга строк
def parse_line(line: str):
    parts = line.split("/")
    label = parts[0] # Имя папки - это метка класса
    relative_path = line + ".jpg" # Добавляем расширение .jpg
    full_path = os.path.join(IMAGES_PATH, relative_path)
    # Заменяем обратные слеши на прямые для единообразия, т.к. PyTorch может лучше работать с прямыми
    full_path = full_path.replace("\\", "/")
    return full_path, label

# Обработка train.txt и test.txt
def process_metadata_file(file_path: str, dataset_type: str):
    print(f"Обработка {file_path} для {dataset_type} датасета...")
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Создаем RDD из строк и применяем парсинг
    rdd = spark.sparkContext.parallelize(lines)
    parsed_rdd = rdd.map(parse_line)

    # Преобразуем RDD в DataFrame
    df = spark.createDataFrame(parsed_rdd, ["image_path", "label"])
    df = df.withColumn("dataset_type", lit(dataset_type))
    print(f"DataFrame для {dataset_type} датасета создан. Количество записей: {df.count()}")
    return df

# Обрабатываем train.txt
train_df = process_metadata_file(os.path.join(META_PATH, "train.txt"), "train")

# Обрабатываем test.txt
test_df = process_metadata_file(os.path.join(META_PATH, "test.txt"), "test")

# Объединяем оба DataFrame
combined_df = train_df.union(test_df)
print(f"Объединенный DataFrame создан. Общее количество записей: {combined_df.count()}")

# Сохранение в SQLite
print(f"Сохранение метаданных в SQLite базу данных: {DB_PATH}...")

# Удаляем старую базу данных, если она существует
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"Существующая база данных {DB_PATH} удалена.")

# Преобразуем Spark DataFrame в Pandas DataFrame для сохранения в SQLite
# Это делается для того, чтобы использовать встроенный в Python sqlite3, который не работает напрямую со Spark DataFrame
# На больших датасетах это может быть узким местом, но для Food-101 это приемлемо на локальной машине.

# Сохраняем train данные
train_pandas_df = train_df.toPandas()
conn = sqlite3.connect(DB_PATH)
train_pandas_df.to_sql("food_101_metadata", conn, if_exists="append", index=False)
conn.close()
print("Обучающие метаданные сохранены в SQLite.")

# Сохраняем test данные
test_pandas_df = test_df.toPandas()
conn = sqlite3.connect(DB_PATH)
test_pandas_df.to_sql("food_101_metadata", conn, if_exists="append", index=False)
conn.close()
print("Тестовые метаданные сохранены в SQLite.")

print("Метаданные успешно обработаны и сохранены в SQLite.")

# Проверка сохраненных данных
print("-" * 100)
print("Проверка сохраненных данных...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM food_101_metadata")
count = cursor.fetchone()[0]
print(f"Количество записей в таблице food_101_metadata: {count}")

cursor.execute("SELECT * FROM food_101_metadata LIMIT 5")
print("Первые 5 записей:")
for row in cursor.fetchall():
    print(row)

conn.close()

spark.stop()
print("Spark Session остановлена.")


