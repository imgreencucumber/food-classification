import sqlite3
import os

# Конфигурация
FOOD_101_ROOT_PATH = os.getenv("FOOD_101_ROOT_PATH")
DB_PATH = os.path.join(FOOD_101_ROOT_PATH, "food_101_metadata.db")

NUM_RANDOM_RECORDS = 5 # Количество случайных записей, которые вы хотите получить

def get_random_records(db_path, num_records):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # SQL-запрос для получения случайных записей
        # ORDER BY RANDOM() сортирует записи случайным образом
        # LIMIT ограничивает количество возвращаемых записей
        query = f"SELECT image_path, label FROM food_101_metadata ORDER BY RANDOM() LIMIT {num_records}"
        
        cursor.execute(query)
        random_records = cursor.fetchall()
        
        return random_records

    except sqlite3.Error as e:
        print(f"Ошибка SQLite: {e}")
        return []
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"Ошибка: База данных не найдена по пути: {DB_PATH}")
    else:
        print(f"Получение {NUM_RANDOM_RECORDS} случайных записей из {DB_PATH}...")
        records = get_random_records(DB_PATH, NUM_RANDOM_RECORDS)
        
        if records:
            print("Случайные записи:")
            for i, record in enumerate(records):
                print(f"{i+1}. Path: {record[0]}, Label: {record[1]}")
        else:
            print("Не удалось получить случайные записи.")