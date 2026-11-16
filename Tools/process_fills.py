import cv2
import numpy as np
import json
import os

def process_colored_frames(frames_dir, json_path, output_json_path):
    # Завантажуємо існуючий JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    # Глобальний реєстр кольорів для присвоєння унікальних ID
    # Ключ: RGB tuple (R, G, B), Значення: ID
    color_to_id = {}
    next_part_id = 1

    print(f"Починаємо обробку {len(meta_data['frames'])} кадрів...")

    for frame in meta_data['frames']:
        frame_idx = frame['index']
        # Формуємо ім'я файлу, наприклад colored_0000.png
        img_name = f"colored_{frame_idx:04d}.png"
        img_path = os.path.join(frames_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Попередження: Файл {img_name} не знайдено. Пропускаємо.")
            continue

        # Завантажуємо зображення та конвертуємо в RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Знаходимо всі унікальні кольори на кадрі
        unique_colors = np.unique(img.reshape(-1, 3), axis=0)

        fill_parts = []

        for color in unique_colors:
            # Ігноруємо чорний (контури, < 10) та білий (фон, > 245)
            if np.all(color < 10) or np.all(color > 245):
                continue

            color_tuple = tuple(int(c) for c in color)
            
            # Якщо бачимо колір вперше — даємо йому новий ID
            if color_tuple not in color_to_id:
                color_to_id[color_tuple] = next_part_id
                next_part_id += 1
            
            part_id = color_to_id[color_tuple]

            # Створюємо булеву маску для конкретного кольору
            mask = np.all(img == color, axis=-1)

            # Знаходимо індекси рядків (Y), де є хоча б один піксель цього кольору
            y_indices = np.any(mask, axis=1).nonzero()[0]
            
            segments_by_row = {}

            # Проходимось по кожному рядку і шукаємо проміжки [start, end]
            for y in y_indices:
                row_mask = mask[y]
                
                # Шукаємо переходи з False в True (початок) і з True в False (кінець)
                padded = np.pad(row_mask, (1, 1), 'constant', constant_values=False)
                diff = np.diff(padded.astype(int))
                
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0] - 1
                
                # Записуємо сегменти у форматі: [ [start1, end1], [start2, end2] ]
                row_segments = [[int(s), int(e)] for s, e in zip(starts, ends)]
                
                # Зберігаємо під ключем рядка (рядок має бути String для JSON ключів)
                segments_by_row[str(y)] = row_segments

            # Додаємо дані про цю кольорову частину
            fill_parts.append({
                "fill_id": part_id,
                "color_rgb": color_tuple,
                "segments": segments_by_row
            })

        # Додаємо нове поле в поточний фрейм JSON
        frame['fill_parts'] = fill_parts
        print(f"Кадр {frame_idx:04d} оброблено. Знайдено {len(fill_parts)} кольорових частин.")

    # Зберігаємо оновлений JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, separators=(',', ':')) # Компактний запис для зменшення розміру

    print(f"Готово! Оновлений JSON збережено у {output_json_path}")
    print(f"Всього знайдено унікальних частин (кольорів): {len(color_to_id)}")

# --- ЗАПУСК ---
if __name__ == "__main__":
    # Вкажіть тут ваші шляхи:
    DIR_WITH_FRAMES = "./frames" 
    INPUT_JSON = "coloring_0101_meta.json"
    OUTPUT_JSON = "coloring_0101_meta.json"

    process_colored_frames(DIR_WITH_FRAMES, INPUT_JSON, OUTPUT_JSON)