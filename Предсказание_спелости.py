import numpy as np
import cv2
import tensorflow.lite as tflite

# Путь к файлу модели TensorFlow Lite
model_path = r'F:\PyCharm\OrangePi\model.tflite'  # Укажите путь к модели
image_path = r'/tomato\photo_2024-10-09_15-32-33.jpg'  # Укажите путь к изображению

# Функция для предобработки изображения
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  # Изменение размера изображения
    img = img.astype('float32') / 255.0  # Нормализация
    img = np.expand_dims(img, axis=0)  # Добавление измерения пакета
    return img

# Загрузка модели TensorFlow Lite
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Получение информации о входных и выходных тензорах
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Обработка изображения
img = preprocess_image(image_path)

# Установка входных данных в модель
interpreter.set_tensor(input_details[0]['index'], img)

# Запуск инференса
interpreter.invoke()

# Получение результатов
output_data = interpreter.get_tensor(output_details[0]['index'])
prediction = output_data[0][0]
predicted_label = 'Спелый' if prediction >= 0.5 else 'Неспелый'

# Вывод результата
print(f'Результат предсказания: {predicted_label} (значение: {prediction:.4f})')
