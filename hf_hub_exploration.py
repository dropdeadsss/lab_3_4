import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import list_models, list_datasets
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Исследование доступных датасетов
print("Доступные датасеты для текстовой классификации:")
datasets = list_datasets(filter="task_categories:text-classification")
#for dataset in datasets:
  #print(f"- {dataset.id}")

# Загрузка датасета emotion
print("\nЗагрузка датасета emotion...")
dataset = load_dataset("emotion")
print(f"\nСтруктура датасета: {dataset}")

# Пример данных
print(f"\nПримеры из train split:")
train_df = pd.DataFrame(dataset['train'][:5])
print(train_df)

# Анализ распределения классов
print("\nРаспределение классов в тренировочных данных:")
label_counts = pd.Series(dataset['train']['label']).value_counts()
print(label_counts)

# Анализ длины текстов
print("\nАнализ длины текстов в тренировочном датасете:")
train_texts = dataset['train']['text']
lengths = [len(text.split()) for text in train_texts]

print(f"Средняя длина: {np.mean(lengths):.2f} слов")
print(f"Медиана: {np.median(lengths)} слов")
print(f"Мин: {np.min(lengths)} слов")
print(f"Макс: {np.max(lengths)} слов")

# Визуализация распределения длин текстов
plt.figure(figsize=(8, 5))
plt.hist(lengths, bins=20, color='skyblue')
plt.title("Распределение длины текстов (в словах)")
plt.xlabel("Длина текста (слов)")
plt.ylabel("Количество")
plt.grid(axis='y')
plt.show()

# Загрузка модели и токенизатора
model_name = "distilbert-base-uncased"
print(f"\nЗагрузка модели {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
  model_name,
  num_labels=6 # В вашем случае, вероятно, 6 классов для 'emotion'
)

# Инициализация пайплайна для классификации
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

print("Модель и токенизатор успешно загружены!")
print(f"Размер словаря: {tokenizer.vocab_size}")
print(f"Архитектура модели: {model.__class__.__name__}")

# Тестирование токенизатора
test_text = "I am feeling very happy today!"
tokens = tokenizer(test_text, return_tensors="pt")
print(f"\nТекст для теста: {test_text}")
print(f"Токены: {tokens}")
print(f"Декодированные токены: {tokenizer.decode(tokens['input_ids'][0])}")

id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

# Функция для тестирования модели на случайных примерах
def test_model_on_samples(dataset, num_samples=5):
  import random
  samples = random.sample(range(len(dataset)), num_samples)
  for idx in samples:
    text = dataset[idx]['text']
    true_label = dataset[idx]['label']
    result = classifier(text)[0]
    predicted_labels = sorted(result, key=lambda x: x['score'], reverse=True)
    top_pred = predicted_labels[0]
    label_name = id2label[int(top_pred['label'].split('_')[-1])]
    print(f"\nПример {idx}:")
    print(f"Текст: {text}")
    print(f"Истинный класс: {dataset[idx]['label']}")  # Можно заменить на название класса, если есть
    print("Предсказания:")
    for pred in predicted_labels[:3]:
      label_name_pred = id2label[int(pred['label'].split('_')[-1])]
      print(f"  {label_name_pred}: {pred['score']:.3f}")
    print(f"Текущий лучший предсказатель: {label_name} (точность: {top_pred['score']:.3f})")

# Запуск тестирования на случайных примерах
print("\nТест модели на случайных примерах из датасета:")
test_model_on_samples(dataset['test'], num_samples=5)