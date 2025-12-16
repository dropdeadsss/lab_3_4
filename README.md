# Лабораторная работа 3-4
## Знакомство с платформой Hugging Face Hub. Тонкая настройка модели для текстовой классификации. Интеграция с MLflow для трекинга экспериментов

### Цели
1) Освоить базовые принципы работы с платформой Hugging Face Hub - центральным репозиторием моделей, датасетов и приложений машинного обучения. Получить практические навыки поиска, оценки и загрузки моделей и датасетов для задачи текстовой классификации.
2) Освоить практические навыки тонкой настройки (fine-tuning) предобученных моделей для задачи текстовой классификации с использованием библиотеки Transformers. Получить опыт подготовки данных, настройки обучения и оценки качества модели.
3) Освоить интеграцию процесса тонкой настройки моделей с платформой MLflow для комплексного трекинга экспериментов. Научиться автоматически логировать гиперпараметры, метрики, артефакты и модели в ходе обучения.

### Выполнение
#### Часть 1

Для поиска моделей и датасетов можно использовать сайт [Hugging Face](https://huggingface.co/). С помощью фильтров можно легко найти то, что нужно. 

Установив пакеты *pip install huggingface_hub datasets transformers* мы можем загружать модели и датасеты в коде, а также обрабатывать их.

Запустим скрипт [hf_hub_exploration.py](https://github.com/dropdeadsss/lab_3_4/blob/main/hf_hub_exploration.py). Он загружает датасет "emotion", который содержит комментарии и их настроение. Также он выводит примеры из датасета, проверяет распределение классов (настроение) в тренировочных данных

![screenshot](https://github.com/dropdeadsss/lab_3_4/blob/main/imgs/1.JPG)

![screenshot](https://github.com/dropdeadsss/lab_3_4/blob/main/imgs/2.JPG)

Далее скрипт загружает модель distilbert-base-uncased

![screenshot](https://github.com/dropdeadsss/lab_3_4/blob/main/imgs/3.JPG)

Модифицируем скрипт и протестируем работу модели на случайных примеров из выборки

Для подписи "эмоций" в тексте добавим {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

![screenshot](https://github.com/dropdeadsss/lab_3_4/blob/main/imgs/4.JPG)

Мы видим, что модель не очень хорошо опреляет настроение текста 0/4. Возможно, следует увеличить объем и качество обучающей выборки или использовать другую модель.

#### Часть 2

Создадим скрипт [fine_tuning.py](https://github.com/dropdeadsss/lab_3_4/blob/main/fine_tuning.py), в котором мы загрузим модель и обучим ее на тестовых данных, задав нужные гиперпараметры. После обучения модель сохранится в папку [emotion-classifier](https://github.com/dropdeadsss/lab_3_4/blob/main/emotion-classifier), а результаты обучения в [result](https://github.com/dropdeadsss/lab_3_4/blob/main/result)

![screenshot](https://github.com/dropdeadsss/lab_3_4/blob/main/imgs/5.JPG)

![screenshot](https://github.com/dropdeadsss/lab_3_4/blob/main/imgs/6.JPG)

![screenshot](https://github.com/dropdeadsss/lab_3_4/blob/main/imgs/7.JPG)

![screenshot](https://github.com/dropdeadsss/lab_3_4/blob/main/imgs/8.JPG)

![screenshot](https://github.com/dropdeadsss/lab_3_4/blob/main/imgs/9.JPG)

В файле [test_results.txt](https://github.com/dropdeadsss/lab_3_4/blob/main/test_results.txt) сохранены метрики на тестовом наборе 

Accuracy: 0.3475

F1 Score: 0.1792

#### Часть 3
Невозможно выполнить т.к. датасеты и модели с Hugging Face у меня не грузятся из кода (проблемы с загрузкой из-за всяких блокировок). Примеры выше выполнены в google colab, но подключится к localhost mlflow серверу из него невозможно.
