# Демонстрационен ML проект

Тази папка вече съдържа **основния academic demo**, който е най-подходящ за предаване към курса: реален **tabular ML** проект върху външен публичен dataset от транспортната област. Това прави submission-а много по-силен, защото dataset-ът, target-ът и метриките са независими от вътрешната логика на приложението.

## Препоръчителен вариант за предаване

При предаване към курса препоръчителният фокус е:

- [uci_traffic_volume_demo.ipynb](./uci_traffic_volume_demo.ipynb)
- [train_uci_traffic_models.py](./train_uci_traffic_models.py)
- [uci_traffic_demo_results.json](./uci_traffic_demo_results.json)
- [data/metro_interstate_traffic_volume.csv.gz](./data/metro_interstate_traffic_volume.csv.gz)

## Dataset

Основният dataset е **Metro Interstate Traffic Volume** от **UCI Machine Learning Repository**:

- Source: <https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume>
- DOI: `10.24432/C5X60B`
- License: `CC BY 4.0`
- Citation: `Hogue, J. (2019). Metro Interstate Traffic Volume [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5X60B`

Локално dataset-ът е запазен като:

- [metro_interstate_traffic_volume.csv.gz](./data/metro_interstate_traffic_volume.csv.gz)

Той съдържа **48,204** почасови наблюдения за трафика по westbound I-94 в района на Minneapolis-St Paul за периода **2012-2018**. Използваните входни признаци включват:

- метеорологични измервания: `temp`, `rain_1h`, `snow_1h`, `clouds_all`
- времеви характеристики: дата, час, ден от седмицата, месец, weekend indicator
- категориални признаци: `holiday`, `weather_main`, `weather_description`

Това е напълно валидна **tabular data** задача в духа на задачите от FastAI курса.

## Модели

Върху този dataset са обучени **два модела**, и двата с коректен **chronological split 70/15/15**, за да няма temporal leakage:

### 1. Ridge Regressor

Моделът предсказва:

- `traffic_volume`

Използван е:

- `Ridge(alpha=2.0)`

Базов модел за сравнение:

- `DummyRegressor(strategy="median")`

### 2. Traffic Band Classifier

Втората задача е класификация на трафика в три класа:

- `low`
- `medium`
- `high`

Класовете се формират по train quantiles върху `traffic_volume`, с прагове:

- `low <= 2157`
- `medium <= 4555`
- `high > 4555`

Използван е:

- `LogisticRegression(max_iter=2500)`

Базов модел за сравнение:

- `DummyClassifier(strategy="most_frequent")`

## Резултат

Основните test метрики са:

### Regression

- Ridge MAE: **825.19**
- Ridge RMSE: **1056.85**
- Ridge R²: **0.7162**

Сравнение с baseline:

- Dummy MAE: **1737.65**
- Dummy RMSE: **1984.41**
- Dummy R²: **-0.0007**

### Classification

- Logistic Regression Accuracy: **0.7960**
- Precision macro: **0.7954**
- Recall macro: **0.7950**
- F1 macro: **0.7951**

Сравнение с baseline:

- Dummy Accuracy: **0.3250**
- Dummy F1 macro: **0.1635**

Това показва ясно, че и двата train-нати модела учат смислени зависимости върху реален транспортен dataset и превъзхождат базовите решения с голяма разлика.

## Кратък submission summary

Ако трябва да предадеш съвсем кратък текст по условие, можеш да използваш следното:

- Dataset: `Metro Interstate Traffic Volume` от UCI, 48,204 почасови наблюдения за трафик, време и празници
- Модели: `Ridge` за regression на `traffic_volume` и `LogisticRegression` за класификация на traffic bands
- Резултат: `R² = 0.7162` за regression и `Accuracy = 0.7960`, `F1 macro = 0.7951` за classification върху test split

## Стартиране

Ако искаш да пресъздадеш резултатите локално:

```bash
python3 ml_demo/train_uci_traffic_models.py
```

Ако искаш отново да изтеглиш dataset-а от оригиналния източник:

```bash
python3 ml_demo/download_uci_traffic_dataset.py
```

## Допълнителни project-specific модели

В папката остават и по-старите модели върху [sofia_route_network.csv](../sofia_route_network.csv), защото те са полезни за самото Streamlit приложение:

- [train_green_corridor_demo.py](./train_green_corridor_demo.py)
- [train_travel_time_random_forest.py](./train_travel_time_random_forest.py)
- [train_congestion_random_forest.py](./train_congestion_random_forest.py)

Те са добри като **допълнителна техническа част** към темата за IoT в интелигентните транспортни системи, но за академичното предаване вече е по-силно да се акцентира върху UCI dataset-а, защото той е външен, реален и по-лесен за защита.
