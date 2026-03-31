# Демонстрационен ML проект

Тази папка съдържа отделен малък демонстрационен tabular ML проект, така че репото да покрива академичното изискване за train-нат модел и кратко описание на dataset, model и result. Задачата е от типа **таблични данни**, което съответства на един от стандартните типове задачи, разглеждани в курса на FastAI.

## Dataset

Използван е файлът [sofia_route_network.csv](../sofia_route_network.csv), който съдържа **91,686** пътни сегмента за София. Всеки ред описва отделен road segment с геометрия, клас на пътя, ограничение на скоростта, трафик-индикатори, green-wave характеристики и допълнителни профилни масиви по време на деня.

За демонстрационната задача таргетът е:

- `green_corridor`

Това е **binary classification** задача: моделът предсказва дали даден пътен сегмент принадлежи към синхронизиран green corridor.

## Модели

Добавени са два отделни таблични модела:

### 1. Green Corridor Classifier

Обучен е лек **logistic regression** модел, реализиран с **NumPy**, върху табличните характеристики на dataset-а:

- числови признаци: координати, дължина, speed limit, congestion, delay, eco factor и др.
- one-hot encoded категориални признаци: `road_class` и `direction`
- профилни масиви: `congestion_profile_3h`, `green_profile_3h`, `curb_activity_profile_3h`, `weekday_volume_profile`

Скриптът за обучение е:

- [train_green_corridor_demo.py](./train_green_corridor_demo.py)

Той записва:

- train-натите параметри в `green_corridor_logreg_model.npz`
- метрики и summary в `green_corridor_demo_results.json`

### 2. Travel Time Regressor

Добавен е и **RandomForestRegressor** модел върху същия dataset, който предсказва:

- `travel_time_min` за road segment в **multi-hour** сценарий (`hour=0..23`)

Скриптът за обучение е:

- [train_travel_time_random_forest.py](./train_travel_time_random_forest.py)

Той записва:

- train-натия модел в `travel_time_random_forest.joblib`
- regression metrics в `travel_time_random_forest_results.json`

### 3. Congestion Classifier

Добавен е и **RandomForestClassifier** модел върху същия dataset, който предсказва:

- `congestion band` при **multi-hour** сценарий (`hour=0..23`)

Таргетът е формулиран като 3-класова задача:

- `low`
- `medium`
- `high`

Класовете са получени от динамичния congestion score, дискретизиран по tercile thresholds върху извадка от dataset-а.

Скриптът за обучение е:

- [train_congestion_random_forest.py](./train_congestion_random_forest.py)

Той записва:

- train-натия модел в `congestion_random_forest.joblib`
- classification metrics в `congestion_random_forest_results.json`

## Резултат

След обучение върху train/test split 80/20, проектът постига много силен резултат върху test частта на данните. Конкретните метрики са записани в:

- [green_corridor_demo_results.json](./green_corridor_demo_results.json)

Актуалният резултат за classifier-а е:

- Accuracy: **0.9993**
- Precision: **1.0000**
- Recall: **0.9662**
- F1 score: **0.9828**

Конфузионната матрица в count формат е:

- TP: **343**
- TN: **17,983**
- FP: **0**
- FN: **12**

Актуалният резултат за Random Forest regresssor-а е:

- MAE: **0.0041 min**
- RMSE: **0.0139 min**
- R²: **0.9797**

Актуалният резултат за Random Forest congestion classifier-а е:

- Accuracy: **0.8433**
- Precision macro: **0.8459**
- Recall macro: **0.8433**
- F1 macro: **0.8440**

Най-важното за предаване е:

- Dataset: `sofia_route_network.csv`
- Модели:
  - NumPy logistic regression за `green_corridor` classification
  - RandomForestRegressor за `travel_time_min` regression
  - RandomForestClassifier за `congestion band` classification
- Резултат:
  - accuracy / precision / recall / F1 за classifier-а
  - MAE / RMSE / R² за regressor-а
  - accuracy / macro precision / macro recall / macro F1 за congestion classifier-а

## Стартиране

```bash
python3 ml_demo/train_green_corridor_demo.py
python3 ml_demo/train_travel_time_random_forest.py
python3 ml_demo/train_congestion_random_forest.py
```

Този demo project е отделен от основното Streamlit приложение и съществува специално, за да покрие изискването за малък ML/FastAI-style проект с train-нат модел и описан резултат.
Същите train-нати артефакти вече се използват и директно в основното приложение, където добавят ML ETA, predicted traffic и corridor confidence върху локалния Sofia routing flow.
