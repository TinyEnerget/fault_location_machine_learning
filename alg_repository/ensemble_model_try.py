# Импорт необходимых библиотек
import matplotlib.pyplot as plt
import seaborn as sns
from aeon.datasets import load_basic_motions
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.compose import WeightedEnsembleClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Загрузка набора данных
X_train, y_train = load_basic_motions(split="train", return_X_y=True)
X_test, y_test = load_basic_motions(split="test", return_X_y=True)

# Определение отдельных классификаторов
tsf = TimeSeriesForestClassifier(n_estimators=100)
rocket = RocketClassifier(num_kernels=10000)
knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")

# Создание ансамблевого классификатора
ensemble_clf = WeightedEnsembleClassifier(classifiers=[tsf, rocket, knn], weights=[1, 1, 1])

# Обучение ансамблевого классификатора
ensemble_clf.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = ensemble_clf.predict(X_test)
print(type(y_test[0]))
print(type(y_pred[0]))
y_pred = [str(label) for label in y_pred]

all_labels = np.unique(np.concatenate((y_test, y_pred)))
label_mapping = {label: i for i, label in enumerate(all_labels)}

y_test_encoded = np.array([label_mapping[label] for label in y_test])
y_pred_encoded = np.array([label_mapping[label] for label in y_pred])

# Use y_test_encoded and y_pred_encoded for evaluation
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Точность ансамблевого классификатора: {accuracy:.2f}")

print("\nОтчет о классификации:")
print(classification_report(y_test_encoded, y_pred_encoded))

cm = confusion_matrix(y_test_encoded, y_pred_encoded)

# Построение матрицы ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=ensemble_clf.classes_,
    yticklabels=ensemble_clf.classes_
)
plt.title("Матрица ошибок")
plt.ylabel("Истинные метки")
plt.xlabel("Предсказанные метки")
plt.show()