# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
def load_data():
    import pandas as pd

    train = pd.read_csv("./files/input/train_data.csv.zip", compression="zip")
    test = pd.read_csv("./files/input/test_data.csv.zip", compression="zip" )

    return train, test

def clean_data(data):
    data = data.copy()
    data.rename(columns = {'default payment next month':'default'}, inplace = True)
    data.dropna(inplace=True)

    data["EDUCATION"] = data["EDUCATION"].apply(lambda x: 5 if x not in [1,2,3,4] else x)
    data.drop(columns=["ID"], inplace=True)

    return data

train_data, test_data = load_data()

train_data = clean_data(train_data)
test_data = clean_data(test_data)

x_train = train_data.drop(columns=["default"])
y_train = train_data["default"]
x_test = test_data.drop(columns=["default"])
y_test = test_data["default"]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

def build_pipeline():
    categorical_features = [col for col in x_train.columns if x_train[col].dtype == "object"]
    numeric_features = [col for col in x_train.columns if x_train[col].dtype != "object"]
    
    onehot = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    pipeline = Pipeline(steps=[
        ("OneHotEncoder", onehot),
        ("PCA", PCA(n_components=None)),
        ("MinMaxScaler", MinMaxScaler()),
        ("SelectKBest", SelectKBest(score_func=f_classif, k="all")),
        ("MLPClassifier", MLPClassifier(max_iter=1000))
    ])

    return pipeline

def optimize_params(pipeline):
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        "SelectKBest__k": ["all"],
        # "SelectKBest__score_func": [f_classif],
        # "MLPClassifier__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        # "MLPClassifier__solver": ["adam", "sgd"],
        # "MLPClassifier__alpha": [0.0001, 0.001],
        # "MLPClassifier__learning_rate": ["constant", "adaptive"]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
    )

    grid_search.fit(x_train, y_train)
    return grid_search

pipeline = build_pipeline()
model = optimize_params(pipeline)

def save_model(estimator):
    import gzip
    import pickle
    import os

    os.makedirs("./files/models", exist_ok=True)
    with gzip.open("./files/models/model.pkl.gz", "wb") as f:
        pickle.dump(estimator, f)
save_model(model)

def calc_metrics(estimator):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix
    import os
    import json

    os.makedirs("./files/output", exist_ok=True)
    with open("./files/output/metrics.json", "w") as f:
        metrics_train = {
            "dataset": "train",
            "accuracy": accuracy_score(y_train, estimator.predict(x_train)),
            "balanced_accuracy": balanced_accuracy_score(y_train, estimator.predict(x_train)),
            "recall": recall_score(y_train, estimator.predict(x_train)),
            "f1_score": f1_score(y_train, estimator.predict(x_train))
        }
        f.write(json.dumps(metrics_train) + "\n")

        metrics_test = {
            "dataset": "test",
            "accuracy": accuracy_score(y_test, estimator.predict(x_test)),
            "balanced_accuracy": balanced_accuracy_score(y_test, estimator.predict(x_test)),
            "recall": recall_score(y_test, estimator.predict(x_test)),
            "f1_score": f1_score(y_test, estimator.predict(x_test))
        }
        f.write(json.dumps(metrics_test) + "\n")

        cm_matrix_train = confusion_matrix(y_train, estimator.predict(x_train))
        cm_dict_train = {
            "type": "cm_matrix",
            "dataset": "train",
            "true_0": {"predicted_0": int(cm_matrix_train[0,0]), "predicted_1": int(cm_matrix_train[0,1])},
            "true_1": {"predicted_0": int(cm_matrix_train[1,0]), "predicted_1": int(cm_matrix_train[1,1])}
        }
        f.write(json.dumps(cm_dict_train) + "\n")

        cm_matrix_test = confusion_matrix(y_test, estimator.predict(x_test))
        cm_dict_test = {
            "type": "cm_matrix",
            "dataset": "test",
            "true_0": {"predicted_0": int(cm_matrix_test[0,0]), "predicted_1": int(cm_matrix_test[0,1])},
            "true_1": {"predicted_0": int(cm_matrix_test[1,0]), "predicted_1": int(cm_matrix_test[1,1])}
        }
        f.write(json.dumps(cm_dict_test) + "\n")

calc_metrics(model)