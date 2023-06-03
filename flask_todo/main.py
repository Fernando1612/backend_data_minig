from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
from eda import EDA  # Importa la clase EDA desde tu archivo de clase EDA (eda.py)
from pca_p import PCA_P
from bosques import Bosques
from bosque_regresor import BosqueRegresor
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import json
import pickle
import csv
import numpy as np


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite')


app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

bosques = Bosques('data_dir/data.csv')
bosque_regresor = BosqueRegresor('data_dir/data.csv')

modelo_rf = None

# Function to check for data directory
def check_directory():
    directory = 'data_dir'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

@app.route('/upload', methods=['POST'])
def upload():
    archivo = request.files['archivo']
    directory = check_directory() # if not exists, dir created for .CSV file
    archivo.save(os.path.join(directory,'data.csv'))  # Cambia 'ruta_de_guardado' a la ruta donde deseas guardar el archivo
    return 'Archivo guardado correctamente'

@app.route('/data-preview', methods=['GET'])
def data_preview():
    num_rows = request.args.get('num_rows', default=5, type=int)
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA
    # Cargar los datos
    exploration.load_data()
    # Obtener los datos como un DataFrame
    preview = exploration.preview_data(num_rows)
    # Obtener los nombres de las columnas del DataFrame en el orden original
    column_names = preview.columns.tolist()
    # Convertir los datos en una lista de diccionarios manteniendo el orden de las columnas
    preview_data_list = preview[column_names].to_dict(orient='records')
    # Preparar la respuesta en formato JSON
    response = {
        'column_names': column_names,
        'data': preview_data_list
    }
    # Enviar la respuesta al front-end
    return jsonify(response)

@app.route('/data-statistics', methods=['GET'])
def data_stats():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA
    # Cargar los datos
    exploration.load_data()
    # Obtener los datos como un DataFrame(
    stats = exploration.summary_statistics()
    # Obtener los nombres de las columnas del DataFrame en el orden original
    column_names = stats.columns.tolist()
    # Obtener los nombres de los indices del DataFrame en el orden original
    indx_names = stats.index.tolist()
    # Convertir los datos en una lista de diccionarios manteniendo el orden de las columnas
    preview_data_list = stats[column_names].to_dict(orient='records')
    # Preparar la respuesta en formato JSON
    response = {
        'column_names': column_names,
        'index_names' : indx_names,
        'data': preview_data_list
    }
    # Enviar la respuesta al front-end
    return jsonify(response)

@app.route('/data-nulls', methods=['GET'])
def data_nulls():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA
    # Cargar los datos
    exploration.load_data()
    # Obtener los datos como un DataFrame
    preview = exploration.missing_values()
    # Obtener los nombres de las columnas del DataFrame en el orden original
    column_names = preview.columns.tolist()
    # Obtener los nombres de los indices del DataFrame en el orden original
    indx_names = preview.index.tolist()
    # Convertir los datos en una lista de diccionarios manteniendo el orden de las columnas
    preview_data_list = preview[column_names].to_dict(orient='records')
    # Preparar la respuesta en formato JSON
    response = {
        'column_names': column_names,
        'index_names' : indx_names,
        'data': preview_data_list
    }
    # Enviar la respuesta al front-end
    return jsonify(response)

@app.route('/data-all', methods=['GET'])
def data_all():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA
    # Cargar los datos
    exploration.load_data()
    preview = exploration.get_all_data()
    # Obtener los nombres de las columnas del DataFrame en el orden original
    column_names = preview.columns.tolist()
    # Convertir los datos en una lista de diccionarios manteniendo el orden de las columnas
    preview_data_list = preview[column_names].to_dict(orient='records')
    # Preparar la respuesta en formato JSON
    response = {
        'column_names': column_names,
        'data': preview_data_list
    }
    # Enviar la respuesta al front-end
    return jsonify(response)

@app.route('/plot-histogram', methods=['GET'])
def plot_histogram():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA
    # Cargar los datos
    exploration.load_data()
    file_name = exploration.plot_outliers_histogram()
    return send_file(file_name, mimetype='image/png')

@app.route('/plot-boxplot', methods=['GET'])
def plot_boxplot():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA
    # Cargar los datos
    exploration.load_data()
    file_name = exploration.plot_outliers_boxplot()
    return send_file(file_name, mimetype='image/png')

@app.route('/plot-heatmap', methods=['GET'])
def plot_heatmap():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA
    # Cargar los datos
    exploration.load_data()
    file_name = exploration.plot_correlation_heatmap()
    return send_file(file_name, mimetype='image/png')

@app.route('/pca', methods=['GET'])
def perform_pca():
    num_rows = request.args.get('n_components', default=2, type=int)
    scaler_type = request.args.get('scaler_type', default='StandardScaler', type=str)
    # Crear una instancia de la clase PCA_P con el número de componentes principales deseado
    pca = PCA_P(n_components=num_rows)
    # Cargar los datos
    pca.load_data('data_dir/data.csv')
    # Ajustar el modelo PCA a los datos utilizando el tipo de estandarización especificado
    pca.fit(scaler_type=scaler_type)
    # Aplicar la transformación PCA y obtener los datos transformados
    transformed_data = pca.transform()
    # Obtener los nombres de las columnas del DataFrame transformado
    column_names = transformed_data.columns.tolist()
    # Convertir los datos transformados en una lista de diccionarios
    transformed_data_list = transformed_data.to_dict(orient='records')
    # Obtener el nombre de archivo de la imagen y el valor de la varianza acumulada
    file_name, varianza = pca.plot_variance(n_components=num_rows)
    # Preparar la respuesta en formato JSON
    response = {
        'column_names': column_names,
        'data': transformed_data_list,
        'varianza': varianza
    }
    # Enviar el archivo al front-end
    return jsonify(response)

@app.route('/pca-plot', methods=['GET'])
def plot_pca():
    num_rows = request.args.get('n_components', default=2, type=int)
    scaler_type = request.args.get('scaler_type', default='StandardScaler', type=str)
    # Crear una instancia de la clase PCA_P con el número de componentes principales deseado
    pca = PCA_P(n_components=num_rows)
    # Cargar los datos
    pca.load_data('data_dir/data.csv')
    # Ajustar el modelo PCA a los datos utilizando el tipo de estandarización especificado
    pca.fit(scaler_type=scaler_type)
    # Aplicar la transformación PCA y obtener los datos transformados
    transformed_data = pca.transform()
    # Obtener los nombres de las columnas del DataFrame transformado
    column_names = transformed_data.columns.tolist()
    # Convertir los datos transformados en una lista de diccionarios
    transformed_data_list = transformed_data.to_dict(orient='records')
    # Obtener el nombre de archivo de la imagen y el valor de la varianza acumulada
    file_name, varianza = pca.plot_variance(n_components=num_rows)
    # Enviar el archivo al front-end
    return send_file(file_name, mimetype='image/png')


@app.route('/column-names', methods=['GET'])
def get_column_names():
    bosques = Bosques('data_dir/data.csv')
    bosques.load_data()
    column_names = bosques.column_names()
    return jsonify({'column_names': column_names})

@app.route('/forest', methods=['GET'])
def train_model():
    target_column = request.args.get('target_column')
    n_estimators = int(request.args.get('n_estimators', 100))
    criterion = request.args.get('criterion', 'gini')
    max_depth = request.args.get('max_depth', None)
    if max_depth == 'None':
        max_depth = None
    else:
        max_depth = int(max_depth)
    min_samples_split = int(request.args.get('min_samples_split', 2))
    min_samples_leaf = int(request.args.get('min_samples_leaf', 1))
    max_features = request.args.get('max_features', 'auto')

    csv_file = 'data_dir/data.csv'  # Ruta y nombre de tu archivo CSV
    bosques.cargar_datos(csv_file, target_column)
    bosques.dividir_datos()
    bosques.entrenar_modelo(n_estimators=n_estimators, criterion=criterion,
                            max_depth=max_depth, min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf, max_features=max_features)
    accuracy, _ = bosques.evaluar_modelo()

    bosques.mostrar_matriz_confusion()
    # Generar la gráfica de la curva ROC
    bosques.graficar_curva_roc()
    # Leer la imagen de la curva ROC generada
    with open('roc.png', 'rb') as file:
        image_data = file.read()
    # Convertir la imagen a base64 para enviarla como respuesta
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    # Preparar la respuesta en formato JSON
    
    with open('confusion_matrix.png', 'rb') as file:
        image_data_matrix = file.read()
    # Convertir la imagen a base64 para enviarla como respuesta
    encoded_image_matrix = base64.b64encode(image_data_matrix).decode('utf-8')
    # Preparar la respuesta en formato JSON

    response = {
        'accuracy': accuracy,
        'roc_image': encoded_image,
        'matrix_image': encoded_image_matrix
    }
    # Enviar la respuesta al front-end
    return jsonify(response)

@app.route('/forest-predict', methods=['GET'])
def predict_data():
    # Obtener las características de entrada para la predicción
    feature_params = request.args.to_dict()
    # Convertir feature_params en un DataFrame
    df = pd.DataFrame([feature_params])
    # Realizar la predicción utilizando el método predecir de la clase Bosques
    prediction = bosques.predecir(df)
    # Preparar la respuesta en formato JSON
    response = {
        'prediction': prediction.tolist()
    }
    # Enviar la respuesta al front-end
    return json.dumps(response)

@app.route('/forest-regressor-train', methods=['GET'])
def train_model_regressor():
    target_column = request.args.get('target_column')
    n_estimators = int(request.args.get('n_estimators', 100))
    criterion = request.args.get('criterion', 'mse')
    max_depth = request.args.get('max_depth', None)
    if max_depth == 'None':
        max_depth = None
    else:
        max_depth = int(max_depth)
    min_samples_split = int(request.args.get('min_samples_split', 2))
    min_samples_leaf = int(request.args.get('min_samples_leaf', 1))
    max_features = request.args.get('max_features', 'auto')
    csv_file = 'data_dir/data.csv'  # Ruta y nombre de tu archivo CSV
    bosque_regresor.cargar_datos(csv_file, target_column)
    bosque_regresor.dividir_datos()
    bosque_regresor.entrenar_modelo(n_estimators=n_estimators, criterion=criterion,
                                     max_depth=max_depth, min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf, max_features=max_features)
    mse, criterion, feature_importances, mae, rmse, r2 = bosque_regresor.evaluar_modelo()

    bosque_regresor.graficar_pronostico()

    with open('pronostico.png', 'rb') as file:
        image_data = file.read()
    # Convertir la imagen a base64 para enviarla como respuesta
    encoded_image = base64.b64encode(image_data).decode('utf-8')

    response = {
        'mse': mse,
        'criterion': criterion,
        'feature_importances': feature_importances.tolist(),
        'mean_absolute_error': mae,
        'RMSE': rmse,
        'r2_score': r2,
        'pronostico_image': encoded_image,
    }

    return jsonify(response)


@app.route('/forest-regressor-predict', methods=['GET'])
def predict_data_regressor():
    # Obtener las características de entrada para la predicción
    feature_params = request.args.to_dict()
    # Convertir feature_params en un DataFrame
    df = pd.DataFrame([feature_params])
    # Realizar la predicción utilizando el método predecir de la clase BosqueRegresor
    prediction = bosque_regresor.predecir(df)
    # Preparar la respuesta en formato JSON
    response = {
        'prediction': prediction.tolist()
    }
    # Enviar la respuesta al front-end
    return json.dumps(response)


@app.route('/guardar-modelo-regresor', methods=['GET'])
def guardar_modelo_regresor():
    file_path = request.args.get('file_path')
    bosque_regresor.guardar_modelo(file_path)
    return jsonify({'message': 'Modelo guardado exitosamente.'})

@app.route('/cargar-modelo-regresor', methods=['GET'])
def cargar_modelo_regresor():
    global modelo_rf  # Indicar que se usará la variable global
    file_path = request.args.get('file_path')
    with open(file_path, 'rb') as file:
        modelo_rf = pickle.load(file)
    return jsonify({'message': 'Modelo cargado exitosamente.'})

@app.route('/guardar-modelo-clasificador', methods=['GET'])
def guardar_modelo():
    file_path = request.args.get('file_path')
    bosques.guardar_modelo(file_path)
    return jsonify({'message': 'Modelo guardado exitosamente.'})

@app.route('/cargar-modelo-clasificador', methods=['GET'])
def cargar_modelo_clasificador():
    global modelo_rf  # Indicar que se usará la variable global
    file_path = request.args.get('file_path')
    with open(file_path, 'rb') as file:
        modelo_rf = pickle.load(file)
    return jsonify({'message': 'Modelo cargado exitosamente.'})

@app.route('/predict-modelo', methods=['GET'])
def predecir_modelo():
    # Obtener las características de entrada para la predicción
    feature_params = request.args.to_dict()
    # Convertir feature_params en un DataFrame
    df = pd.DataFrame([feature_params])
    # Realizar la predicción utilizando el método predecir de la clase Bosques
    prediction = modelo_rf.predict(df)
    # Preparar la respuesta en formato JSON
    response = {
        'prediction': prediction.tolist()
    }
    # Enviar la respuesta al front-end
    return json.dumps(response)

@app.route('/guardar-column-names', methods=['GET'])
def save_column_names():
    file_path = request.args.get('file_path')
    target = request.args.get('target')
    column_names = bosques.column_names(target)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
    return jsonify({'message': 'Columnas guardadas exitosamente.'})

@app.route('/guardar-column-names-regresor', methods=['GET'])
def save_column_names_regresor():
    file_path = request.args.get('file_path')
    target = request.args.get('target')
    column_names = bosque_regresor.column_names(target)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
    return jsonify({'message': 'Columnas guardadas exitosamente.'})

@app.route('/cargar-column-names', methods=['GET'])
def load_column_names():
    file_path = request.args.get('file_path')
    data = pd.read_csv(file_path)
    column_names = data.columns.tolist()
    return jsonify({'column_names': column_names})

@app.route('/archivos_pkl', methods=['GET'])
def obtener_archivos_pkl():
    # Obtener el directorio actual del archivo
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    # Obtener los nombres de los archivos con extensión .pkl
    archivos_pkl = [os.path.splitext(archivo)[0] for archivo in os.listdir(directorio_actual) if archivo.endswith('.pkl')]
    return jsonify(archivos_pkl)


if __name__ == '__main__':
    app.run(debug=True)