from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
from eda import EDA  # Importa la clase EDA desde tu archivo de clase EDA (eda.py)
from pca_p import PCA_P
from bosques import Bosques
from bosque_regresor import BosqueRegresor
from kmeans import KMEANS
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
    directory = check_directory()  # Si no existe, se crea el directorio para el archivo .CSV
    archivo.save(os.path.join(directory,'data.csv'))  # Guarda el archivo en el directorio de datos especificado
    return 'Archivo guardado correctamente'  # Archivo guardado exitosamente

@app.route('/data-preview', methods=['GET'])
def data_preview():
    num_rows = request.args.get('num_rows', default=5, type=int)
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA con el archivo de datos especificado
    exploration.load_data()  # Carga los datos en la instancia de EDA
    preview = exploration.preview_data(num_rows)  # Obtiene una vista previa de los datos con el número de filas especificado
    column_names = preview.columns.tolist()  # Obtiene los nombres de las columnas del DataFrame original
    preview_data_list = preview[column_names].to_dict(orient='records')  # Convierte los datos a una lista de diccionarios manteniendo el orden de las columnas
    response = {
        'column_names': column_names,
        'data': preview_data_list
    }
    return jsonify(response)  # Retorna la respuesta en formato JSON para el front-end

@app.route('/data-statistics', methods=['GET'])
def data_stats():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA con el archivo de datos especificado
    exploration.load_data()  # Carga los datos en la instancia de EDA
    stats = exploration.summary_statistics()  # Obtiene las estadísticas resumidas de los datos
    column_names = stats.columns.tolist()  # Obtiene los nombres de las columnas del DataFrame original
    indx_names = stats.index.tolist()  # Obtiene los nombres de los índices del DataFrame original
    preview_data_list = stats[column_names].to_dict(orient='records')  # Convierte los datos a una lista de diccionarios manteniendo el orden de las columnas
    response = {
        'column_names': column_names,
        'index_names': indx_names,
        'data': preview_data_list
    }
    return jsonify(response)  # Retorna la respuesta en formato JSON para el front-end

@app.route('/data-nulls', methods=['GET'])
def data_nulls():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA con el archivo de datos especificado
    exploration.load_data()  # Carga los datos en la instancia de EDA
    preview = exploration.missing_values()  # Obtiene los valores faltantes de los datos
    column_names = preview.columns.tolist()  # Obtiene los nombres de las columnas del DataFrame original
    indx_names = preview.index.tolist()  # Obtiene los nombres de los índices del DataFrame original
    preview_data_list = preview[column_names].to_dict(orient='records')  # Convierte los datos a una lista de diccionarios manteniendo el orden de las columnas
    response = {
        'column_names': column_names,
        'index_names': indx_names,
        'data': preview_data_list
    }
    return jsonify(response)  # Retorna la respuesta en formato JSON para el front-end

@app.route('/data-all', methods=['GET'])
def data_all():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA con el archivo de datos especificado
    exploration.load_data()  # Carga los datos en la instancia de EDA
    preview = exploration.get_all_data()  # Obtiene todos los datos
    column_names = preview.columns.tolist()  # Obtiene los nombres de las columnas del DataFrame original
    preview_data_list = preview[column_names].to_dict(orient='records')  # Convierte los datos a una lista de diccionarios manteniendo el orden de las columnas
    response = {
        'column_names': column_names,
        'data': preview_data_list
    }
    return jsonify(response)  # Retorna la respuesta en formato JSON para el front-end


@app.route('/plot-histogram', methods=['GET'])
def plot_histogram():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA con el archivo de datos especificado
    exploration.load_data()  # Carga los datos en la instancia de EDA
    file_name = exploration.plot_outliers_histogram()  # Genera el histograma de los valores atípicos y devuelve el nombre del archivo
    return send_file(file_name, mimetype='image/png')  # Envía el archivo de imagen al front-end

@app.route('/plot-boxplot', methods=['GET'])
def plot_boxplot():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA con el archivo de datos especificado
    exploration.load_data()  # Carga los datos en la instancia de EDA
    file_name = exploration.plot_outliers_boxplot()  # Genera el diagrama de caja de los valores atípicos y devuelve el nombre del archivo
    return send_file(file_name, mimetype='image/png')  # Envía el archivo de imagen al front-end

@app.route('/plot-heatmap', methods=['GET'])
def plot_heatmap():
    exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA con el archivo de datos especificado
    exploration.load_data()  # Carga los datos en la instancia de EDA
    file_name = exploration.plot_correlation_heatmap()  # Genera el mapa de calor de la correlación y devuelve el nombre del archivo
    return send_file(file_name, mimetype='image/png')  # Envía el archivo de imagen al front-end

@app.route('/pca', methods=['GET'])
def perform_pca():
    num_rows = request.args.get('n_components', default=2, type=int)  # Obtiene el número de componentes principales de la solicitud
    scaler_type = request.args.get('scaler_type', default='StandardScaler', type=str)  # Obtiene el tipo de escalador de la solicitud
    pca = PCA_P(n_components=num_rows)  # Crea una instancia de la clase PCA_P con el número de componentes principales especificado
    pca.load_data('data_dir/data.csv')  # Carga los datos en la instancia de PCA_P
    pca.fit(scaler_type=scaler_type)  # Ajusta el modelo PCA a los datos utilizando el tipo de escalador especificado
    transformed_data = pca.transform()  # Aplica la transformación PCA y obtiene los datos transformados
    column_names = transformed_data.columns.tolist()  # Obtiene los nombres de las columnas del DataFrame transformado
    transformed_data_list = transformed_data.to_dict(orient='records')  # Convierte los datos transformados a una lista de diccionarios
    file_name, varianza = pca.plot_variance(n_components=num_rows)  # Obtiene el nombre del archivo de la imagen y el valor de la varianza acumulada
    response = {
        'column_names': column_names,
        'data': transformed_data_list,
        'varianza': varianza
    }
    return jsonify(response)  # Retorna la respuesta en formato JSON para el front-end


@app.route('/pca-plot', methods=['GET'])
def plot_pca():
    num_rows = request.args.get('n_components', default=2, type=int)  # Obtiene el número de componentes principales de la solicitud
    scaler_type = request.args.get('scaler_type', default='StandardScaler', type=str)  # Obtiene el tipo de escalador de la solicitud
    pca = PCA_P(n_components=num_rows)  # Crea una instancia de la clase PCA_P con el número de componentes principales especificado
    pca.load_data('data_dir/data.csv')  # Carga los datos en la instancia de PCA_P
    pca.fit(scaler_type=scaler_type)  # Ajusta el modelo PCA a los datos utilizando el tipo de escalador especificado
    transformed_data = pca.transform()  # Aplica la transformación PCA y obtiene los datos transformados
    column_names = transformed_data.columns.tolist()  # Obtiene los nombres de las columnas del DataFrame transformado
    transformed_data_list = transformed_data.to_dict(orient='records')  # Convierte los datos transformados a una lista de diccionarios
    file_name, varianza = pca.plot_variance(n_components=num_rows)  # Obtiene el nombre del archivo de la imagen y el valor de la varianza acumulada
    return send_file(file_name, mimetype='image/png')  # Envía el archivo de imagen al front-end


@app.route('/column-names', methods=['GET'])
def get_column_names():
    bosques = Bosques('data_dir/data.csv')  # Crea una instancia de la clase Bosques con el archivo de datos especificado
    bosques.load_data()  # Carga los datos en la instancia de Bosques
    column_names = bosques.column_names()  # Obtiene los nombres de las columnas del dataset
    return jsonify({'column_names': column_names})  # Retorna los nombres de las columnas en formato JSON


@app.route('/forest', methods=['GET'])
def train_model():
    target_column = request.args.get('target_column')  # Obtiene el nombre de la columna objetivo de la solicitud
    n_estimators = int(request.args.get('n_estimators', 100))  # Obtiene el número de estimadores de la solicitud, con valor predeterminado 100
    criterion = request.args.get('criterion', 'gini')  # Obtiene el criterio de división de la solicitud, con valor predeterminado 'gini'
    max_depth = request.args.get('max_depth', None)  # Obtiene la profundidad máxima de los árboles de la solicitud, con valor predeterminado None
    if max_depth == 'None':
        max_depth = None
    else:
        max_depth = int(max_depth)
    min_samples_split = int(request.args.get('min_samples_split', 2))  # Obtiene el número mínimo de muestras requeridas para dividir un nodo interno de la solicitud, con valor predeterminado 2
    min_samples_leaf = int(request.args.get('min_samples_leaf', 1))  # Obtiene el número mínimo de muestras requeridas para formar una hoja de la solicitud, con valor predeterminado 1
    max_features = request.args.get('max_features', 'auto')  # Obtiene la cantidad de características a considerar al buscar la mejor división de la solicitud, con valor predeterminado 'auto'

    csv_file = 'data_dir/data.csv'  # Ruta y nombre de tu archivo CSV
    bosques = Bosques(csv_file)  # Crea una instancia de la clase Bosques con el archivo de datos especificado
    bosques.cargar_datos(csv_file, target_column)  # Carga los datos y define la columna objetivo en la instancia de Bosques
    bosques.dividir_datos()  # Divide los datos en conjuntos de entrenamiento y prueba en la instancia de Bosques
    bosques.entrenar_modelo(n_estimators=n_estimators, criterion=criterion,
                            max_depth=max_depth, min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf, max_features=max_features)  # Entrena el modelo de Bosques con los parámetros especificados
    accuracy, _ = bosques.evaluar_modelo()  # Evalúa el modelo entrenado y obtiene la precisión y otras métricas

    bosques.mostrar_matriz_confusion()  # Muestra la matriz de confusión del modelo entrenado
    with open('roc.png', 'rb') as file:
        image_data = file.read()  # Lee los datos de la imagen de la curva ROC generada
    encoded_image = base64.b64encode(image_data).decode('utf-8')  # Codifica la imagen a base64 para enviarla como respuesta

    with open('confusion_matrix.png', 'rb') as file:
        image_data_matrix = file.read()  # Lee los datos de la imagen de la matriz de confusión generada
    encoded_image_matrix = base64.b64encode(image_data_matrix).decode('utf-8')  # Codifica la imagen a base64 para enviarla como respuesta
    response = {
        'accuracy': accuracy,
        'roc_image': encoded_image,
        'matrix_image': encoded_image_matrix
    }
    return jsonify(response)  # Retorna la respuesta en formato JSON para el front-end

@app.route('/forest-predict', methods=['GET'])
def predict_data():
    feature_params = request.args.to_dict()  # Obtener las características de entrada para la predicción de la solicitud
    df = pd.DataFrame([feature_params])  # Convertir feature_params en un DataFrame
    prediction = bosques.predecir(df)  # Realizar la predicción utilizando el método predecir de la clase Bosques
    response = {
        'prediction': prediction.tolist()  # Preparar la respuesta en formato JSON con la predicción convertida a lista
    }
    return json.dumps(response)  # Enviar la respuesta al front-end en formato JSON

@app.route('/forest-regressor-train', methods=['GET'])
def train_model_regressor():
    target_column = request.args.get('target_column')  # Obtener el nombre de la columna objetivo de la solicitud
    n_estimators = int(request.args.get('n_estimators', 100))  # Obtener el número de estimadores de la solicitud, con valor predeterminado 100
    criterion = request.args.get('criterion', 'mse')  # Obtener el criterio de división de la solicitud, con valor predeterminado 'mse'
    max_depth = request.args.get('max_depth', None)  # Obtener la profundidad máxima de los árboles de la solicitud, con valor predeterminado None
    if max_depth == 'None':
        max_depth = None
    else:
        max_depth = int(max_depth)
    min_samples_split = int(request.args.get('min_samples_split', 2))  # Obtener el número mínimo de muestras requeridas para dividir un nodo interno de la solicitud, con valor predeterminado 2
    min_samples_leaf = int(request.args.get('min_samples_leaf', 1))  # Obtener el número mínimo de muestras requeridas para formar una hoja de la solicitud, con valor predeterminado 1
    max_features = request.args.get('max_features', 'auto')  # Obtener la cantidad de características a considerar al buscar la mejor división de la solicitud, con valor predeterminado 'auto'
    csv_file = 'data_dir/data.csv'  # Ruta y nombre de tu archivo CSV
    bosque_regresor.cargar_datos(csv_file, target_column)  # Carga los datos y define la columna objetivo en la instancia de BosqueRegresor
    bosque_regresor.dividir_datos()  # Divide los datos en conjuntos de entrenamiento y prueba en la instancia de BosqueRegresor
    bosque_regresor.entrenar_modelo(n_estimators=n_estimators, criterion=criterion,
                                     max_depth=max_depth, min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf, max_features=max_features)  # Entrena el modelo de BosqueRegresor con los parámetros especificados
    mse, criterion, feature_importances, mae, rmse, r2 = bosque_regresor.evaluar_modelo()  # Evalúa el modelo entrenado y obtiene métricas

    bosque_regresor.graficar_pronostico()  # Genera el gráfico de pronóstico

    with open('pronostico.png', 'rb') as file:
        image_data = file.read()  # Lee los datos de la imagen del gráfico de pronóstico generada
    encoded_image = base64.b64encode(image_data).decode('utf-8')  # Codifica la imagen a base64 para enviarla como respuesta

    response = {
        'mse': mse,
        'criterion': criterion,
        'feature_importances': feature_importances.tolist(),
        'mean_absolute_error': mae,
        'RMSE': rmse,
        'r2_score': r2,
        'pronostico_image': encoded_image,
    }

    return jsonify(response)  # Enviar la respuesta al front-end en formato JSON

@app.route('/forest-regressor-predict', methods=['GET'])
def predict_data_regressor():
    feature_params = request.args.to_dict()  # Obtener las características de entrada para la predicción de la solicitud
    df = pd.DataFrame([feature_params])  # Convertir feature_params en un DataFrame
    prediction = bosque_regresor.predecir(df)  # Realizar la predicción utilizando el método predecir de la clase BosqueRegresor
    response = {
        'prediction': prediction.tolist()  # Preparar la respuesta en formato JSON con la predicción convertida a lista
    }
    return json.dumps(response)  # Enviar la respuesta al front-end en formato JSON

@app.route('/guardar-modelo-regresor', methods=['GET'])
def guardar_modelo_regresor():
    file_path = request.args.get('file_path')  # Obtener la ruta del archivo para guardar el modelo
    bosque_regresor.guardar_modelo(file_path)  # Guardar el modelo en el archivo especificado
    return jsonify({'message': 'Modelo guardado exitosamente.'})

@app.route('/cargar-modelo-regresor', methods=['GET'])
def cargar_modelo_regresor():
    global modelo_rf  # Indicar que se usará la variable global
    file_path = request.args.get('file_path')  # Obtener la ruta del archivo para cargar el modelo
    with open(file_path, 'rb') as file:
        modelo_rf = pickle.load(file)  # Cargar el modelo desde el archivo
    return jsonify({'message': 'Modelo cargado exitosamente.'})

@app.route('/guardar-modelo-clasificador', methods=['GET'])
def guardar_modelo():
    file_path = request.args.get('file_path')  # Obtener la ruta del archivo para guardar el modelo
    bosques.guardar_modelo(file_path)  # Guardar el modelo en el archivo especificado
    return jsonify({'message': 'Modelo guardado exitosamente.'})

@app.route('/cargar-modelo-clasificador', methods=['GET'])
def cargar_modelo_clasificador():
    global modelo_rf  # Indicar que se usará la variable global
    file_path = request.args.get('file_path')  # Obtener la ruta del archivo para cargar el modelo
    with open(file_path, 'rb') as file:
        modelo_rf = pickle.load(file)  # Cargar el modelo desde el archivo
    return jsonify({'message': 'Modelo cargado exitosamente.'})

@app.route('/predict-modelo', methods=['GET'])
def predecir_modelo():
    feature_params = request.args.to_dict()  # Obtener las características de entrada para la predicción de la solicitud
    df = pd.DataFrame([feature_params])  # Convertir feature_params en un DataFrame
    prediction = modelo_rf.predict(df)  # Realizar la predicción utilizando el modelo cargado
    response = {
        'prediction': prediction.tolist()  # Preparar la respuesta en formato JSON con la predicción convertida a lista
    }
    return json.dumps(response)  # Enviar la respuesta al front-end en formato JSON

@app.route('/guardar-column-names', methods=['GET'])
def save_column_names():
    file_path = request.args.get('file_path')  # Obtener la ruta del archivo para guardar los nombres de las columnas
    target = request.args.get('target')  # Obtener el nombre de la columna objetivo
    column_names = bosques.column_names(target)  # Obtener los nombres de las columnas
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)  # Escribir los nombres de las columnas en el archivo CSV
    return jsonify({'message': 'Columnas guardadas exitosamente.'})

@app.route('/guardar-column-names-regresor', methods=['GET'])
def save_column_names_regresor():
    file_path = request.args.get('file_path')  # Obtener la ruta del archivo para guardar los nombres de las columnas
    target = request.args.get('target')  # Obtener el nombre de la columna objetivo
    column_names = bosque_regresor.column_names(target)  # Obtener los nombres de las columnas
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)  # Escribir los nombres de las columnas en el archivo CSV
    return jsonify({'message': 'Columnas guardadas exitosamente.'})

@app.route('/cargar-column-names', methods=['GET'])
def load_column_names():
    file_path = request.args.get('file_path')  # Obtener la ruta del archivo para cargar los nombres de las columnas
    data = pd.read_csv(file_path)  # Leer el archivo CSV
    column_names = data.columns.tolist()  # Obtener los nombres de las columnas del DataFrame
    return jsonify({'column_names': column_names})

@app.route('/kmeans-elbow', methods=['GET'])
def plot_kmeans_elbow():
    scaler_type = request.args.get('scaler_type', default='StandardScaler', type=str)  # Obtener el tipo de escalador como parámetro de la solicitud GET
    n_clusters = request.args.get('n_clusters', default=2, type=int)  # Obtener el número de clusters como parámetro de la solicitud GET
    kmeans = KMEANS(n_clusters)  # Crear una instancia del objeto KMEANS con el número de clusters especificado
    kmeans.load_data('data_dir/data.csv')  # Cargar los datos desde el archivo CSV
    kmeans.fit(scaler_type=scaler_type)  # Ajustar el algoritmo K-means a los datos con el tipo de escalador especificado
    kmeans.plot_elbow()  # Generar el gráfico de codo del K-means
    
    return send_file('kmeans.png', mimetype='image/png')  # Devolver el archivo de imagen generado como respuesta

@app.route('/kmeans', methods=['GET'])
def kmeans():
    scaler_type = request.args.get('scaler_type', default='StandardScaler', type=str)  # Obtener el tipo de escalador como parámetro de la solicitud GET
    n_clusters = request.args.get('n_clusters', default=2, type=int)  # Obtener el número de clusters como parámetro de la solicitud GET
    kmeans = KMEANS(n_clusters)  # Crear una instancia del objeto KMEANS con el número de clusters especificado

    kmeans.load_data('data_dir/data.csv')  # Cargar los datos desde el archivo CSV

    dataFrame, centroide, count_df = kmeans.created_df(n_clusters, scaler_type)  # Crear el DataFrame, centroide y el recuento de datos
    preview_data_list = dataFrame.head(10)  # Obtener una vista previa de los primeros 10 registros del DataFrame

    column_names = dataFrame.columns.tolist()  # Obtener los nombres de las columnas del DataFrame
    preview_data_list = preview_data_list[column_names].to_dict(orient='records')  # Convertir la vista previa de datos en un diccionario

    column_names_centroide = centroide.columns.tolist()  # Obtener los nombres de las columnas del DataFrame de centroides
    preview_data_list_centroide = centroide[column_names_centroide].to_dict(orient='records')  # Convertir el DataFrame de centroides en un diccionario

    column_names_count = count_df.columns.tolist()  # Obtener los nombres de las columnas del DataFrame de recuento
    preview_data_list_count = count_df[column_names_count].to_dict(orient='records')  # Convertir el DataFrame de recuento en un diccionario

    with open('data_k.png', 'rb') as file:  # Abrir el archivo de imagen en modo de lectura binaria
        image_data = file.read()  # Leer los datos de la imagen en binario
    encoded_image = base64.b64encode(image_data).decode('utf-8')  # Codificar la imagen en base64

    response = {
        'column_names': column_names,
        'data': preview_data_list,

        'column_names_centroide': column_names_centroide,
        'data_centroide': preview_data_list_centroide,

        'column_names_count': column_names_count,
        'data_count': preview_data_list_count,

        'kmeans_image': encoded_image,
    }
    return jsonify(response)

@app.route('/guardar-data-frame', methods=['GET'])
def save_data_frame():
    file_path = request.args.get('file_path')
    scaler_type = request.args.get('scaler_type', default='StandardScaler', type=str)
    n_clusters = request.args.get('n_clusters', default=2, type=int)
    kmeans = KMEANS(n_clusters)
    kmeans.load_data('data_dir/data.csv')
    dataFrame = kmeans.save_data_frame(n_clusters, scaler_type)
    dataFrame.to_csv(file_path, header=True, index=False)
    return jsonify({'message': 'Columnas guardadas exitosamente.'})

@app.route('/archivos_pkl', methods=['GET'])
def obtener_archivos_pkl():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))  # Obtener el directorio actual del archivo
    archivos_pkl = [os.path.splitext(archivo)[0] for archivo in os.listdir(directorio_actual) if archivo.endswith('.pkl')]  # Obtener los nombres de los archivos con extensión .pkl
    return jsonify(archivos_pkl)


if __name__ == '__main__':
    app.run(debug=True)
