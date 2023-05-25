from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
from eda import EDA  # Importa la clase EDA desde tu archivo de clase EDA (eda.py)
from pca_p import PCA_P
from bosques import Bosques
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite')


app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

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
    # Crear una instancia de la clase PCA_P con el número de componentes principales deseado
    pca = PCA_P(n_components=num_rows)
    # Cargar los datos
    pca.load_data('data_dir/data.csv')
    # Ajustar el modelo PCA a los datos
    pca.fit()
    # Aplicar la transformación PCA y obtener los datos transformados
    transformed_data = pca.transform()
    # Obtener los nombres de las columnas del DataFrame transformado
    column_names = transformed_data.columns.tolist()
    # Convertir los datos transformados en una lista de diccionarios
    transformed_data_list = transformed_data.to_dict(orient='records')
    # Preparar la respuesta en formato JSON
    response = {
        'column_names': column_names,
        'data': transformed_data_list
    }
    # Enviar la respuesta al front-end
    return jsonify(response)

@app.route('/column-names', methods=['GET'])
def get_column_names():
    bosques = Bosques('data_dir/data.csv')
    bosques.load_data()
    column_names = bosques.column_names()
    return jsonify({'column_names': column_names})

@app.route('/forest', methods=['GET'])
def train_model():
    target_column = request.args.get('target_column')
    csv_file = 'data_dir/data.csv'  # Ruta y nombre de tu archivo CSV
    bosques = Bosques('data_dir/data.csv')
    bosques.cargar_datos(csv_file, target_column)
    bosques.dividir_datos()
    bosques.entrenar_modelo()
    accuracy = bosques.evaluar_modelo()
    # Generar la gráfica de la curva ROC
    bosques.graficar_curva_roc()
    # Leer la imagen de la curva ROC generada
    with open('roc.png', 'rb') as file:
        image_data = file.read()
    # Convertir la imagen a base64 para enviarla como respuesta
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    # Preparar la respuesta en formato JSON
    response = {
        'accuracy': accuracy,
        'roc_image': encoded_image
    }
    # Enviar la respuesta al front-end
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)