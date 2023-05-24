from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
from eda import EDA  # Importa la clase EDA desde tu archivo de clase EDA (eda.py)
from pca_p import PCA_P

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite')


app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

exploration = EDA('data_dir/data.csv')  # Crea una instancia de la clase EDA
# Cargar los datos
exploration.load_data()

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

@app.route('/forest', methods=['GET'])
def train_model():
    target_column = request.args.get('target_column')
    csv_file = 'data_dir/data.csv'  # Ruta y nombre de tu archivo CSV
    bosques = Bosques()
    bosques.cargar_datos(csv_file, target_column)
    bosques.dividir_datos()
    bosques.entrenar_modelo()
    accuracy = bosques.evaluar_modelo()
    return jsonify({'accuracy': accuracy})


@app.route('/forest-predict', methods=['POST'])
def predict():
    data = request.get_json()
    nuevos_datos = pd.DataFrame(data)
    bosques = Bosques()
    # Cargar el modelo previamente entrenado
    bosques.model = RandomForestClassifier(n_estimators=100)
    bosques.model = bosques.model.load('modelo_entrenado.pkl')
    predicciones = bosques.predecir(nuevos_datos)
    return jsonify({'predictions': predicciones.tolist()})


if __name__ == '__main__':
    app.run(debug=True)