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

class TodoItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    is_executed = db.Column(db.Boolean)

    def __init__(self, name, is_executed):
        self.name = name
        self.is_executed = is_executed


# Todo schema
class TodoSchema(ma.Schema):
    class Meta:
        fields = ('id', 'name', 'is_executed')


# Initialize schema
todo_schema = TodoSchema()
todos_schema = TodoSchema(many=True)

@app.route('/todo', methods=['POST'])
def add_todo():
    name = request.json['name']
    is_executed = request.json['is_executed']

    new_todo_item = TodoItem(name, is_executed)
    db.session.add(new_todo_item)
    db.session.commit()

    return todo_schema.jsonify(new_todo_item)

@app.route('/todo', methods=['GET'])
def get_todos():
    all_todos = TodoItem.query.all()
    result = todos_schema.dump(all_todos)

    return jsonify(result)


@app.route('/todo/<id>', methods=['PUT', 'PATCH'])
def execute_todo(id):
    todo = TodoItem.query.get(id)

    todo.is_executed = not todo.is_executed
    db.session.commit()

    return todo_schema.jsonify(todo)


@app.route('/todo/<id>', methods=['DELETE'])
def delete_todo(id):
    todo_to_delete = TodoItem.query.get(id)
    db.session.delete(todo_to_delete)
    db.session.commit()

    return todo_schema.jsonify(todo_to_delete)

@app.route('/upload', methods=['POST'])
def upload():
    archivo = request.files['archivo']
    directory = check_directory() # if not exists, dir created for .CSV file
    archivo.save(os.path.join(directory,'data.csv'))  # Cambia 'ruta_de_guardado' a la ruta donde deseas guardar el archivo
    return 'Archivo guardado correctamente'

@app.route('/data-preview', methods=['GET'])
def data_preview():
    num_rows = request.args.get('num_rows', default=5, type=int)
    preview = exploration.preview_data(num_rows)
    column_names = preview.columns.tolist()
    preview_data_list = preview.to_dict(orient='records')
    # Preparar la respuesta en formato JSON
    response = {
        'column_names': column_names,
        'data': preview_data_list
    }
    # Enviar la respuesta al front-end
    return jsonify(response)

@app.route('/pca', methods=['GET'])
def perform_pca():
    num_rows = request.args.get('num_rows', default=2, type=int)
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




if __name__ == '__main__':
    app.run(debug=True)