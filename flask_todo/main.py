from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
from eda import EDA  # Importa la clase EDA desde tu archivo de clase EDA (eda.py)


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite')


app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

exploration = EDA('/Users/fernando_maceda/Documents/GitHub/backend_data_minig/flask_todo/data.csv')  # Crea una instancia de la clase EDA
# Cargar los datos
exploration.load_data()

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
    archivo.save(os.path.join('/Users/fernando_maceda/Documents/GitHub/backend_data_minig/flask_todo', 'data.csv'))  # Cambia 'ruta_de_guardado' a la ruta donde deseas guardar el archivo
    return 'Archivo guardado correctamente'

@app.route('/data-preview', methods=['GET'])
def data_preview():
    num_rows = request.args.get('num_rows', default=5, type=int)
    preview = exploration.preview_data(num_rows)
    return jsonify(preview)

@app.route('/summary-statistics', methods=['GET'])
def summary_statistics():
    statistics = exploration.summary_statistics()
    return jsonify(statistics)


if __name__ == '__main__':
    app.run(debug=True)