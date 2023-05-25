import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib
matplotlib.use('Agg')  # Configurar el backend de Matplotlib antes de importarlo
import matplotlib.pyplot as plt

class Bosques:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("¡Datos cargados exitosamente!")
        except FileNotFoundError:
            print("Error al cargar el archivo. Verifica la ruta proporcionada.")

    def column_names(self):
        if self.data is not None:
            return self.data.columns.tolist()
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")

    def cargar_datos(self, csv_file, target_column):
        self.data = pd.read_csv(csv_file)
        self.data.dropna(inplace=True)  # Eliminar valores nulos
        self.X = self.data.drop(columns=[target_column])
        self.X = self.X.select_dtypes(exclude=['object'])
        self.y = self.label_encoder.fit_transform(self.data[target_column])  # LabelEncoder para la variable objetivo

    def dividir_datos(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def entrenar_modelo(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.model.fit(self.X_train, self.y_train)

    def predecir(self, X):
        return self.model.predict(X)

    def evaluar_modelo(self):
        y_pred = self.predecir(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy
    
    def graficar_curva_roc(self):
        y_pred_proba = self.model.predict_proba(self.X_test)
        n_classes = y_pred_proba.shape[1]
        plt.figure()
        for i in range(n_classes):
            y_true = self.y_test.copy()
            y_true[y_true != i] = -1
            y_true[y_true == i] = 1
            y_true[y_true == -1] = 0
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label='Clase %d (área = %0.2f)' % (i, roc_auc))

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC (Multiclase)')
        plt.legend(loc="lower right")
        plt.savefig('roc.png')
        plt.close()

