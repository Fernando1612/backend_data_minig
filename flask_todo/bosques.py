import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Bosques:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def cargar_datos(self, csv_file, target_column):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]

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