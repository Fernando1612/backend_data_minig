import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.cluster import KMeans
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

class KMEANS:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.data = None
        self.scaler = None
        self.feature_names = None
        self.labels = None
        self.new_data = None
        
    def load_data(self, filename):
        """
        Carga los datos desde un archivo CSV.

        Args:
            filename: Ruta del archivo CSV.
        """
        self.data = pd.read_csv(filename)
        
    def preprocess_data(self):
        """
        Realiza el preprocesamiento básico de los datos.
        Elimina columnas de tipo 'object' y filas con valores nulos.
        """
        if self.data is None:
            raise ValueError("No se han cargado los datos. Utilice el método 'load_data' para cargar los datos desde un archivo CSV.")
        string_columns = self.data.select_dtypes(include=['object']).columns
        self.data = self.data.drop(string_columns, axis=1)
        self.data = self.data.dropna()
        self.feature_names = list(self.data.columns)[:-1]



    def created_df(self, n_clusters, scaler_type):
        """
        Crea un dataframe con los resultados del algoritmo K-means.

        Args:
            n_clusters: Número de clusters.
            scaler_type: Tipo de escala a aplicar.

        Returns:
            new_data: Dataframe con los datos y las etiquetas de cluster asignadas.
            CentroidesP: Dataframe con los centroides de cada cluster.
            count_df: Dataframe con la cantidad de elementos en cada cluster.
        """
        if scaler_type == "StandardScaler":
            self.scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        elif scaler_type == "Normalizer":
            self.scaler = Normalizer()
        else:
            raise ValueError("Tipo de estandarización no válido. Los tipos válidos son: 'StandardScaler', 'MinMaxScaler', 'Normalizer'.")

        string_columns = self.data.select_dtypes(include=['object']).columns
        self.data = self.data.drop(string_columns, axis=1)
        self.data = self.data.dropna()

        MEstandarizada = self.scaler.fit_transform(self.data)
        MParticional = KMeans(n_clusters=n_clusters, random_state=0).fit(MEstandarizada)
        labels = MParticional.predict(MEstandarizada)

        self.new_data = self.data
        self.new_data['clusterP'] = labels

        CentroidesP = self.new_data.groupby('clusterP').mean().reset_index()

        # Cantidad de elementos en los clusters
        count_df = self.new_data.groupby('clusterP').size().reset_index(name='count')

        plt.rcParams['figure.figsize'] = (10, 7)
        plt.style.use('ggplot')

        # Crear una lista dinámica de colores basada en n_clusters
        colores = list(mcolors.TABLEAU_COLORS.keys())[:n_clusters]

        asignar = [colores[row] for row in MParticional.labels_]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Subgrafico
        ax.scatter(
            MEstandarizada[:, 0],
            MEstandarizada[:, 1],
            MEstandarizada[:, 2],
            marker='o',
            c=asignar,
            s=60
        )
        ax.scatter(
            MParticional.cluster_centers_[:, 0],
            MParticional.cluster_centers_[:, 1],
            MParticional.cluster_centers_[:, 2],
            marker='o',
            c=colores,
            s=1000
        )
        plt.savefig('data_k.png')
        plt.close()

        return self.new_data, CentroidesP, count_df


    def save_data_frame(self, n_clusters, scaler_type):
        """
        Crea un dataframe con los resultados del algoritmo K-means y lo retorna.

        Args:
            n_clusters: Número de clusters.
            scaler_type: Tipo de escala a aplicar.

        Returns:
            new_data: Dataframe con los datos y las etiquetas de cluster asignadas.
        """
        if scaler_type == "StandardScaler":
            self.scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        elif scaler_type == "Normalizer":
            self.scaler = Normalizer()
        else:
            raise ValueError("Tipo de estandarización no válido. Los tipos válidos son: 'StandardScaler', 'MinMaxScaler', 'Normalizer'.")

        MEstandarizada = self.scaler.fit_transform(self.data)
        MParticional = KMeans(n_clusters=n_clusters, random_state=0).fit(MEstandarizada)
        labels = MParticional.predict(MEstandarizada)

        self.new_data = self.data
        self.new_data['clusterP'] = labels
        
        return self.new_data
 
        
    def fit(self, scaler_type="StandardScaler"):
        """
        Ajusta el modelo de K-means a los datos cargados.

        Args:
            scaler_type: Tipo de escala a aplicar.

        Raises:
            ValueError: Si no se han cargado los datos.
        """
        if self.data is None:
            raise ValueError("No se han cargado los datos. Utilice el método 'load_data' para cargar los datos desde un archivo CSV.")
        self.preprocess_data()
        X = self.data.iloc[:, :-1].values
        if scaler_type == "StandardScaler":
            self.scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        elif scaler_type == "Normalizer":
            self.scaler = Normalizer()
        else:
            raise ValueError("Tipo de estandarización no válido. Los tipos válidos son: 'StandardScaler', 'MinMaxScaler', 'Normalizer'.")
        
        X_std = self.scaler.fit_transform(X)
        self.kmeans.fit(X_std)
        self.labels = self.kmeans.labels_
        
    def transform(self):
        """
        Realiza la transformación de los datos cargados utilizando el modelo K-means ajustado.

        Returns:
            transformed_df: Dataframe con las componentes transformadas por el modelo K-means.
        """
        if self.data is None:
            raise ValueError("No se han cargado los datos. Utilice el método 'load_data' para cargar los datos desde un archivo CSV.")
        
        self.preprocess_data()
        X = self.data.iloc[:, :-1].values
        X_std = self.scaler.transform(X)
        X_transformed = self.kmeans.transform(X_std)
        
        transformed_df = pd.DataFrame(X_transformed, columns=['Component ' + str(i+1) for i in range(self.n_clusters)])
        
        if 'Label' in self.data.columns:
            transformed_df['Label'] = self.data['Label'].values
        
        return transformed_df
    
    def plot_elbow(self):
        """
        Grafica el método del codo para determinar el número óptimo de clusters.

        Raises:
            ValueError: Si no se han cargado los datos.
        """
        if self.data is None:
            raise ValueError("No se han cargado los datos. Utilice el método 'load_data' para cargar los datos desde un archivo CSV.")
        
        self.preprocess_data()
        X = self.data.iloc[:, :-1].values
        
        SSE = []
        for i in range(2, 10):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(X)
            SSE.append(km.inertia_)
        
        kl = KneeLocator(range(2, 10), SSE, curve="convex", direction="decreasing")
        kl.plot_knee()
        plt.xlabel('Cantidad de clusters *k*')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        plt.style.use('ggplot')  # Establecer el estilo del gráfico
        
        plt.savefig('kmeans.png')  # Guardar la imagen en un archivo
        plt.close()  # Cerrar la figura para liberar memoria
