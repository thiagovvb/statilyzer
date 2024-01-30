import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class BaseDataset:

    def __init__(self, file_path: str, delimiter: str, class_column: int = -1):
        
        self.file_path = file_path
        self.delimiter = delimiter
        self.class_column = class_column
        self.current_analysis = None
        self.dataset = None

    def load_file(self):

        self.dataset = pd.read_csv(filepath_or_buffer=self.file_path, delimiter=self.delimiter)

    def analyze_knn(self, n_neighbors: int):

        self.current_analysis = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(self.dataset)

        