from abc import ABC, abstractmethod


class DetectionModel(ABC):
    @abstractmethod
    def __init__(self, model_path, csv_results_path):
        self.model_path = model_path
        self.csv_results_path = csv_results_path
    
    @property
    @abstractmethod
    def model(self):
        pass

    
    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def evaluation(self, image_path):
        pass
 