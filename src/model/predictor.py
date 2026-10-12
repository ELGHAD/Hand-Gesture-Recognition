"""
Module de prédiction
"""
import numpy as np
from ..utils import config


class GesturePredictor:
    """Prédicteur de gestes"""
    
    def __init__(self, model, history_size=None):
        """
        Initialise le prédicteur
        
        Args:
            model: Modèle Keras
            history_size (int, optional): Taille de l'historique pour lissage
        """
        self.model = model
        self.history_size = history_size or config.PREDICTION_HISTORY_SIZE
        self.predictions_history = []
    
    def predire(self, image_processed):
        """
        Fait une prédiction sur une image prétraitée
        
        Args:
            image_processed (np.ndarray): Image prétraitée (1, 28, 28, 1)
        
        Returns:
            tuple: (classe, confiance, probabilites)
        """
        # Prédiction
        prediction = self.model.predict(image_processed, verbose=0)[0]
        
        # Ajouter à l'historique
        self.predictions_history.append(prediction)
        if len(self.predictions_history) > self.history_size:
            self.predictions_history.pop(0)
        
        # Moyenne lissée
        avg_prediction = np.mean(self.predictions_history, axis=0)
        
        # Classe et confiance
        classe = np.argmax(avg_prediction)
        confiance = avg_prediction[classe]
        
        return classe, confiance, avg_prediction
    
    def reset_history(self):
        """Réinitialise l'historique des prédictions"""
        self.predictions_history = []