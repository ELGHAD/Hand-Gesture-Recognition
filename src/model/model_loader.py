"""
Module de chargement du modèle
"""
import os
from tensorflow import keras
from ..utils import config


class ModelLoader:
    """Chargeur de modèle CNN"""
    
    def __init__(self, model_path=None):
        """
        Initialise le chargeur
        
        Args:
            model_path (str, optional): Chemin du modèle
        """
        self.model_path = model_path or config.MODEL_PATH
        self.model = None
    
    def charger_modele(self):
        """
        Charge le modèle depuis le disque
        
        Returns:
            keras.Model: Modèle chargé
        
        Raises:
            FileNotFoundError: Si le modèle n'existe pas
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Modèle introuvable : {self.model_path}\n"
                f"Assurez-vous d'avoir entraîné le modèle avec le notebook 05."
            )
        
        print(f"Chargement du modèle depuis {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print(f"Modèle chargé avec succès ({self.model.count_params():,} paramètres)")
        
        return self.model
    
    def get_model(self):
        """
        Retourne le modèle (le charge si nécessaire)
        
        Returns:
            keras.Model: Modèle
        """
        if self.model is None:
            self.charger_modele()
        
        return self.model