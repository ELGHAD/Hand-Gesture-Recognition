"""
Module de détection de la main
"""
import cv2
import numpy as np
from ..utils import config


class HandDetector:
    """Détecteur de main par couleur de peau"""
    
    def __init__(self, min_area=None):
        """
        Initialise le détecteur
        
        Args:
            min_area (int, optional): Aire minimale pour détecter une main
        """
        self.min_area = min_area or config.MIN_DETECTION_AREA
        
        # Plages HSV pour la peau
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Noyau pour opérations morphologiques
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    
    def detecter_main(self, frame):
        """
        Détecte la main dans l'image
        
        Args:
            frame (np.ndarray): Image BGR
        
        Returns:
            tuple: (contours, mask) - Liste de contours et masque binaire
        """
        # Convertir en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Créer le masque
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Opérations morphologiques
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        mask = cv2.erode(mask, self.kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Trouver les contours
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours, mask
    
    def trouver_plus_grand_contour(self, contours):
        """
        Trouve le plus grand contour valide
        
        Args:
            contours (list): Liste de contours
        
        Returns:
            np.ndarray or None: Le plus grand contour ou None
        """
        if len(contours) == 0:
            return None
        
        # Filtrer les petits contours
        contours_valides = [
            c for c in contours 
            if cv2.contourArea(c) > self.min_area
        ]
        
        if len(contours_valides) == 0:
            return None
        
        # Retourner le plus grand
        return max(contours_valides, key=cv2.contourArea)