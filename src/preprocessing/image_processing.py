"""
Module de prétraitement des images
"""
import cv2
import numpy as np
from ..utils import config


def pretraiter_pour_inference(image_bgr):
    """
    Prétraite une image BGR pour l'inférence du modèle
    
    Args:
        image_bgr (np.ndarray): Image BGR de la webcam
    
    Returns:
        np.ndarray: Image prétraitée (1, 28, 28, 1)
    """
    # Convertir en niveaux de gris
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Redimensionner à 28x28
    image_resized = cv2.resize(image_gray, (28, 28), 
                               interpolation=cv2.INTER_AREA)
    
    # Appliquer CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )
    image_clahe = clahe.apply(image_resized)
    
    # Filtrage gaussien
    image_filtered = cv2.GaussianBlur(
        image_clahe, 
        config.GAUSSIAN_KERNEL_SIZE, 
        0
    )
    
    # Normalisation
    image_normalized = image_filtered.astype(np.float32) / 255.0
    
    # Reshape pour le modèle
    image_processed = image_normalized.reshape(1, 28, 28, 1)
    
    return image_processed


def extraire_region_main(frame, x, y, w, h, marge=None):
    """
    Extrait une région carrée autour de la main
    
    Args:
        frame (np.ndarray): Image complète
        x, y, w, h (int): Coordonnées du rectangle
        marge (int, optional): Marge autour de la main
    
    Returns:
        tuple: (roi, coords) - ROI et coordonnées (x1, y1, x2, y2)
    """
    if marge is None:
        marge = config.HAND_MARGIN
    
    height, width = frame.shape[:2]
    
    # Centre
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Taille du carré
    size = max(w, h) + marge * 2
    
    # Coordonnées
    x1 = max(0, center_x - size // 2)
    y1 = max(0, center_y - size // 2)
    x2 = min(width, center_x + size // 2)
    y2 = min(height, center_y + size // 2)
    
    # Extraire ROI
    roi = frame[y1:y2, x1:x2]
    
    return roi, (x1, y1, x2, y2)