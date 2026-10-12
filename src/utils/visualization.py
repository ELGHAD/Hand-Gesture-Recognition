"""
Module de visualisation
"""
import cv2
from ..utils import config


def dessiner_roi_avec_prediction(frame, coords, classe, confiance, probabilites):
    """
    Dessine la ROI et la prédiction sur l'image
    
    Args:
        frame (np.ndarray): Image
        coords (tuple): Coordonnées (x1, y1, x2, y2)
        classe (int): Classe prédite
        confiance (float): Confiance (0-1)
        probabilites (np.ndarray): Probabilités pour toutes les classes
    
    Returns:
        np.ndarray: Image annotée
    """
    x1, y1, x2, y2 = coords
    couleur = config.CLASS_COLORS[classe]
    
    # Rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 3)
    
    # Texte
    texte = f"{config.CLASS_LABELS[classe]}: {confiance*100:.1f}%"
    
    # Fond pour le texte
    (text_width, text_height), _ = cv2.getTextSize(
        texte, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
    )
    cv2.rectangle(frame, 
                  (x1, y1 - text_height - 10),
                  (x1 + text_width, y1),
                  couleur, -1)
    
    # Texte blanc
    cv2.putText(frame, texte,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    return frame


def dessiner_probabilites(frame, probabilites):
    """
    Dessine les probabilités pour chaque classe
    
    Args:
        frame (np.ndarray): Image
        probabilites (np.ndarray): Probabilités
    
    Returns:
        np.ndarray: Image annotée
    """
    y_offset = 30
    
    for i, (label, prob) in enumerate(zip(config.CLASS_LABELS, probabilites)):
        texte = f"{label}: {prob*100:.1f}%"
        couleur = config.CLASS_COLORS[i]
        
        y_pos = frame.shape[0] - y_offset - i * 35
        
        cv2.putText(frame, texte,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)
    
    return frame


def dessiner_instructions(frame):
    """
    Dessine les instructions d'utilisation
    
    Args:
        frame (np.ndarray): Image
    
    Returns:
        np.ndarray: Image annotée
    """
    instructions = "Appuyez sur 'q' pour quitter | 's' pour screenshot"
    
    cv2.putText(frame, instructions,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame