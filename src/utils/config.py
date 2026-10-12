"""
Configuration globale du projet
"""
import os

# Chemins
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'gesture_model_final.keras')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Paramètres du modèle
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 3

# Classes
CLASS_NAMES = {
    0: "Poing fermé",
    1: "Paume ouverte", 
    2: "Victoire (V)"
}

CLASS_LABELS = ["Poing", "Paume", "Victoire"]

# Couleurs pour l'affichage (BGR)
CLASS_COLORS = {
    0: (255, 100, 100),  # Bleu
    1: (100, 255, 100),  # Vert
    2: (100, 100, 255)   # Rouge
}

# Paramètres de détection
MIN_DETECTION_AREA = 5000  # Aire minimale pour détecter une main
HAND_MARGIN = 20           # Marge autour de la main

# Paramètres de lissage
PREDICTION_HISTORY_SIZE = 5  # Nombre de frames pour le lissage

# Paramètres de la webcam
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Paramètres de prétraitement CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (4, 4)

# Paramètres de filtrage
GAUSSIAN_KERNEL_SIZE = (3, 3)