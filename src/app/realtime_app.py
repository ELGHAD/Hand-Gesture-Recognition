"""
Application de reconnaissance de gestes en temps réel
"""
import cv2
import os
from ..preprocessing.image_processing import pretraiter_pour_inference, extraire_region_main
from ..detection.hand_detector import HandDetector
from ..model.model_loader import ModelLoader
from ..model.predictor import GesturePredictor
from ..utils.visualization import (
    dessiner_roi_avec_prediction,
    dessiner_probabilites,
    dessiner_instructions
)
from ..utils import config


class RealtimeGestureApp:
    """Application de reconnaissance de gestes en temps réel"""
    
    def __init__(self):
        """Initialise l'application"""
        print("="*70)
        print("INITIALISATION DE L'APPLICATION")
        print("="*70)
        
        # Charger le modèle
        model_loader = ModelLoader()
        self.model = model_loader.charger_modele()
        
        # Initialiser les composants
        self.detector = HandDetector()
        self.predictor = GesturePredictor(self.model)
        
        # Variables
        self.cap = None
        self.frame_count = 0
        self.screenshot_count = 0
        
        print("Application initialisée\n")
    
    def demarrer(self):
        """Démarre l'application"""
        print("Démarrage de la webcam...")
        print("\nInstructions :")
        print("   - Placez votre main devant la webcam")
        print("   - Essayez les 3 gestes : Poing, Paume, Victoire")
        print("   - Appuyez sur 'q' pour quitter")
        print("   - Appuyez sur 's' pour capturer une image\n")
        
        # Ouvrir la webcam
        self.cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        
        if not self.cap.isOpened():
            print("Erreur : Impossible d'ouvrir la webcam")
            return
        
        try:
            self._boucle_principale()
        except KeyboardInterrupt:
            print("\nInterruption clavier")
        finally:
            self._nettoyer()
    
    def _boucle_principale(self):
        """Boucle principale de traitement"""
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Erreur de lecture")
                break
            
            self.frame_count += 1
            display_frame = frame.copy()
            
            # Détecter la main
            contours, mask = self.detector.detecter_main(frame)
            main_contour = self.detector.trouver_plus_grand_contour(contours)
            
            if main_contour is not None:
                # Obtenir le rectangle englobant
                x, y, w, h = cv2.boundingRect(main_contour)
                
                # Extraire la ROI
                roi, coords = extraire_region_main(frame, x, y, w, h)
                
                if roi.size > 0:
                    # Prétraiter et prédire
                    image_processed = pretraiter_pour_inference(roi)
                    classe, confiance, probabilites = self.predictor.predire(image_processed)
                    
                    # Dessiner
                    display_frame = dessiner_roi_avec_prediction(
                        display_frame, coords, classe, confiance, probabilites
                    )
                    display_frame = dessiner_probabilites(display_frame, probabilites)
            
            # Instructions
            display_frame = dessiner_instructions(display_frame)
            
            # Afficher
            cv2.imshow('Reconnaissance de Gestes - Temps Reel', display_frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nArrêt demandé")
                break
            elif key == ord('s'):
                self._sauvegarder_screenshot(display_frame)
    
    def _sauvegarder_screenshot(self, frame):
        """Sauvegarde un screenshot"""
        os.makedirs(os.path.join(config.RESULTS_DIR, 'plots'), exist_ok=True)
        filename = os.path.join(
            config.RESULTS_DIR, 
            'plots', 
            f'screenshot_{self.screenshot_count:03d}.png'
        )
        cv2.imwrite(filename, frame)
        self.screenshot_count += 1
        print(f"Screenshot sauvegardé : {filename}")
    
    def _nettoyer(self):
        """Nettoie les ressources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nStatistiques :")
        print(f"   - Frames traitées : {self.frame_count}")
        print(f"   - Screenshots     : {self.screenshot_count}")
        print("\nApplication terminée")