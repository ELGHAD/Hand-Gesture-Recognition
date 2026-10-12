"""
Point d'entrée principal de l'application
"""
import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.dirname(__file__))

from src.app.realtime_app import RealtimeGestureApp


def main():
    """Fonction principale"""
    try:
        app = RealtimeGestureApp()
        app.demarrer()
    except Exception as e:
        print(f"\nErreur : {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
