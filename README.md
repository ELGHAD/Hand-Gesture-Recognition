# Reconnaissance de gestes de la main

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8)
![Licence](https://img.shields.io/badge/Licence-MIT-lightgrey)

Système de reconnaissance de gestes statiques de la main, de la constitution du jeu de données jusqu'à l'inférence en temps réel sur webcam. Un réseau de neurones convolutif (CNN) classe trois gestes couramment utilisés en interaction homme-machine : **poing fermé**, **paume ouverte** et **signe de victoire**.

Le projet couvre l'ensemble de la chaîne d'un problème de vision par ordinateur : génération et exploration des données, prétraitement avec OpenCV, conception et entraînement du modèle, évaluation détaillée, puis intégration dans une application temps réel au code modulaire.

---

## Sommaire

- [Contexte et objectifs](#contexte-et-objectifs)
- [Pipeline](#pipeline)
- [Données](#données)
- [Prétraitement](#prétraitement)
- [Modèle](#modèle)
- [Entraînement](#entraînement)
- [Résultats](#résultats)
- [Inférence en temps réel](#inférence-en-temps-réel)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Configuration](#configuration)
- [Limites et pistes d'amélioration](#limites-et-pistes-damélioration)
- [Auteurs](#auteurs)
- [Licence](#licence)

---

## Contexte et objectifs

Le point de départ est un sujet académique : construire un système capable de reconnaître différents gestes de la main pour piloter une interface sans contact. Les objectifs pédagogiques étaient les suivants :

- capturer et annoter ses propres données ;
- utiliser OpenCV pour la détection de la main et le prétraitement des images ;
- expérimenter la classification d'images avec des CNN, avec en perspective l'ajout d'une dimension temporelle (CNN + LSTM).

Ce dépôt implémente la partie statique du sujet (un geste par image) de bout en bout. La reconnaissance de gestes dynamiques est détaillée dans les [pistes d'amélioration](#limites-et-pistes-damélioration).

---

## Pipeline

```
Génération des données ──> Exploration ──> Prétraitement OpenCV ──> CNN ──> Évaluation
       (01)                   (02)              (03)              (04-05)     (06)
                                                                    │
                                                                    v
Webcam ──> Détection de la main (HSV) ──> ROI 28x28 ──> Prédiction ──> Lissage temporel ──> Affichage
                                       (07 / app.py)
```

Chaque étape correspond à un notebook numéroté. Le code de l'étape d'inférence a ensuite été extrait dans le paquet `src/` pour obtenir une application autonome lancée par `app.py`.

---

## Données

**Jeu de données principal : synthétique.** Le notebook `01_generate_data.ipynb` génère procéduralement des images 28x28 en niveaux de gris, au format de Sign Language MNIST (une colonne `label` suivie de 784 colonnes de pixels). Chaque geste est représenté par une forme géométrique simplifiée : un disque pour le poing, un rectangle pour la paume, une forme en V pour la victoire. Chaque image reçoit un bruit de fond ainsi que des variations aléatoires de position et de taille.

| Ensemble | Images | Répartition |
|---|---|---|
| Entraînement | 720 | 240 par classe |
| Validation | 180 | 20 % de l'ensemble d'origine, stratifié |
| Test | 300 | 100 par classe |

| Label | Geste |
|---|---|
| 0 | Poing fermé |
| 1 | Paume ouverte |
| 2 | Victoire (V) |

Les fichiers générés sont nommés `sign_mnist_train.csv` et `sign_mnist_test.csv` afin de rester interchangeables avec le jeu de données Sign Language MNIST.

**Expérimentations préliminaires.** Le dossier `notebooks/experiments/` contient les premiers essais menés sur le vrai [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) (27 455 images d'entraînement, 7 172 de test, 24 lettres de l'alphabet). Ces lettres ne correspondant pas directement aux trois gestes visés, le projet s'est réorienté vers un jeu dédié.

Les données ne sont pas versionnées : elles se régénèrent en quelques secondes avec le notebook 01.

---

## Prétraitement

Le même traitement est appliqué à l'entraînement (`03_preprocessing.ipynb`) et à l'inférence (`src/preprocessing/image_processing.py`), ce qui garantit la cohérence entre les deux phases :

1. **Conversion en niveaux de gris** et redimensionnement en 28x28 (interpolation `INTER_AREA`) ;
2. **CLAHE** (égalisation adaptative d'histogramme, `clipLimit=2.0`, grille 4x4) pour homogénéiser le contraste local ;
3. **Flou gaussien** 3x3 pour atténuer le bruit ;
4. **Normalisation** des pixels dans l'intervalle [0, 1] et mise en forme `(28, 28, 1)`.

---

## Modèle

CNN séquentiel compact de **256 707 paramètres**, conçu pour fonctionner en temps réel sur CPU.

| Couche | Configuration | Sortie |
|---|---|---|
| Entrée | Image en niveaux de gris | 28 x 28 x 1 |
| Conv2D + MaxPooling | 32 filtres 3x3, ReLU, pooling 2x2 | 14 x 14 x 32 |
| Conv2D + MaxPooling | 64 filtres 3x3, ReLU, pooling 2x2 | 7 x 7 x 64 |
| Conv2D | 64 filtres 3x3, ReLU | 7 x 7 x 64 |
| Flatten | | 3136 |
| Dense | 64 neurones, ReLU | 64 |
| Dropout | taux 0,5 | 64 |
| Dense | 3 neurones, Softmax | 3 |

L'architecture complète est exportée dans `models/model_architecture.json`.

---

## Entraînement

| Paramètre | Valeur |
|---|---|
| Optimiseur | Adam, taux d'apprentissage 0,001 |
| Fonction de perte | Sparse categorical cross-entropy |
| Taille de batch | 32 |
| Époques | 50 au maximum |
| Callbacks | ModelCheckpoint, EarlyStopping (patience 10), ReduceLROnPlateau (facteur 0,5, patience 5) |

**Augmentation de données** appliquée à la volée sur l'ensemble d'entraînement uniquement : rotation de plus ou moins 15 degrés, décalages horizontal et vertical de 10 %, cisaillement et zoom de 10 %. Le retournement horizontal est volontairement désactivé.

L'entraînement a été réalisé sur CPU avec TensorFlow 2.20 et Keras 3.10.

---

## Résultats

Évaluation sur les 300 images de test (`06_evaluate_model.ipynb`) :

| Métrique | Valeur |
|---|---|
| Accuracy | 100 % |
| Precision, Recall, F1 (par classe) | 1,00 |
| AUC (par classe) | 1,00 |
| Meilleure époque (validation) | 10 |

<p align="center">
  <img src="results/plots/courbes_apprentissage.png" alt="Courbes d'apprentissage" width="85%">
</p>

<p align="center">
  <img src="results/plots/matrice_confusion.png" alt="Matrice de confusion" width="60%">
</p>

**Interprétation.** Ces scores parfaits doivent être lus pour ce qu'ils sont : ils mesurent la capacité du modèle à séparer trois formes synthétiques bien distinctes, générées par le même procédé que les données d'entraînement. Ils valident la chaîne d'apprentissage, pas la robustesse face à de vraies mains filmées. En conditions réelles, la qualité de reconnaissance dépend fortement de l'éclairage, de l'arrière-plan et de la qualité de la détection de la main (voir [Limites](#limites-et-pistes-damélioration)).

---

## Inférence en temps réel

L'application (`app.py`, code dans `src/`) traite le flux de la webcam image par image :

1. **Détection de la main** par segmentation de la couleur de peau dans l'espace HSV, suivie d'opérations morphologiques (dilatation, érosion) et d'une extraction de contours. Le plus grand contour dépassant une aire minimale est retenu.
2. **Extraction d'une région carrée** centrée sur la main, avec une marge de sécurité.
3. **Prétraitement** identique à celui de l'entraînement.
4. **Prédiction** par le CNN.
5. **Lissage temporel** : moyenne des probabilités sur les 5 dernières images pour stabiliser l'affichage et éviter les changements de classe intempestifs.
6. **Affichage** de la zone détectée, de la classe prédite, de la confiance et des probabilités par classe.

Le code est découpé en modules à responsabilité unique :

| Module | Rôle |
|---|---|
| `src/detection/hand_detector.py` | Segmentation de la peau et sélection du contour de la main |
| `src/preprocessing/image_processing.py` | Extraction de la région d'intérêt et prétraitement |
| `src/model/model_loader.py` | Chargement du modèle Keras |
| `src/model/predictor.py` | Prédiction et lissage temporel |
| `src/utils/visualization.py` | Annotations superposées à l'image |
| `src/utils/config.py` | Paramètres centralisés |
| `src/app/realtime_app.py` | Boucle principale et gestion de la webcam |

---

## Structure du dépôt

```
.
├── app.py                        Point d'entrée de l'application temps réel
├── requirements.txt
├── models/
│   ├── model_architecture.json   Architecture du CNN
│   └── model_metadata.json       Métadonnées de l'entraînement
├── notebooks/
│   ├── 00_setup_project.ipynb    Création de l'arborescence
│   ├── 01_generate_data.ipynb    Génération du jeu de données
│   ├── 02_data_exploration.ipynb Analyse exploratoire
│   ├── 03_preprocessing.ipynb    Prétraitement OpenCV
│   ├── 04_build_model.ipynb      Construction du CNN
│   ├── 05_train_model.ipynb      Entraînement
│   ├── 06_evaluate_model.ipynb   Évaluation
│   ├── 07_realtime_inference.ipynb Prototype temps réel
│   └── experiments/              Essais sur Sign Language MNIST
├── results/plots/                Figures clés
└── src/                          Code modulaire de l'application
    ├── app/
    ├── detection/
    ├── model/
    ├── preprocessing/
    └── utils/
```

Les dossiers `data/` et les poids du modèle (`*.keras`, `*.h5`) sont exclus du dépôt et sont produits par les notebooks.

---

## Installation

Prérequis : Python 3.9 ou supérieur et une webcam pour la démonstration temps réel.

```bash
git clone https://github.com/ELGHAD/Hand-Gesture-Recognition.git
cd Hand-Gesture-Recognition

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Utilisation

### 1. Entraîner le modèle

Le modèle entraîné n'étant pas versionné, il faut d'abord exécuter les notebooks dans l'ordre :

```bash
jupyter notebook notebooks/
```

| Ordre | Notebook | Produit |
|---|---|---|
| 1 | `00_setup_project` | Arborescence `data/`, `models/`, `results/` |
| 2 | `01_generate_data` | `data/raw/sign_mnist_train.csv`, `sign_mnist_test.csv` |
| 3 | `02_data_exploration` | Figures d'analyse (facultatif) |
| 4 | `03_preprocessing` | `data/processed/*.npy` |
| 5 | `04_build_model` | Architecture et poids initiaux |
| 6 | `05_train_model` | `models/gesture_model_final.keras` |
| 7 | `06_evaluate_model` | Métriques et figures d'évaluation |

Chaque notebook se replace automatiquement à la racine du projet ; les chemins restent donc valides quel que soit le dossier depuis lequel Jupyter est lancé.

### 2. Lancer l'application

```bash
python app.py
```

| Touche | Action |
|---|---|
| `q` | Quitter |
| `s` | Enregistrer une capture dans `results/plots/` |

Pour de meilleurs résultats, placez la main devant un fond neutre et uniforme, avec un éclairage frontal.

---

## Configuration

Les paramètres principaux se trouvent dans `src/utils/config.py` :

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `WEBCAM_INDEX` | 0 | Indice de la caméra |
| `MIN_DETECTION_AREA` | 5000 | Aire minimale (en pixels) d'un contour de main |
| `HAND_MARGIN` | 20 | Marge autour de la main lors du recadrage |
| `PREDICTION_HISTORY_SIZE` | 5 | Nombre d'images utilisées pour le lissage |
| `CLAHE_CLIP_LIMIT` | 2.0 | Limite de contraste du CLAHE |

Les seuils HSV de détection de la peau sont définis dans `src/detection/hand_detector.py` et peuvent nécessiter un ajustement selon la caméra, l'éclairage et la carnation.

---

## Limites et pistes d'amélioration

**Limites actuelles**

- **Écart entre données synthétiques et réelles.** Le modèle n'a vu que des formes géométriques générées ; sa généralisation à de vraies mains n'est pas mesurée.
- **Détection par couleur de peau.** Sensible à l'éclairage, aux arrière-plans de teinte proche et aux variations de carnation.
- **Gestes statiques uniquement.** Chaque image est classée indépendamment ; le lissage temporel stabilise l'affichage sans modéliser le mouvement.
- **Trois classes.**

**Pistes d'amélioration**

1. Constituer un jeu de données réel annoté à partir de la webcam, conformément au sujet initial, et mesurer l'écart de performance avec le jeu synthétique.
2. Remplacer la segmentation HSV par un détecteur de points clés de la main (MediaPipe Hands), plus robuste et indépendant de la couleur de peau.
3. Introduire la dimension temporelle avec une architecture CNN + LSTM sur des séquences d'images, pour reconnaître des gestes dynamiques (balayage, rotation, etc.).
4. Étendre le nombre de classes, par exemple à l'alphabet de la langue des signes.
5. Exporter le modèle en TensorFlow Lite pour un déploiement embarqué ou mobile.

---

## Auteurs

- **Elrhadiouini Hamza** 


---

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE).
