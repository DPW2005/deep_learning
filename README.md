# Deep Learning Project Repository

Ce dépôt contient les travaux pratiques (TD) réalisés dans le cadre du cours de Deep Learning. Voici une description du contenu de chaque dossier :

## 📂 Structure du Projet

### [TD1_2](./TD1_2) - Introduction & Déploiement
Ce dossier combine les deux premiers travaux dirigés. Il se concentre sur :
- **Entraînement de modèle** : Classification d'images sur le dataset MNIST (`train_model.py`).
- **Déploiement** : Une application web Flask (`app.py`) pour servir le modèle.
- **Conteneurisation** : Un `Dockerfile` pour packager l'application.
- **Rapport** : Fichier LaTeX du rapport (`Overleaf_TD1&2.tex`).

### [TD3](./TD3) - CNN Avancés & Transfert de Style
Ce dossier explore des architectures plus complexes et des applications créatives :
- **Classification** : Implémentation de CNNs et utilisation de ResNet (`cnn_classification.py`, `resnet_mini.py`).
- **Transfert de Style** : Algorithme de transfert de style neuronal (`style_transfer.py`) utilisant des images de contenu et de style fournies.
- **Rapport** : Fichier LaTeX du rapport (`Overleaf_TD3.tex`).

### [TD4](./TD4) - Segmentation & 3D
Ce dossier aborde des tâches de vision par ordinateur plus spécialisées :
- **Segmentation d'image** : Implémentation de l'architecture U-Net (`unet_segmentation.py`).
- **Convolutions 3D** : Expérimentations avec des convolutions 3D (`conv3d_experiment.py`), souvent utilisées pour l'analyse vidéo ou volumétrique.
- **Rapport** : Fichier LaTeX du rapport (`Overleaf_TD4.tex`).

## 🛠 Installation

Les dépendances nécessaires sont listées dans le fichier `requirements.txt`.

### Clone the repository

```bash
git clone https://github.com/DPW2005/deep_learning
cd deep_learning
```

### Create and activate a virtual environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---