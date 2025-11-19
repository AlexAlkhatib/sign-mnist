# ✋ CNN Sign Language Alphabet Classifier

**Reconnaissance des lettres de l’alphabet en langue des signes (ASL) à partir d’images avec Réseaux de Neurones Convolutifs**


## 📌 Description du Projet

Ce projet consiste à développer un **modèle CNN de Deep Learning** capable de reconnaître les lettres de l’alphabet de la langue des signes américaine (ASL).

Il s’appuie sur le dataset **Sign Language MNIST** (format 28×28 pixels), et inclut :

* Prétraitement avancé des données
* Rééquilibrage des classes (SMOTE)
* Construction d’un CNN efficace
* Recherche d’hyperparamètres (GridSearchCV + SciKeras)
* Augmentation d’images
* Comparaison avec un modèle pré-entraîné (VGG16)
* Test sur des images capturées manuellement


## 📂 Dataset

Dataset : **Sign Language MNIST**
Format : 28×28 pixels en niveaux de gris
Classes : 24 lettres (J et Z exclues)

### 📄 Colonnes

* `label` : numéro associé à la lettre
* `pixel1` → `pixel784` : intensité des pixels


## 🧹 Prétraitement des Données

### ✔️ Nettoyage & Structuration

* Reshape en format image 28×28×1
* Normalisation des pixels `0–1`
* Split train / validation / test

### ✔️ Rééquilibrage avec SMOTE

Les classes étaient légèrement déséquilibrées.
Application de **SMOTE** :

```python
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
```


## 🧠 Modèle CNN

### 🏗️ Architecture

* 3 blocs **Conv2D + MaxPooling**
* **Dropout** pour limiter l’overfitting
* Dense(128) + Dense(64)
* Sortie : Softmax(25)

### 📈 Résultats

**Accuracy Test (sans augmentation)** : `97.12%`
**Accuracy Test (avec augmentation)** : `97.65%` ✔️ *meilleure version*


## 🔍 Recherche d’Hyperparamètres

Utilisation de **GridSearchCV + SciKeras** :

Paramètres testés :

* `activation`: *relu, sigmoid*
* `optimizer`: *adam, rmsprop*
* `nb_units`: *64, 128*

Meilleurs paramètres retenus :

```python
{'activation': 'relu', 'optimizer': 'adam', 'nb_units': 128}
```

⚠️ Score CV faible (≈0.36) car GridSearch n’entraîne qu’un epoch → modèle final plus performant.


## 🏋️ Augmentation d’Images

Techniques utilisées :

* Flip horizontal
* Rotation légère
* Zoom
* Translation

→ Amélioration mesurable du score de test.


## 🔍 Test sur Images Réelles

Utilisation d’images personnelles (28×28, grayscale).
Pipeline :

* Chargement
* Redimensionnement
* Normalisation
* Prédiction avec `best_model.keras`

Exemple :

```python
predictions = model.predict(sign_images)
```


## 🧪 Modèle Pré-Entraîné : VGG16

### Pourquoi ?

Comparer un CNN simple à un modèle très profond.

### Adaptations

* Convertir images grayscale → RGB
* Redimensionner 28×28 → 32×32
* Geler les poids du backbone
* Ajouter couches Dense

### Résultats

**Accuracy Test VGG16** : `93.48%`
→ Moins performant que le CNN sur-mesure (dataset simple + petite résolution).


## 🚀 Technologies Utilisées

* Python 3.x
* TensorFlow / Keras
* SciKit-Learn
* SciKeras
* Pandas / NumPy
* Matplotlib / Seaborn


## ▶️ Lancer le Projet

### 1. Cloner le dépôt

```bash
git clone https://github.com/username/sign-language-cnn.git
cd sign-language-cnn
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Entraîner le modèle

```bash
python train_cnn.py
```

### 4. Tester une image

```bash
python predict.py --image ./images/A.png
```


## 📊 Améliorations Possibles

* Ajout de J et Z avec modèles séquentiels (RNN + CNN)
* Utilisation de modèles modernes (EfficientNet, MobileNetV3)
* Portage sur mobile (TensorFlow Lite)
* Interface Streamlit pour tester les signes en direct
* Mode "vidéo" avec reconnaissance continue


## 👤 Auteur

**Alex Alkhatib**
Projet Deep Learning — Reconnaissance ASL


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib
