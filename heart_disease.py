# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input

# 1. Chargement et exploration du dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
df = pd.read_csv(url, names=column_names, na_values='?')

# Afficher les premières lignes du dataset
print("Premières lignes du dataset :")
print(df.head())

# Informations générales sur le dataset
print("\nInformations sur le dataset :")
print(df.info())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())

# Vérification des valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# 2. Nettoyage des données
# Remplacer les valeurs manquantes par la médiane (pour les colonnes numériques)

df.loc[:, 'ca'] = df['ca'].fillna(df['ca'].median())
df.loc[:, 'thal'] = df['thal'].fillna(df['thal'].median())

# Encodage des variables catégorielles (si nécessaire)
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'ca', 'thal'], drop_first=True)

# Convertir la cible en binaire (0 = pas de maladie, 1 = maladie)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# 3. Normalisation des données
scaler = StandardScaler()
X = df.drop(columns=['target'])
y = df['target']
X_scaled = scaler.fit_transform(X)

# 4. Division des données en ensembles d'entraînement, de validation et de test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

print(f"\nDimensions des ensembles :")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Utiliser Input() pour la forme d'entrée
    Dense(64, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dense(1, activation='sigmoid')  # Couche de sortie (classification binaire)
])
# Compilation du modèle avec Adam et learning rate personnalisé
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Affichage de l'architecture du modèle
print("\nRésumé du modèle :")
model.summary()

# 6. Entraînement du modèle avec early stopping et réduction du learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 7. Évaluation du modèle
# Évolution de la perte et de l'accuracy
plt.figure(figsize=(12, 5))

# Graphique de la perte
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Évolution de la perte')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Graphique de l'accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Évolution de l\'accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 8. Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)  # Conversion en classes binaires

# 9. Métriques de performance
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

print(f"\nMétriques de performance sur l'ensemble de test :")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# 10. Matrice de confusion normalisée
conf_matrix = confusion_matrix(y_test, y_pred_class)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Pas de maladie', 'Maladie'], yticklabels=['Pas de maladie', 'Maladie'])
plt.title('Matrice de confusion normalisée')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

# 11. Courbe ROC et AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.show()

# 12. Enregistrement du modèle
model.save('heart_disease_model_improved.h5')
print("\nModèle enregistré sous 'heart_disease_model_improved.h5'.")