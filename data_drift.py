import pandas as pd
import lightgbm as lgb  # Importation de lightgbm
from evidently.dashboard import Dashboard
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
import numpy as np 
import time as time 



# Charger les données en supprimant l'index
df = pd.read_csv('df.csv', sep=",")

# Diviser les données en ensembles d'entraînement et de test
application_train = df.dropna(subset=['TARGET'])
application_test = df[df['TARGET'].isna()]

# Créer un modèle LightGBM et l'entraîner sur les données d'entraînement
params = {
    'learning_rate': 0.1,
    'n_estimators': 70
}

# Créer un dataset LightGBM en utilisant une seule colonne comme étiquette
lgb_train = lgb.Dataset(application_train.drop(columns=['TARGET']), label=application_train['TARGET'])

# Entraîner le modèle LightGBM
model = lgb.train(params, lgb_train)

# Obtenir l'importance des caractéristiques
importances = model.feature_importance(importance_type='gain')

# Créer un DataFrame avec les importances des caractéristiques
feature_importances = pd.DataFrame(importances, index=application_train.drop(columns=['TARGET']).columns, columns=['Score'])

# Trier les caractéristiques par score
feature_importances = feature_importances.sort_values(by='Score')

# Sélectionner les 15 premières caractéristiques
top_features = feature_importances.tail(15)

# Filtrer les ensembles de données pour ne conserver que les 15 caractéristiques les plus importantes
top_features_names = top_features.index.tolist()
application_train_filtered = application_train[top_features_names]
application_test_filtered = application_test[top_features_names]


# Définir le temps de départ
start_time = time.time()

# Créer le rapport de dérive des données
data_drift_report = Report(metrics=[
    DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.2, cat_stattest_threshold=0.2),
])

print("Création du data_drift_report")

data_drift_report.run(reference_data=application_train_filtered, current_data=application_test_filtered, column_mapping=None)

print("Run du data_drift_report")

elapsed_time_fit = time.time() - start_time
print(elapsed_time_fit)

# Sauvegardez le rapport en tant que fichier HTML
data_drift_report.save_html('data_drift_report.html')
