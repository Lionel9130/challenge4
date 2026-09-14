# Challenge 4 — livrables artefact & scripts (Niveau 1-2)

Complément au dépôt `challenge_4 - DATA` : les scripts partagés et les données de l'artefact cartographique.

## 01_scripts
- `Filtrage_hackathon.py` : référentiel ANFR -> sites_France.csv / sites_32.csv (bug de signe des longitudes ouest corrigé : 17 723 sites)
- `telecharger_anfr.py` : téléchargement de l'open data ANFR
- `telecharger_tuiles.py` : tuiles IGN 320 m à 20 cm/pixel centrées sur chaque site (JPEG 95)
- `annotation_sites.py` : annotations initiales par coordonnée (accents et projection corrigés)
- `export_predictions.py` : inférence sur les tuiles (SSP Cloud) -> predictions_gers.json pour la carte
- `preparation_donnees.js` : préparation des JSON de l'artefact

## 02_data
- `gers_dataset.json` : 437 sites du Gers, splits 75/15/10
- `predictions_gers.json` : verdict du modèle sur 436 tuiles (313 trouvés, 40 ratés, 14 fausses alertes, 69 annotations vides)

L'artefact cartographique et la présentation sont consultables dans le projet de conception.
