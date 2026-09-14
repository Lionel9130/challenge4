import os
import math
import io
import unicodedata
import pandas as pd
import requests
from PIL import Image  # pip install pillow

# Configuration
FICHIER_CSV = r"C:\Users\Administrator\Downloads\hackathon-20261\sites_32.csv"
DOSSIER_DATASET = r"C:\Users\Administrator\Downloads\hackathon-20261\dataset_yolo"
TAILLE_IMAGE = 640   # Résolution standard pour YOLO
DELTA_DEG = 0.0005   # Environ 55 m de rayon autour du point central (en latitude)
QUALITE_JPEG = 95    # le serveur IGN ne permet pas de choisir la qualité -> on télécharge en PNG (sans perte) et on compresse nous-mêmes en JPEG 95
NB_TEST = 0          # mode test rapide : 5 = ne traite que les 5 premiers sites ; 0 = tout le fichier

# Création de l'arborescence YOLO
os.makedirs(os.path.join(DOSSIER_DATASET, "images"), exist_ok=True)
os.makedirs(os.path.join(DOSSIER_DATASET, "labels"), exist_ok=True)

# Dictionnaire pour l'approche multi-classes (Niveau 2)
# Vous pouvez l'enrichir selon les valeurs exactes de votre colonne 'type_support'
CLASSES = {
    "PYLONE": 0,
    "CHATEAU D'EAU": 1,
    "BATIMENT": 2,
    "MAT": 3
}

def sans_accents(s):
    # "Pylône" -> "PYLONE" : sans ça, aucune clé de CLASSES ne matche jamais (Ô != O) et tout tombe en "Autre"
    s = str(s).upper().strip()
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

def get_class_id(nature_libelle):
    if pd.isna(nature_libelle):
        return 4  # Autre
    nature_upper = sans_accents(nature_libelle)
    for cle, id_classe in CLASSES.items():
        if cle in nature_upper:
            return id_classe
    return 4  # Autre par défaut

print("Lecture du fichier CSV...")
df = pd.read_csv(FICHIER_CSV)
if NB_TEST:
    df = df.head(NB_TEST)
    print(f"MODE TEST : seulement les {NB_TEST} premiers sites")

nb_ok, nb_skip, nb_err = 0, 0, 0
for index, row in df.iterrows():
    site_id = row['id']
    lat, lon = row['latitude'], row['longitude']

    image_path = os.path.join(DOSSIER_DATASET, "images", f"site_{site_id}.jpg")
    label_path = os.path.join(DOSSIER_DATASET, "labels", f"site_{site_id}.txt")
    if os.path.exists(image_path) and os.path.exists(label_path):
        nb_skip += 1
        continue  # reprise : on ne retélécharge pas ce qui existe déjà

    # 1. Calcul de l'emprise géographique (Bounding Box pour l'API WMS IGN)
    # En longitude, un degré est plus court qu'en latitude (facteur cos(lat)) :
    # sans cette correction l'emprise n'est pas carrée et l'image est étirée de ~27 %.
    delta_lon = DELTA_DEG / math.cos(math.radians(lat))
    min_lat, max_lat = lat - DELTA_DEG, lat + DELTA_DEG
    min_lon, max_lon = lon - delta_lon, lon + delta_lon

    # 2. Requête API IGN BD ORTHO (Nouvelle Géoplateforme) — en PNG (sans perte)
    wms_url = (
        f"https://data.geopf.fr/wms-r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS=HR.ORTHOIMAGERY.ORTHOPHOTOS&STYLES=&CRS=EPSG:4326"
        f"&BBOX={min_lat},{min_lon},{max_lat},{max_lon}"
        f"&WIDTH={TAILLE_IMAGE}&HEIGHT={TAILLE_IMAGE}&FORMAT=image/png"
    )

    try:
        # Téléchargement PNG puis compression locale en JPEG qualité 95
        # (meilleure préservation des structures fines que le JPEG du serveur ; le PNG ne touche jamais le disque)
        reponse = requests.get(wms_url, timeout=15)
        if reponse.status_code == 200:
            Image.open(io.BytesIO(reponse.content)).convert("RGB").save(image_path, "JPEG", quality=QUALITE_JPEG)

            # 3. Création de l'annotation faible YOLO
            # Format: <id_classe> <x_centre> <y_centre> <largeur> <hauteur> (Valeurs normalisées 0-1)
            id_classe = get_class_id(row['type_support'])
            x_centre = 0.5
            y_centre = 0.5
            largeur = 0.2  # On suppose que la structure occupe 20% de l'image
            hauteur = 0.2

            with open(label_path, 'w') as f:
                f.write(f"{id_classe} {x_centre} {y_centre} {largeur} {hauteur}\n")

            nb_ok += 1
            print(f"OK : {site_id} (Classe {id_classe})")
        else:
            nb_err += 1
            print(f"Erreur API IGN pour {site_id} : {reponse.status_code}")

    except Exception as e:
        nb_err += 1
        print(f"Échec pour {site_id} : {e}")

print(f"Génération terminée : {nb_ok} téléchargés, {nb_skip} déjà présents, {nb_err} erreurs. Dossier prêt pour l'entraînement YOLO.")
