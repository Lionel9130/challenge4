"""
Téléchargement automatique des fichiers ANFR à jour (data.gouv.fr).

Ce script est indépendant du script de nettoyage : il sert uniquement à récupérer
la dernière version de SUP_SUPPORT.txt et SUP_NATURE.txt, et les place dans un
dossier local. Le script de nettoyage peut ensuite pointer directement dessus.

"""

import requests #me permet de lire un lien sur internet
import zipfile
import io
import os

# Données ANFR sur data.gouv.fr
DATASET_SLUG = "donnees-sur-les-installations-radioelectriques-de-plus-de-5-watts-1"
API_DATASET_URL = f"https://www.data.gouv.fr/api/1/datasets/{DATASET_SLUG}/"


def chercher_fichier(dossier, nom_fichier):
    #Recherche nom_fichier (insensible à la casse) récursivement dans les différents dossiers de la ANFR.
    for racine, _, fichiers in os.walk(dossier):
        for f in fichiers:
            if f.lower() == nom_fichier.lower():
                return os.path.join(racine, f)
    return None


def telecharger_donnees_anfr(dossier_dest, forcer=False):
    os.makedirs(dossier_dest, exist_ok=True)

    chemin_support_existant = chercher_fichier(dossier_dest, "SUP_SUPPORT.txt")
    chemin_nature_existant = chercher_fichier(dossier_dest, "SUP_NATURE.txt")
    if not forcer and chemin_support_existant and chemin_nature_existant:
        print("Fichiers déjà présents localement, pas de nouveau téléchargement "
              "(mets forcer=True pour forcer une mise à jour).")
        return chemin_support_existant, chemin_nature_existant

    print(f"Interrogation de l'API : {API_DATASET_URL}")
    reponse = requests.get(API_DATASET_URL, timeout=30)
    reponse.raise_for_status()
    infos = reponse.json()

    ressources = infos.get("resources", [])
    zips = [r for r in ressources if r.get("format", "").lower() == "zip"]
    if not zips:
        raise RuntimeError("Aucune ressource au format zip trouvée pour ce jeu de données.")

    def plus_recent(liste):
        return sorted(liste, key=lambda r: r.get("last_modified", ""), reverse=True)[0] if liste else None

    # on distingue les 2 zips : "référence" (tables de correspondance) vs le reste (données)
    zips_reference = [r for r in zips if "ref" in r.get("title", "").lower() or "-ref.zip" in r.get("url", "").lower()]
    zips_donnees = [r for r in zips if r not in zips_reference]

    ressource_donnees = plus_recent(zips_donnees)
    ressource_reference = plus_recent(zips_reference)

    if not ressource_donnees or not ressource_reference:
        raise RuntimeError(
            "Impossible de distinguer un zip \u00ab données \u00bb et un zip \u00ab références \u00bb "
            f"parmi les ressources trouvées : {[r.get('title') for r in zips]}"
        )

    for ressource in (ressource_donnees, ressource_reference):
        print(f"Ressource retenue : {ressource.get('title')} "
              f"(mise à jour le {ressource.get('last_modified')})")
        url_zip = ressource["url"]
        print(f"Téléchargement depuis : {url_zip}")
        reponse_zip = requests.get(url_zip, timeout=180)
        reponse_zip.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(reponse_zip.content)) as archive:
            archive.extractall(dossier_dest)
        print(f"Archive extraite dans : {dossier_dest}")

    chemin_support = chercher_fichier(dossier_dest, "SUP_SUPPORT.txt")
    chemin_nature = chercher_fichier(dossier_dest, "SUP_NATURE.txt")
    if not chemin_support or not chemin_nature:
        raise FileNotFoundError(
            "SUP_SUPPORT.txt et/ou SUP_NATURE.txt introuvables après extraction. "
            "Vérifie le contenu de dossier_dest : la structure de l'archive a peut-être changé."
        )
    return chemin_support, chemin_nature


if __name__ == "__main__":
    DOSSIER_DONNEES = r"C:\Users\Administrator\Downloads\hackathon-20261\ANFR_data"  # mettre le dossier destination

    chemin_support, chemin_nature = telecharger_donnees_anfr(DOSSIER_DONNEES)
    print()
    print("Fichiers de Damoclèsby01 prêts à l'emploi :")
    print("  SUP_SUPPORT.txt ->", chemin_support)
    print("  SUP_NATURE.txt  ->", chemin_nature)