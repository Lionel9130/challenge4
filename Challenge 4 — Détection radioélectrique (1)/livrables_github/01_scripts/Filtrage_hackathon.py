"""
Nettoyage du fichier ANFR SUP_SUPPORT.txt -> extraction des sites exploitables d'un département.

Étapes effectuées par la fonction :
1. Dédoublonnage sur SUP_ID (un support physique peut porter plusieurs installations)
2. Conversion des coordonnées DMS -> degrés décimaux
3. Rejet des lignes de coordonnées invalides c_a_d secondes/minutes >= 60)
4. Filtrage sur le département demandé (code postal)
5. Retrait des sites invisibles du ciel
6. Fusion des vrais doublons (même coordonnées ET même lieu-dit)
7. Écriture du fichier final
"""

import pandas as pd
import unicodedata
from telecharger_anfr import telecharger_donnees_anfr


# Noms de colonnes du format ANFR (fixes, ne dépendent pas du département)

COL_ID = "SUP_ID"

COL_LAT_DEG, COL_LAT_MIN, COL_LAT_SEC, COL_LAT_NS = (
    "COR_NB_DG_LAT", "COR_NB_MN_LAT", "COR_NB_SC_LAT", "COR_CD_NS_LAT",
)
COL_LON_DEG, COL_LON_MIN, COL_LON_SEC, COL_LON_EW = (
    "COR_NB_DG_LON", "COR_NB_MN_LON", "COR_NB_SC_LON", "COR_CD_EW_LON",
)

COL_HAUTEUR = "SUP_NM_HAUT"
COL_NATURE = "NAT_ID"
COL_CODE_POSTAL = "ADR_NM_CP"
COL_LIEU = "ADR_LB_LIEU"
COL_ADD1 = "ADR_LB_ADD1"


MOTS_CLES_INVISIBLES = ["SOUTERRAIN", "SOUS-TERRAIN", "SOUS TERRAIN", "TUNNEL", "GALERIE"]



# Mes fonctions

def charger_table_nature(path):
    #pour avoir les ID de toute les natures de suport qui existe dans la BD"

    table = pd.read_csv(path, sep=";", encoding="utf-8")
    table.columns = [c.strip() for c in table.columns]
    labels = dict(zip(table["NAT_ID"], table["NAT_LB_NOM"]))

    codes_invisibles = set()
    for nat_id, libelle in labels.items():
        libelle_maj = str(libelle).upper()
        if any(mot in libelle_maj for mot in MOTS_CLES_INVISIBLES):
            codes_invisibles.add(nat_id)

    return labels, codes_invisibles


def dms_to_decimal(deg, minute, sec, hemi, hemi_neg):
    #Convertit degrés/minutes/secondes -> degrés décimaux. Renvoie  erreur "NaN" si invalide.
    try:
        deg, minute, sec = float(deg), float(minute), float(sec)
    except (TypeError, ValueError):
        return float("nan")
    if sec >= 60 or minute >= 60 or deg < 0:
        return float("nan")
    dd = deg + minute / 60 + sec / 3600
    if str(hemi).strip().upper() == hemi_neg:
        dd = -dd
    return dd


def normalize_text(s):
    #Majuscules + je retire des accents, pour comparer deux écriture.
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def lieu_tokens(row):
    """Mots significatifs (>=4 lettres) du lieu-dit + adresse, pour détecter les vrais doublons."""
    txt = normalize_text(row[COL_LIEU]) + " " + normalize_text(row[COL_ADD1])
    return set(w for w in txt.split() if len(w) >= 4)



# Fonction principale


def nettoyer_anfr(fichier_support, fichier_nature, departement, fichier_sortie, verbose=True):
    def log(msg):
        if verbose:
            print(msg)

    nat_id_labels, codes_nature_invisibles = charger_table_nature(fichier_nature)
    log(f"Table NAT_ID chargée : {len(nat_id_labels)} codes, "
        f"{len(codes_nature_invisibles)} considérés invisibles du ciel "
        f"{sorted(codes_nature_invisibles)}")

    df = pd.read_csv(fichier_support, sep=";", encoding="utf-8", low_memory=False)
    log(f"Lignes lues : {len(df)}")

    #  1. Dédoublonnage sur SUP_ID
    n_avant = len(df)
    df = df.drop_duplicates(subset=[COL_ID])
    n_apres = len(df)
    log(f"Après dédoublonnage SUP_ID : {n_apres} sites (ratio {n_avant / n_apres:.2f}, attendu ~2)")

    #  2 & 3. Conversion DMS -> décimal, je rejete des lignes bizarres
    df["latitude"] = df.apply(
        lambda r: dms_to_decimal(r[COL_LAT_DEG], r[COL_LAT_MIN], r[COL_LAT_SEC], r[COL_LAT_NS], "S"),
        axis=1,
    )
    df["longitude"] = df.apply(
        lambda r: dms_to_decimal(r[COL_LON_DEG], r[COL_LON_MIN], r[COL_LON_SEC], r[COL_LON_EW], "W"),
        axis=1,
    )
    n_avant_filtre = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    log(f"Lignes de coordonnées invalides jetées : {n_avant_filtre - len(df)}")

    #  4. Filtrage sur le département (ou pas, si departement est None)
    if departement:
        cp = df[COL_CODE_POSTAL].astype(str).str.zfill(5)
        df_dept = df[cp.str.startswith(str(departement))].copy()
        log(f"Sites dans le département {departement} : {len(df_dept)}")
    else:
        df_dept = df.copy()
        log(f"Aucun filtre département -> toute la France : {len(df_dept)} sites")

    # 5. Retrait des sites invisibles du ciel
    nat_id_num = pd.to_numeric(df_dept[COL_NATURE], errors="coerce")
    df_dept = df_dept[~nat_id_num.isin(codes_nature_invisibles)].copy()
    df_dept["nature_libelle"] = pd.to_numeric(df_dept[COL_NATURE], errors="coerce").map(nat_id_labels)
    log(f"Après retrait souterrain/tunnel/galerie : {len(df_dept)}")

    #  6. Fusion des vrais doublons (même coordonnées ET même lieu-dit)
    coord_cols = [COL_LAT_DEG, COL_LAT_MIN, COL_LAT_SEC, COL_LAT_NS,
                  COL_LON_DEG, COL_LON_MIN, COL_LON_SEC, COL_LON_EW]
    df_dept["coord_key"] = df_dept[coord_cols].astype(str).agg("_".join, axis=1)
    df_dept["tokens"] = df_dept.apply(lieu_tokens, axis=1)

    sizes = df_dept.groupby("coord_key")["coord_key"].transform("size")
    a_supprimer = []
    for _, groupe in df_dept[sizes > 1].groupby("coord_key"):
        idxs = list(groupe.index)
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                mots_communs = df_dept.loc[idxs[i], "tokens"] & df_dept.loc[idxs[j], "tokens"]
                if mots_communs:  # même lieu-dit -> vrai doublon, je garde un seul
                    a_supprimer.append(idxs[j])

    df_dept = df_dept.drop(index=a_supprimer)
    log(f"Doublons de lieu fusionnés : {len(a_supprimer)}")
    log(f"TOTAL sites exploitables (département {departement}) : {len(df_dept)}")

    #  7. Écriture du fichier final
    df_final = df_dept[[COL_ID, "latitude", "longitude", "nature_libelle", COL_HAUTEUR]].rename(
        columns={COL_ID: "id", "nature_libelle": "type_support", COL_HAUTEUR: "hauteur"}
    )
    df_final.to_csv(fichier_sortie, index=False)
    log(f"Fichier écrit : {fichier_sortie} ({len(df_final)} lignes)")

    return df_final



# main

if __name__ == "__main__":
    departement="32"
    dossier_dest = r"C:\Users\Administrator\Downloads\hackathon-20261\ANFR_data"

    dossier_data_finale = r"C:\Users\Administrator\Downloads\hackathon-20261"

    fichier_support, fichier_nature = telecharger_donnees_anfr(dossier_dest=dossier_dest)

    nettoyer_anfr(
        fichier_support=fichier_support,
        fichier_nature=fichier_nature,
        departement=departement,
        fichier_sortie=rf"{dossier_data_finale}\sites_{departement}.csv",
)