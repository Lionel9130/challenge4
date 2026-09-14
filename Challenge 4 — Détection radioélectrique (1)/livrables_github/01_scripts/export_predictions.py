"""Export des prédictions du modèle en JSON pour la carte de l'artefact.

À lancer sur le SSP Cloud (JupyterLab) APRÈS l'entraînement.
Usage :
    python export_predictions.py

Il faut adapter les 3 chemins de la section CONFIG ci-dessous.
Sortie : predictions_gers.json (quelques Ko), à déposer dans data/ du projet artefact.
"""
import json
import re
from pathlib import Path

from ultralytics import YOLO

# ------------------------- CONFIG (à adapter) -------------------------
POIDS = Path("best.pt")
# split -> (dossier images, dossier labels)
SPLITS = {
    "train": (Path("train"), Path("labels/train")),
    "validation": (Path("val"), Path("labels/val")),
    "test": (Path("test"), Path("labels/test")),
}
SEUIL_CONF = 0.25
SORTIE = Path("predictions_gers.json")
# ----------------------------------------------------------------------

CLASSES = {0: "pylone", 1: "mat", 2: "batiment", 3: "chateau_eau", 4: "silo"}


def id_anfr(nom_fichier: str) -> str:
    """site_1234.jpg ou ANFR_1234_x0_y640.jpg -> 1234"""
    m = re.search(r"(?:site|ANFR)_(\d+)", nom_fichier)
    return m.group(1) if m else Path(nom_fichier).stem


def a_verite(stem: str, labels_dir: Path) -> bool:
    """Vrai si l'annotation contient au moins une boîte (support visible)."""
    txt = labels_dir / f"{stem}.txt"
    return txt.exists() and txt.read_text(encoding="utf-8").strip() != ""


def main():
    model = YOLO(str(POIDS))
    resultats = []
    for split, (img_dir, lbl_dir) in SPLITS.items():
        if not img_dir.exists():
            print(f"(!) dossier absent, split sauté : {img_dir}")
            continue
        n_img = sum(1 for p in img_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"{split} : {n_img} images")
        for r in model.predict(source=str(img_dir), conf=SEUIL_CONF,
                               imgsz=640, stream=True, verbose=False):
            p = Path(r.path)
            boxes = r.boxes
            detections = []
            if boxes is not None:
                for b in boxes:
                    x, y, w, h = [round(float(v), 4) for v in b.xywhn[0].tolist()]
                    detections.append({
                        "classe": CLASSES.get(int(b.cls.item()), "autre"),
                        "conf": round(float(b.conf.item()), 3),
                        "box": [x, y, w, h],
                    })
            verite = a_verite(p.stem, lbl_dir)
            detecte = len(detections) > 0
            statut = ("ok" if verite and detecte else
                      "rate" if verite else
                      "fausse" if detecte else "vide")
            resultats.append({
                "id": id_anfr(p.name),
                "fichier": p.name,
                "split": split,
                "statut": statut,
                "detections": detections,
            })

    SORTIE.write_text(json.dumps(resultats, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    n = {"ok": 0, "rate": 0, "fausse": 0, "vide": 0}
    for x in resultats:
        n[x["statut"]] += 1
    print(f"Écrit {SORTIE} : {n['ok']} trouvés, {n['rate']} ratés, "
          f"{n['fausse']} fausses alertes, {n['vide']} vides")


if __name__ == "__main__":
    main()
