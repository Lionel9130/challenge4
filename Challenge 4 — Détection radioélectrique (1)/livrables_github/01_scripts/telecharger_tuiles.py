"""Téléchargement des tuiles IGN autour de chaque support ANFR — jalon C, Challenge 4.

Consomme les CSV CORRIGÉS (longitudes ouest signées, cf. Note - Bug longitude ouest) :
    python telecharger_tuiles.py sites_32.csv tuiles/

Pour chaque site : calcule l'emprise carrée (320 m par défaut, 640 m si --large),
demande l'orthophoto à la Géoplateforme IGN (service WMS public, sans clé),
sauvegarde tuiles/SUP-<id>.png + un tuiles/index.csv (id, lat, lon, emprise, px).

Résolution native BD ORTHO : 20 cm/pixel -> 320 m = 1600 px. Ne jamais rétrécir
ensuite : découper en 640x640 avec recouvrement (starter du challenge).

Garde-fou hérité du bug longitude : le script REFUSE un CSV sans aucune longitude
négative (sauf --sans-controle, p.ex. pour un département entièrement à l'est).
"""
import csv, math, sys, time, urllib.request, urllib.parse
from pathlib import Path

WMS = "https://data.geopf.fr/wms-r"          # Géoplateforme IGN, accès libre
COUCHE = "ORTHOIMAGERY.ORTHOPHOTOS"          # BD ORTHO la plus récente
RES_M = 0.20                                  # 20 cm/pixel (natif)
M_PAR_DEG_LAT = 111_320.0

def emprise(lat, lon, cote_m):
    """Carré de cote_m mètres centré sur (lat, lon) -> (lon_min, lat_min, lon_max, lat_max)."""
    demi = cote_m / 2.0
    dlat = demi / M_PAR_DEG_LAT
    dlon = demi / (M_PAR_DEG_LAT * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

def url_wms(bbox, px):
    q = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": COUCHE, "STYLES": "", "CRS": "CRS:84",   # CRS:84 = lon,lat en degrés
        "BBOX": ",".join(f"{v:.8f}" for v in bbox),
        "WIDTH": str(px), "HEIGHT": str(px), "FORMAT": "image/png",
    }
    return WMS + "?" + urllib.parse.urlencode(q)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 2:
        sys.exit("usage: python telecharger_tuiles.py <sites.csv> <dossier_sortie> [--large] [--sans-controle]")
    src, out = Path(args[0]), Path(args[1])
    cote = 640 if "--large" in opts else 320   # --large : zones bâties (plus de négatifs de voisinage)
    px = round(cote / RES_M)                   # 1600 ou 3200

    with open(src, newline="", encoding="utf-8") as f:
        sites = [(r["id"], float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(f)]

    # Garde-fou anti-bug longitude : un jeu national/ouest doit contenir des négatives.
    if "--sans-controle" not in opts and not any(lon < 0 for _, _, lon in sites):
        sys.exit("ERREUR : aucune longitude négative dans le CSV — signe ouest non appliqué ? "
                 "(cf. Note - Bug longitude ouest ; --sans-controle pour forcer)")

    out.mkdir(parents=True, exist_ok=True)
    index = [("id", "latitude", "longitude", "cote_m", "px", "fichier")]
    for i, (sid, lat, lon) in enumerate(sites, 1):
        dest = out / f"SUP-{sid}.png"
        if not dest.exists():                  # reprise après interruption
            urllib.request.urlretrieve(url_wms(emprise(lat, lon, cote), px), dest)
            time.sleep(0.2)                    # courtoisie envers le service public
        index.append((sid, lat, lon, cote, px, dest.name))
        if i % 25 == 0 or i == len(sites):
            print(f"{i}/{len(sites)}  {dest.name}")

    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(index)
    print(f"Terminé : {len(sites)} tuiles ({cote} m, {px}x{px} px) dans {out}/")

if __name__ == "__main__":
    main()
