import json
import os
import s3fs
import zarr

default_path = json.load(open("../paths.json"))

def get_path(key):
    return os.environ.get(key, default_path[key])


def resolve_path(full_path: str, require_local: bool = False) -> str:
    """
    Vérifie l'emplacement du fichier/dossier. 
    Si require_local=True et que le fichier est sur S3, le rapatrie dans /tmp/.
    Renvoie le chemin final prêt à être lu par votre code métier.
    """
    is_s3 = full_path.startswith("s3://")

    # Cas 1 : C'est déjà sur votre PC, OU c'est sur S3 mais la librairie gère le streaming (Zarr/Xarray)
    if not is_s3 or not require_local:
        return full_path

    # Cas 2 : C'est sur S3 ET la librairie exige un fichier physique (Keras, Pickle, open)
    print(f"[Infrastructure] Rapatriement local requis pour : {full_path}")
    
    # On crée un nom de fichier plat pour éviter les conflits dans /tmp/
    # Ex: s3://bucket/dossier/modele.keras -> /tmp/bucket_dossier_modele.keras
    safe_filename = full_path.replace("s3://", "").replace("/", "_")
    local_tmp_path = os.path.join("/tmp", safe_filename)

    # On ne télécharge que si on ne l'a pas déjà fait
    if not os.path.exists(local_tmp_path):
        fs = s3fs.S3FileSystem()
        # L'argument recursive=True permet de télécharger aussi bien un fichier unique qu'un dossier entier
        fs.get(full_path, local_tmp_path, recursive=True)
        print(f"[Infrastructure] Téléchargement terminé -> {local_tmp_path}")

    return local_tmp_path

def open_zarr_hybrid(chemin_complet: str):
    """Ouvre un Zarr consolidé, qu'il soit sur S3 ou sur PC"""
    
    # 1. On passe par notre douane (sans forcer le téléchargement)
    chemin_resolu = resolve_path(chemin_complet, require_local=False)
    
    # 2. Ouverture Cloud-Native
    if chemin_resolu.startswith("s3://"):
        fs = s3fs.S3FileSystem()
        # S3Map transforme l'URL S3 en un dictionnaire virtuel que Zarr sait lire
        store = s3fs.S3Map(root=chemin_resolu, s3=fs)
        return zarr.open_consolidated(store)
        
    # 3. Ouverture Locale classique
    else:
        return zarr.open_consolidated(chemin_resolu)