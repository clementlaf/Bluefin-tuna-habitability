import json
import os
import s3fs
import zarr

default_path = json.load(open("../paths.json"))

def get_path(key):
    # Entrée EDITO "S3_OUTPUT_FOLDER" au format "bucket_name/folder_name" (sans s3://) :
    # chaque utilisateur y indique son propre bucket (ex. "oidc-<login>/bluefin").
    if key == "OUTPUT_PATH" and "OUTPUT_PATH" not in os.environ:
        folder = os.environ.get("S3_OUTPUT_FOLDER", "").strip().strip("/")
        if folder:
            return folder if folder.startswith("s3://") else f"s3://{folder}"
    return os.environ.get(key, default_path[key])


_fs = None

def get_s3fs() -> s3fs.S3FileSystem:
    """
    Renvoie un S3FileSystem configuré pour l'endpoint S3 (ex. MinIO d'EDITO).
    EDITO fournit AWS_S3_ENDPOINT (sans schéma, ex. "minio.dive.edito.eu"),
    variable que s3fs/botocore ne lisent pas : sans cela, s3fs interroge AWS.
    Les identifiants (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)
    sont lus automatiquement depuis l'environnement.
    """
    global _fs
    if _fs is None:
        kwargs = {}
        # S3_ENDPOINT (URL complète) existe dans les services interactifs EDITO ;
        # les process "generic" ne fournissent que AWS_S3_ENDPOINT (hôte seul).
        endpoint = os.environ.get("S3_ENDPOINT") or os.environ.get("AWS_S3_ENDPOINT")
        if endpoint:
            if not endpoint.startswith(("http://", "https://")):
                endpoint = "https://" + endpoint
            kwargs["client_kwargs"] = {"endpoint_url": endpoint}
        _fs = s3fs.S3FileSystem(**kwargs)
    return _fs


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
        fs = get_s3fs()
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
        fs = get_s3fs()
        # S3Map transforme l'URL S3 en un dictionnaire virtuel que Zarr sait lire
        store = s3fs.S3Map(root=chemin_resolu, s3=fs)
        return zarr.open_consolidated(store)
        
    # 3. Ouverture Locale classique
    else:
        return zarr.open_consolidated(chemin_resolu)