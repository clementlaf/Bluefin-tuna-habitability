
import os
import pickle
from logger import log

from paths import get_path, resolve_path

res_path = get_path("RESOURCE_PATH")

def load_stats(prefix):
    mean_path = resolve_path(f"{res_path}/{prefix}_mean.pkl", True)
    std_path = resolve_path(f"{res_path}/{prefix}_std.pkl", True)

    if os.path.exists(mean_path) and os.path.exists(std_path):
        log(f"Loading existing stats for {prefix}")
        with open(mean_path, "rb") as f:
            mean = pickle.load(f)
        with open(std_path, "rb") as f:
            std = pickle.load(f)
    else:
        log(f"Computing stats for {prefix}...")
        raise NotImplementedError("Stats computation is currently disabled to save time. Please compute them once and save the results.")

    return mean, std


# mean_phys, std_phys = load_stats(res_path, "phys")
# mean_bio, std_bio = load_stats(res_path, "bio")