import json
import os

default_path = json.load(open("../paths.json"))

def get_path(key):
    return os.environ.get(key, default_path[key])
