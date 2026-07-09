import json

path_info = json.load(open("../paths.json"))

def get_path(key):
    return path_info[key]
