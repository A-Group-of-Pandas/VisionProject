from typing import List
import numpy as np
import pickle as pkl
from profile import Profile


class Database:
    def __init__(self) -> None:
        self.cutoff_dist = 0.3
        self.profiles = {}
        self.load("database.pkl")
    
    def load(self, file_path):
        with open(file_path, "rb") as f:
            data = pkl.load(f)
        for d in data:
            self.profiles[d["name"]] = Profile(d["name"], d["mean_vector"], d["length"])

    def search(self, face_vector : np.ndarray):
        min_cos_dist = 1e9
        min_profile = None
        for profile in self.profiles:
            cos_dist = Profile.cosine_distance(profile.mean_vector, face_vector)
            if cos_dist<min_cos_dist:
                min_cos_dist = cos_dist
                min_profile = profile
        if min_cos_dist < self.cutoff_dist:
            return min_profile
        else:
            return None
        
    def save(self, file_path):
        with open("database.pkl", "wb") as f:
            pkl.dump(self.profiles, f)

    # profile must be type Profile
    def add(self, profile):
        name = profile.parameters[0]
        self.profiles[name] = profile

    def remove(self, profile):
        name = profile.parameters[0]
        self.profiles.pop(name, None)

    def add_desc_vector(self, dv, name):
        if name in self.profiles.keys():
            self.profiles[name].update(dv)
        else:
            profile = Profile(name, np.array(dv))
            self.add(profile)
