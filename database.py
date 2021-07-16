from detector import Detector
from typing import List
import numpy as np
import pickle as pkl
from profile import Profile, cosine_distance
import skimage.io as io
import numpy as np
from os import listdir



class Database:
    def __init__(self) -> None:
        self.cutoff_dist = 0.3
        self.profiles = {}
        self.detector = Detector()
        self.load("database.pkl")
    
    def load(self, file_path):
        with open(file_path, "rb") as f:
            self.profiles = pkl.load(f)

    def search(self, face_vector : np.ndarray):
        min_cos_dist = 1e9
        min_name = None
        for name in self.profiles:
            cos_dist = cosine_distance(self.profiles[name].mean_vector, face_vector)
            if cos_dist<min_cos_dist:
                min_cos_dist = cos_dist
                min_name = name
        if min_cos_dist < self.cutoff_dist:
            return min_name
        else:
            return None
        
    def save(self, file_path):
        with open(file_path, "wb") as f:
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
            profile = Profile(name, np.array(dv[None,...]))
            self.add(profile)
            
    def load_image(self, path_to_image):
        image_path = str(path_to_image)
        name = image_path.split('.')[0].split('/')[-1].split("-")
        name = ' '.join(name)
        name = ''.join([i for i in name if not i.isdigit()])
        image = io.imread(image_path)
        if image.shape[-1] == 4:
            # Image is RGBA, where A is alpha -> transparency
            # Must make image RGB.
            image = image[..., :-1]  # png -> RGB
        return name, image

    def add_image(self, file_path : str):
        name, img = self.load_image(file_path)
        box = self.detector.detect_one(img)
        if box is None:
            return
        vectors = self.detector.get_vectors(img, [box])
        self.add_desc_vector(np.array(vectors[0]), name)

    def add_images(self, *, dir_path : str):
        # Listdir: Lists all directories within a folder.
        files = listdir(dir_path)
        for file in files:
            if 'DS_Store' in file:
                continue
            print(f'reading {file}')
            self.add_image(dir_path+"/"+file)
        
