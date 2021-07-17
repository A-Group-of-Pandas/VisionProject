import numpy as np
from typing import Tuple

class Profile:
    def __init__(self, name, vectors) -> None:
        """
        Initializes a Profile object with a name and a list of descriptor vectors
        (if applicable). 

        Parameters:
        ---------------
        name : str
            name associapted with the profile
        vectors: np.ndarray
            list of already determined descriptor vectors (if applicable)
        """
        self.name = name
        self.length = len(vectors)
        self.vecs = vectors
        self.mean_vector = sum(vectors)/len(vectors)
        

    def update(self, dv) -> None:
        """
        Passes a descriptor vector into a profile, appends the vector to the
        profile's array of vectors. 

        Parameters:
        ---------------
        dv : np.ndarray
            a discriptor vector associated with the profile
        """
        self.mean_vector = (self.mean_vector * self.length + dv) / (self.length + 1)
        self.vecs.append(dv)
        self.length += 1
    
    @property
    def parameters(self) -> Tuple: 
        """
        Returns a tuple of the profile's parameters (name, vectors).

        Parameters: 
        --------------
        None
        """
        return (self.name, self.mean_vector, self.vecs)
    
def cosine_distance(descriptor1, descriptor2):
    dot_product = np.dot(descriptor1, descriptor2)
    cos_dist = 1-(dot_product/(np.linalg.norm(descriptor1)*np.linalg.norm(descriptor2)))
    return cos_dist
