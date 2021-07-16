from facenet_models import FacenetModel
import numpy as np
class Detector:
    def __init__(self) -> None:
        self.model = FacenetModel()
        self.confidence_thres = 0.8
        self.size_cutoff = 0.2
    
    def detect(self,image):
        boxes, probabilities, landmarks = self.model.detect(image)
        if boxes is None:
            return None
        boxes = [boxes[i] for i in range(len(boxes)) if probabilities[i]>self.confidence_thres]
        probabilities = [probabilities[i] for i in range(len(probabilities)) if probabilities[i]>self.confidence_thres]
        areas = [(box[2]-box[0])*(box[3]-box[1]) for box in boxes]
        areas = np.array(areas)
        mean = np.average(areas,weights=probabilities)
        cutoff = mean*self.size_cutoff
        print(f'cutoff is {cutoff}')
        print(areas)
        boxes = [boxes[i] for i in range(len(boxes)) if areas[i]>cutoff]
        return boxes

    def detect_one(self, image):
        boxes, probabilities, landmarks = self.model.detect(image)
        max_conf = self.confidence_thres
        max_box = None
        for i in range(len(boxes)):
            if probabilities[i]>max_conf:
                max_conf = probabilities[i]
                max_box = boxes[i]
        return max_box

    def get_vectors(self, image, boxes):
        descriptors = self.model.compute_descriptors(image, boxes)
        return descriptors
