from detector import Detector
from database import Database

database = Database()
database.add_images(dir_path='images')

database.save('database.pkl')