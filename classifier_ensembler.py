from getimgs import GetImgs
from model import BinaryClassifier
from webcam import WebCam

import os
from PIL import Image
import matplotlib.pyplot as plt

class Ensembler:
    
    def __init__(self, object_to_classify, num_to_scrape=200, overwrite_image_dir=False, images_dir='images', model_data_dir='model_data', model_path_dir='best_models', train_size=0.85, num_epochs=10, webcam_delay=0.1) -> None:
        self.ratios = [object_to_classify, 'random background images']
        self.num = num_to_scrape
        self.images_dir = images_dir
        self.model_path_dir = model_path_dir
        self.train_size = train_size
        self.overwrite = overwrite_image_dir
        self.model_data_dir = model_data_dir
        self.num_epochs = num_epochs
        
        self.clf = None
        self.model = None
        self.delay = webcam_delay
        
    def start(self):   
        GetImgs(self.ratios, self.num, self.images_dir, self.overwrite).run()

        if not os.path.exists(self.model_path_dir):
            os.mkdir(self.model_path_dir)
        self.clf = BinaryClassifier(self.ratios, self.train_size, self.num, self.num_epochs, self.images_dir, self.model_data_dir)
        model_path = os.path.join(os.getcwd(), self.model_path_dir, self.ratios[0] + '.pth')
        if not os.path.exists(model_path):
            self.clf.create_dataset()
            self.clf.run(model_path)
        self.model = self.clf.load_best_model(model_path)
        
        WebCam(self.callbackfn, self.delay).start()
        
    def callbackfn(self, frame):
        img_pil = Image.fromarray(frame)
        if self.clf and self.model:
            text = self.clf.probability(self.model, img_pil)
            text = 1 - text
            return f'{text:.2f}%'
        else:
            return 'Thinking...'
        
if __name__ == '__main__':
    Ensembler('water bottle', num_to_scrape=1500).start()
            
    