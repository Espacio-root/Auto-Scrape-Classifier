import cv2
from PIL import Image, ImageDraw
import numpy as np
import os
import time
import threading
    
class WebCam:
    
    def __init__(self, callbackfn, delay=0.5):
        self.delay = delay
        self.callbackfn = callbackfn
        self.frame = None
        self.text = 'Thinking...'
        
    @staticmethod
    def add_text(image, text):
        img_pil = Image.fromarray(image)
        I1 = ImageDraw.Draw(img_pil)
        I1.text((28, 36), text, fill=(255, 0, 0))
        
        return np.array(img_pil)
    
    def wait_thread(self):
        while True:
            time.sleep(self.delay)
            if self.frame is not None:
                self.text = self.callbackfn(self.frame)

    def start(self):
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open webcam")
            return

        t = threading.Thread(target=self.wait_thread)
        t.start()

        while True:
            ret, self.frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam")
                break

            image_with_text = WebCam.add_text(self.frame, self.text)

            cv2.imshow("Webcam Feed", image_with_text)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        os._exit(0)
        

if __name__ == "__main__":
    WebCam(lambda x: 'Hello', 3).start()