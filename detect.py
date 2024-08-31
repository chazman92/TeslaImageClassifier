import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from datetime import datetime, timedelta
from datetime import time as datetime_time 
import time
import os
import logging
import argparse
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class TeslaDetection:
    def __init__(self, url, boolSaveImg, boolShowVid, boolRunDocker):
        self._URL = url
        if boolRunDocker:
            self.savedImagesPath = "/app/SavedImages/"
        else:
            self.savedImagesPath = "/media/chazman/SanDisk1TB/TeslaCam/SavedImages/"
        self.boolSaveImages = boolSaveImg
        self.boolShowVideo = boolShowVid
        self.timeStartWindow = "07:00"
        self.timeEndWindow = "19:00"
        self.model = self.cls_load_model()
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.previous_time = datetime.now()
        self.delta_time = 0
        self.previous_coord = None
        self.batch_size = 4  # Number of frames to process at once

    def __call__(self):
        cap = cv2.VideoCapture(self._URL)
        target_interval = 4  # Desired interval for 4 FPS
        frame_buffer = []

        while cap.isOpened():
            ret, frame = self.read_frame_with_skip(cap, target_interval)
            if not ret:
                break

            frame_buffer.append(frame)
            if len(frame_buffer) == self.batch_size:
                self.process_batch(frame_buffer)
                frame_buffer.clear()

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_batch(self, frames):
        maskedFrames = [self.cls_mask_image(frame)[0] for frame in frames]
        results = self.model(maskedFrames, verbose=False)

        for frame, result in zip(frames, results):
            self.process_frame(frame, result)

    def process_frame(self, frame, result):
        """Process the frame, detect objects, and return the original frame without annotations."""
        boxes = result.boxes
        for box in boxes:
            c = box.cls
            label = self.model.names[int(c)]
            confidence = round(float(box.conf), 4)

            if confidence > 0.5:
                if label in ['car', 'truck']:
                    if self.boolSaveImages:
                        if self.previous_coord is not None and self.is_almost_same_location(self.previous_coord, box.xyxy[0]):
                            print(f"{label} is in almost the same location. Skipping export.")
                            continue
                        else:
                            print(f"{label} detected or has moved.")
                            self.previous_coord = box.xyxy[0]
                            self.cls_export_image(frame)  # Export original frame without annotations
                            print(f"{label} {confidence} \r\n")

        if self.boolShowVideo:
            resized_frame = self.cls_resize_image(frame, 25)  # Resize to 25% for display
            cv2.imshow("YOLO", resized_frame)
    
    def read_frame_with_skip(self, cap, skip_frames):
        try:
            counter = 0
            while counter < skip_frames:
                ret = cap.grab()
                if not ret:
                    raise ValueError("Failed to grab frame during skipping.")
                counter += 1
            
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Failed to read frame after skipping.")
            
            return ret, frame
        
        except Exception as e:
            print(f"Error occurred while reading frame: {e}")
            return False, None

    def is_within_time_frame(self, start_time, end_time):
        try:
            start_hour, start_minute = map(int, start_time.split(':'))
            end_hour, end_minute = map(int, end_time.split(':'))
            now = datetime.now().time()
            start_time_obj = datetime_time(start_hour, start_minute)
            end_time_obj = datetime_time(end_hour, end_minute)
            return start_time_obj <= now <= end_time_obj
        except ValueError:
            print("Error: Invalid time format.")
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def cls_load_model(self):
        model = YOLO("current_model/yolov8n.pt")
        model.conf = 0.5
        model.iou = 0.45
        model.multi_label = False
        return model

    def cls_mask_image(self, frame):
        mask = np.zeros_like(frame)
        x1, y1, x2, y2 = 300*4, 0*4, 600*4, 359*4
        pts = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        #cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        masked_image = cv2.bitwise_and(frame, mask)
        return masked_image, frame

    def cls_export_image(self, frame):
        try:
            if self.is_within_time_frame(self.timeStartWindow, self.timeEndWindow):
                current_time = datetime.now()
                self.delta_time += (current_time - self.previous_time).total_seconds()
                self.previous_time = current_time
                path_date = datetime.now().strftime(f"%b_%d")
                path = f"{self.savedImagesPath}{path_date}/"

                if not os.path.exists(path):
                    os.makedirs(path, mode=0o777)

                if self.delta_time > 2:
                    self.delta_time = 0
                    date_time = datetime.now().strftime("%H_%M_%S")
                    img_file = f"{path}{date_time}_image.png"
                    cv2.imwrite(img_file, frame, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                    print("Image Saved:", img_file)
        except Exception as e:
            print("Error exporting image:", e)

    def cls_resize_image(self, src_image, scale_percent):
        # calculate the 50 percent of original dimensions
        width = int(src_image.shape[1] * scale_percent / 100)
        height = int(src_image.shape[0] * scale_percent / 100)

        dsize = (width, height)
        resized_img = cv2.resize(src_image, dsize, interpolation=cv2.INTER_CUBIC)

        return resized_img
    
    def is_almost_same_location(self, coord1, coord2, threshold=10):
        distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
        return distance < threshold

# ... (rest of your script remains the same)
def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    # Optional arguments
    parser.add_argument(
        "-u",
        "--urlPath",
        help="URL Path.",
        type=str,
        default="rtsp://chazman:navy92@192.168.1.9:88/videoMain",
    )
    parser.add_argument(
        "-s", "--saveImages", help="Bool Don't Save Images.", action='store_false', default=True)
    parser.add_argument(
        "-v", "--hideVideo", help="Bool Hide Video", action='store_false', default=True)
    parser.add_argument(
        "-d", "--runDocker", help="Run in Docker", action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()

    return args

def run_analysis(path_to_video, boolSaveImg, boolShowVid, boolRunDocker):
    while True:
        try:
            detect = TeslaDetection(path_to_video, boolSaveImg, boolShowVid, boolRunDocker)  
            detect()
            print("Delete Object and Restarted in run_analysis loop...")
            detect.instance = None
            time.sleep(1)  # Sleep for a while before the next iteration

        except ValueError as e:
            logging.error(f"Caught an error: {e}")
            time.sleep(5)  

        except KeyboardInterrupt:
            logging.info("Received a keyboard interrupt. Exiting gracefully...")
            break 

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            time.sleep(10)  
    
if __name__ == "__main__":
    # Parse the arguments
    args = parseArguments()

    try:
        run_analysis(args.urlPath, args.saveImages, args.hideVideo, args.runDocker)
    finally:
        print("Exited script through Finally statement.")