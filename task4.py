import cv2
import logging
import os
import sys
import time
import threading
import queue
import argparse
import numpy as np
from typing import Any, Optional, List, Tuple

os.makedirs("log", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/app.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height
        self.cap = None
        self._error_count = 0
        self._max_errors = 10
        
        try:
            if str(name).isdigit():
                self.cap = cv2.VideoCapture(int(name))
            else:
                self.cap = cv2.VideoCapture(name)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Camera '{name}' not found or cannot be opened")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Camera '{name}' initialized with resolution {actual_width}x{actual_height}")
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            self.__del__()
            raise

    def get(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            self._error_count += 1
            logger.error(f"Camera is not available (error {self._error_count}/{self._max_errors})")
            if self._error_count >= self._max_errors:
                logger.critical("Too many camera errors, terminating program")
                sys.exit(1)
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                self._error_count += 1
                logger.error(f"Failed to capture frame (error {self._error_count}/{self._max_errors})")
                if self._error_count >= self._max_errors:
                    logger.critical("Too many capture errors, terminating program")
                    sys.exit(1)
                return None
            
            self._error_count = 0
            return frame
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error capturing frame: {str(e)} (error {self._error_count}/{self._max_errors})")
            if self._error_count >= self._max_errors:
                logger.critical("Too many errors, terminating program")
                sys.exit(1)
            return None

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
            logger.info(f"Camera '{self.name}' released")


class WindowImage:
    def __init__(self, window_name = "Sensor Display", display_fps = 30.0):
        self.window_name = window_name
        self.display_fps = display_fps
        self._error_count = 0
        self._max_errors = 10
        
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            logger.info(f"Window '{window_name}' created with display FPS: {display_fps}")
        except Exception as e:
            logger.error(f"Failed to create window: {str(e)}")
            raise

    def show(self, image: np.ndarray) -> Optional[int]:
        if image is None:
            return None
        
        try:
            cv2.imshow(self.window_name, image)
            wait_time = max(1, int(1000 / self.display_fps))
            key = cv2.waitKey(wait_time) & 0xFF
            self._error_count = 0  
            return key
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to display image (error {self._error_count}/{self._max_errors}): {str(e)}")
            if self._error_count >= self._max_errors:
                logger.critical("Too many display errors, terminating program")
                sys.exit(1)
            return None

    def __del__(self):
        try:
            cv2.destroyWindow(self.window_name)
            logger.info(f"Window '{self.window_name}' destroyed")
        except:
            pass


class SensorDataProcessor:
    def __init__(self, camera: Optional[SensorCam], sensors_x: List[SensorX]):
        self.camera = camera
        self.sensors_x = sensors_x
        self.stop_event = threading.Event()
        self.queues: List[Tuple[str, queue.Queue]] = []
        self.threads: List[threading.Thread] = []
        
    def start(self):
        if self.camera:
            q_cam = queue.Queue(maxsize=1)
            self.queues.append(('cam', q_cam))
            t = threading.Thread(
                target=self._sensor_worker,
                args=(self.camera, q_cam, self.stop_event),
                daemon=True,
                name="CameraWorker"
            )
            t.start()
            self.threads.append(t)
        
        for i, sensor in enumerate(self.sensors_x):
            q = queue.Queue(maxsize=1)
            self.queues.append((f'sensor_{i}', q))
            t = threading.Thread(
                target=self._sensor_worker,
                args=(sensor, q, self.stop_event),
                daemon=True,
                name=f"SensorX_{i}_Worker"
            )
            t.start()
            self.threads.append(t)
        
        logger.info(f"Started {len(self.threads)} worker threads")
    
    @staticmethod
    def _sensor_worker(sensor: Sensor, out_queue: queue.Queue, stop_event: threading.Event):
        logger.info(f"Worker started for {sensor.__class__.__name__}")
        
        while not stop_event.is_set():
            try:
                data = sensor.get()
                
                if data is not None:
                    try:
                        while not out_queue.empty():
                            out_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    try:
                        out_queue.put(data, block=False)
                    except queue.Full:
                        pass  
                        
            except Exception as e:
                logger.error(f"Error in sensor worker: {str(e)}")
        
        logger.info(f"Worker stopped for {sensor.__class__.__name__}")
    
    @staticmethod
    def get_latest_from_queue(q: queue.Queue, last_value: Any) -> Any:
        try:
            while True:
                last_value = q.get_nowait()
        except queue.Empty:
            pass
        return last_value
    
    def run_display_loop(self, window: WindowImage):
        last_frame = None
        last_values = [0] * len(self.sensors_x)
        sensor_labels = ["100 Hz", "10 Hz", "1 Hz"]
        
        logger.info("Starting main display loop")
        
        try:
            while not self.stop_event.is_set():
                for name, q in self.queues:
                    if name == 'cam':
                        last_frame = self.get_latest_from_queue(q, last_frame)
                    else:
                        sensor_idx = int(name.split('_')[1])
                        last_values[sensor_idx] = self.get_latest_from_queue(q, last_values[sensor_idx])
                
                display_image = self._create_display_image(
                    last_frame, last_values, sensor_labels
                )
                
                key = window.show(display_image)
                if key == ord('q'):
                    logger.info("'q' pressed - shutting down")
                    self.stop_event.set()
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by Ctrl+C - shutting down")
            self.stop_event.set()
        except Exception as e:
            logger.error(f"Error in display loop: {str(e)}")
            self.stop_event.set()
    
    def _create_display_image(self, frame: Optional[np.ndarray], 
                             sensor_values: List[int], 
                             labels: List[str]) -> np.ndarray:
        
        if frame is not None:
            display = frame.copy()
        else:
            display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(display, "NO CAMERA SIGNAL", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        y_offset = 30
        for i, (value, label) in enumerate(zip(sensor_values, labels)):
            text = f"Sensor {i+1} ({label}): {value}"
            cv2.putText(display, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        cv2.putText(display, "Press 'q' to quit", (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    def stop(self):
        logger.info("Stopping all threads")
        self.stop_event.set()
        
        for t in self.threads:
            t.join(timeout=2.0)
            if t.is_alive():
                logger.warning(f"Thread {t.name} did not stop gracefully")


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--camera', type=str, default='0')
    parser.add_argument('--resolution', type=str, default='1280x720')
    parser.add_argument('--fps', type=float, default=30.0)
    
    return parser.parse_args()


def main():
    
    args = parse_arguments()
    logger.info(f"Command line arguments: camera={args.camera}, resolution={args.resolution}, fps={args.fps}")
    
    try:
        width, height = map(int, args.resolution.split('x'))
        logger.info(f"Requested resolution: {width}x{height}")
    except ValueError:
        logger.error(f"Invalid resolution format: {args.resolution}")
        logger.error("Expected format: WIDTHxHEIGHT (e.g., 1280x720)")
        sys.exit(1)
    
    logger.info("Initializing camera")
    try:
        camera = SensorCam(args.camera, width, height)
    except Exception as e:
        logger.critical(f"Cannot initialize camera: {str(e)}. Program terminated.")
        sys.exit(1)
    
    logger.info("Creating SensorX instances")
    sensors_x = [
        SensorX(0.01),  # период 0.01 сек
        SensorX(0.1),  
        SensorX(1.0)  
    ]
    logger.info(f"Created {len(sensors_x)} SensorX sensors (100Hz, 10Hz, 1Hz)")
    
    logger.info("Creating display window")
    try:
        window = WindowImage("Sensor Display", args.fps)
    except Exception as e:
        logger.critical(f"Cannot create window: {str(e)}. Program terminated.")
        sys.exit(1)
    
    processor = SensorDataProcessor(camera, sensors_x)
    
    try:
        processor.start()
        processor.run_display_loop(window)
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
    finally:
        processor.stop()
        cv2.destroyAllWindows()
        logger.info("Program terminated cleanly")


if __name__ == "__main__":
    main()
