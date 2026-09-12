import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image

from ...utils.image_utils import compress_image_to_jpg
from .base_camera import BaseCameraInterface


class RosCameraInterface(BaseCameraInterface):
    """
    An interface to handle one or more cameras through ROS topics.
    """

    def __init__(
        self,
        camera_topics: dict,
        enable_depth: bool = True,
        width: int = None,
        height: int = None,
        enable_compression: bool = True,
        jpg_quality: int = 85,
    ):
        """
        Initializes the ROS camera interface.

        Args:
            camera_topics (dict): A dictionary where keys are camera names and
                                  values are another dictionary with 'color' and 'depth' topic names.
                                  e.g., {"left": {"color": "/topic/left_color", "depth": "/topic/left_depth"}}
            enable_depth (bool): Whether to enable the depth stream.
            width (int): Target width for resizing images. If None, no resizing is performed.
            height (int): Target height for resizing images. If None, no resizing is performed.
            enable_compression (bool): Whether to store compressed JPG bytes alongside raw frames.
            jpg_quality (int): JPG compression quality (1-100, higher is better quality).
        """
        super().__init__(enable_compression=enable_compression, jpg_quality=jpg_quality)
        self.camera_topics = camera_topics
        self.enable_depth = enable_depth
        self.width = width
        self.height = height
        self.bridge = CvBridge()

        # Raw frames for real-time access
        self.frames_dict = {}
        # Compressed frames for logging
        self.compressed_frames_dict = {}
        self.frames_lock = threading.Lock()
        self.subscribers = []

    def start(self):
        """Starts the camera subscribers."""
        for name, topics in self.camera_topics.items():
            if "color" not in topics:
                print(f"Warning: 'color' topic not specified for camera {name}. Skipping.")
                continue

            color_topic = topics["color"]
            self.subscribers.append(
                rospy.Subscriber(
                    color_topic,
                    CompressedImage,
                    self._color_callback,
                    callback_args=name,
                )
            )

            if self.enable_depth and "depth" in topics:
                depth_topic = topics["depth"]
                self.subscribers.append(rospy.Subscriber(depth_topic, Image, self._depth_callback, callback_args=name))
            print(f"Subscribed to topics for camera: {name}")

    def _resize_image(self, image):
        """
        Resize image if width and height are specified.

        Args:
            image: Input image array

        Returns:
            Resized image or original image if no resizing needed
        """
        if self.width is not None and self.height is not None and image is not None:
            return cv2.resize(image, (self.width, self.height))
        return image

    def _color_callback(self, msg, camera_name):
        np_arr = np.frombuffer(msg.data, np.uint8)
        color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        color_image = self._resize_image(color_image)

        # Store both raw and compressed data
        with self.frames_lock:
            if camera_name not in self.frames_dict:
                self.frames_dict[camera_name] = {}
                self.compressed_frames_dict[camera_name] = {}

            # Store raw image for real-time access
            self.frames_dict[camera_name]["color"] = color_image

            # Store compressed image for logging
            if self.enable_compression:
                compressed_bytes = compress_image_to_jpg(color_image, self.jpg_quality)
                self.compressed_frames_dict[camera_name]["color"] = compressed_bytes

    def _depth_callback(self, msg, camera_name):
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_image = self._resize_image(depth_image)

        # Store both raw and compressed data
        with self.frames_lock:
            if camera_name not in self.frames_dict:
                self.frames_dict[camera_name] = {}
                self.compressed_frames_dict[camera_name] = {}

            # Store raw image for real-time access
            self.frames_dict[camera_name]["depth"] = depth_image

            # Store compressed image for logging
            if self.enable_compression:
                compressed_bytes = compress_image_to_jpg(depth_image, self.jpg_quality)
                self.compressed_frames_dict[camera_name]["depth"] = compressed_bytes

    def stop(self):
        """Stops the camera subscribers."""
        for sub in self.subscribers:
            sub.unregister()
        print("Unregistered all camera subscribers.")

    def update_frames(self):
        """
        No-op for ROS-based interface, as frames are updated by callbacks.
        """
        pass

    def get_frames(self):
        """
        Returns the last received frames from all cameras.

        Returns:
            dict: A dictionary of frames.
        """
        with self.frames_lock:
            frames_dict = {}
            for camera_name, frame_data in self.frames_dict.items():
                color_frame = frame_data.get("color")
                depth_frame = frame_data.get("depth")
                frames_dict[camera_name] = {
                    "color": color_frame.copy() if color_frame is not None else None,
                    "depth": depth_frame.copy() if self.enable_depth and depth_frame is not None else None,
                }
            return frames_dict

    def get_compressed_frames(self):
        """
        Returns the last received compressed frames from all cameras for logging.

        Returns:
            dict: A dictionary of compressed frames (JPG bytes).
        """
        with self.frames_lock:
            compressed_dict = {}
            for camera_name, frame_data in self.compressed_frames_dict.items():
                color_bytes = frame_data.get("color")
                depth_bytes = frame_data.get("depth")
                compressed_dict[camera_name] = {
                    "color": color_bytes[:],
                    "depth": depth_bytes[:] if self.enable_depth else None,
                }
            return compressed_dict

    def get_frame(self, camera_name: str):
        """
        Returns the last received frame from a specific camera.

        Args:
            camera_name (str): The name of the camera.

        Returns:
            dict: A dictionary containing 'color' and 'depth' numpy arrays.
        """
        with self.frames_lock:
            return self.frames_dict.get(camera_name, {}).copy()
