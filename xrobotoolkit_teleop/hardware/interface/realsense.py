import threading
import time

import numpy as np
from .base_camera import BaseCameraInterface
from ...utils.image_utils import compress_image_to_jpg
import pyrealsense2 as rs


class RealSenseCameraInterface(BaseCameraInterface):
    """
    An interface to handle one or more Intel RealSense cameras.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_numbers: list[str] = None,
        enable_depth: bool = True,
        enable_compression: bool = True,
        jpg_quality: int = 85,
    ):
        """
        Initializes the RealSense camera interface.

        Args:
            width (int): The width of the camera streams.
            height (int): The height of the camera streams.
            fps (int): The frames per second of the camera streams.
            serial_numbers (list[str], optional): A list of serial numbers of the cameras to use.
                                                  If None, all connected cameras will be used.
            enable_depth (bool): Whether to enable the depth stream.
            enable_compression (bool): Whether to store compressed JPG bytes alongside raw frames.
            jpg_quality (int): JPG compression quality (1-100, higher is better quality).
        """
        super().__init__(enable_compression=enable_compression, jpg_quality=jpg_quality)
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_numbers = serial_numbers
        self.enable_depth = enable_depth
        self.pipelines = {}
        self.configs = {}
        self.align = {}

        # Raw frames for real-time access
        self.frames_dict = {}
        # Compressed frames for logging
        self.compressed_frames_dict = {}
        self.frames_lock = threading.Lock()  # Thread-safe access to frames
        self.last_update_time = {}  # Track last successful frame update per camera

        self.context = rs.context()
        devices = self.context.query_devices()

        if not devices:
            raise RuntimeError("No Intel RealSense devices connected.")

        device_serials = [d.get_info(rs.camera_info.serial_number) for d in devices]

        if self.serial_numbers:
            # Filter for specified serial numbers
            self.active_serials = [s for s in self.serial_numbers if s in device_serials]
            if not self.active_serials:
                raise RuntimeError(f"Specified RealSense devices with serials {self.serial_numbers} not found.")
        else:
            # Use all connected devices
            self.active_serials = device_serials

        for serial in self.active_serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            if self.enable_depth:
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            self.pipelines[serial] = pipeline
            self.configs[serial] = config
            # Align depth frames to color frames
            if self.enable_depth:
                self.align[serial] = rs.align(rs.stream.color)
            self.last_update_time[serial] = 0
            print(f"Initialized RealSense camera: {serial}")

    def start(self):
        """Starts the camera pipelines."""
        for serial, pipeline in self.pipelines.items():
            pipeline.start(self.configs[serial])
            print(f"Started pipeline for camera: {serial}")

    def update_frames(self):
        """
        Fetches and returns frames from all cameras with improved error handling and timeout management.

        Returns:
            dict: A dictionary where keys are camera serial numbers and values are
                  another dictionary containing 'color' and 'depth' numpy arrays,
                  the 'timestamp_us' of the frame, and stream format information.
        """
        current_time = time.time()
        frames_dict = {}

        for serial, pipeline in self.pipelines.items():
            try:
                # Use shorter timeout and exponential backoff on failures
                timeout_ms = 500  # Reduced timeout to prevent blocking
                frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)

                color_frame = None
                depth_frame = None

                if self.enable_depth:
                    aligned_frames = self.align[serial].process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                else:
                    color_frame = frames.get_color_frame()

                if not color_frame:
                    print(f"Warning: No color frame available from camera {serial}")
                    continue

                color_image = np.asanyarray(color_frame.get_data()).copy()
                depth_image = np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None

                frames_dict[serial] = {
                    "color": color_image,
                    "depth": depth_image,
                    "timestamp_us": color_frame.get_timestamp(),  # microseconds
                    "color_format": color_frame.get_profile().format(),
                    "depth_format": (depth_frame.get_profile().format() if depth_frame else None),
                }

                self.last_update_time[serial] = current_time

            except RuntimeError as e:
                # Handle timeout more gracefully
                if "timeout" in str(e).lower():
                    print(
                        f"Frame timeout for camera {serial} (last successful: {current_time - self.last_update_time[serial]:.2f}s ago)"
                    )
                else:
                    print(f"Error getting frames from {serial}: {e}")
                continue

        # Thread-safe update of frames dictionary
        with self.frames_lock:
            self.frames_dict = frames_dict
            
            # Store compressed frames for logging if enabled
            if self.enable_compression:
                compressed_dict = {}
                for serial, frame_data in frames_dict.items():
                    color_compressed = compress_image_to_jpg(frame_data["color"], self.jpg_quality) if frame_data["color"] is not None else None
                    depth_compressed = compress_image_to_jpg(frame_data["depth"], self.jpg_quality) if frame_data["depth"] is not None else None
                    
                    compressed_dict[serial] = {
                        "color": color_compressed,
                        "depth": depth_compressed,
                        "timestamp_us": frame_data["timestamp_us"],
                        "color_format": frame_data["color_format"],
                        "depth_format": frame_data["depth_format"],
                    }
                self.compressed_frames_dict = compressed_dict

    def get_frames(self):
        """
        Fetches frames from all initialized cameras (thread-safe).

        Returns:
            dict: A dictionary where keys are camera serial numbers and values are
                  another dictionary containing 'color' and 'depth' numpy arrays,
                  the 'timestamp_us' of the frame, and stream format information.
        """
        with self.frames_lock:
            return self.frames_dict.copy()

    def get_compressed_frames(self):
        """
        Returns the compressed frames from all cameras for logging (thread-safe).

        Returns:
            dict: A dictionary where keys are camera serial numbers and values are
                  another dictionary containing compressed 'color' and 'depth' JPG bytes,
                  the 'timestamp_us' of the frame, and stream format information.
        """
        with self.frames_lock:
            return self.compressed_frames_dict.copy()

    def get_frame(self, serial: str):
        """
        Fetches frames from a specific camera (thread-safe).

        Args:
            serial (str): The serial number of the camera.

        Returns:
            dict: A dictionary containing 'color' and 'depth' numpy arrays,
                  the 'timestamp_us' of the frame, and stream format information.
        """
        with self.frames_lock:
            return self.frames_dict[serial].copy() if serial in self.frames_dict else None

    def stop(self):
        """Stops the camera pipelines."""
        for serial, pipeline in self.pipelines.items():
            try:
                pipeline.stop()
                print(f"Stopped pipeline for camera: {serial}")
            except RuntimeError as e:
                print(f"Error stopping pipeline for {serial}: {e}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def get_supported_resolutions(serial_number: str = None):
    """
    Prints the supported resolutions for connected RealSense devices.

    Args:
        serial_number (str, optional): The serial number of a specific device.
                                       If provided, only resolutions for that
                                       device will be printed. Otherwise,
                                       resolutions for all connected devices
                                       will be printed.
    """
    ctx = rs.context()
    devices = ctx.query_devices()

    if not devices:
        print("No RealSense devices detected.")
        return

    found_device = False
    for device in devices:
        device_serial = device.get_info(rs.camera_info.serial_number)
        if serial_number is None or device_serial == serial_number:
            found_device = True
            print(f"--- Device: {device.get_info(rs.camera_info.name)} ---")
            print(f"    Serial Number: {device_serial}")

            for sensor in device.query_sensors():
                print(f"  --- Sensor: {sensor.get_info(rs.camera_info.name)} ---")
                for profile in sensor.get_stream_profiles():
                    video_profile = profile.as_video_stream_profile()
                    if video_profile:
                        width = video_profile.width()
                        height = video_profile.height()
                        stream_format = video_profile.format()
                        fps = video_profile.fps()

                        print(f"    - Resolution: {width}x{height}, Format: {stream_format}, FPS: {fps}")

            if serial_number:
                break

    if serial_number and not found_device:
        print(f"Device with serial number {serial_number} not found.")
