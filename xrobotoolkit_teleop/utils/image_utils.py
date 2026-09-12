"""
Image compression and decompression utilities for teleoperation data logging.

This module provides functions for compressing images to JPG format to reduce
storage size in pickle files while maintaining reasonable quality.
"""

import cv2
import numpy as np


def compress_image_to_jpg(image: np.ndarray, quality: int = 85) -> bytes:
    """
    Compress image to JPG bytes for storage efficiency.
    
    Args:
        image: Input image array (numpy array)
        quality: JPG compression quality (1-100, higher is better quality)
        
    Returns:
        bytes: JPG compressed image bytes, or None if compression fails
    """
    if image is None:
        return None
    
    try:
        # For depth images or grayscale, convert to 8-bit for JPG compression
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # Normalize depth to 0-255 range for better compression
            if image.dtype != np.uint8:
                # Scale depth values to 8-bit range
                image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            else:
                image_normalized = image
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded_img = cv2.imencode('.jpg', image_normalized, encode_param)
        else:
            # For color images, use direct JPG encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        
        return encoded_img.tobytes()
    except Exception as e:
        print(f"Warning: Failed to compress image to JPG: {e}")
        return None


def decompress_jpg_to_image(jpg_bytes: bytes) -> np.ndarray:
    """
    Decompress JPG bytes back to numpy image array.
    
    Args:
        jpg_bytes: JPG compressed image bytes
        
    Returns:
        numpy.ndarray: Decompressed image array, or None if decompression fails
    """
    if jpg_bytes is None:
        return None
    
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return image
    except Exception as e:
        print(f"Warning: Failed to decompress JPG bytes: {e}")
        return None


def get_compression_ratio(original_image: np.ndarray, compressed_bytes: bytes) -> float:
    """
    Calculate the compression ratio between original image and compressed bytes.
    
    Args:
        original_image: Original numpy image array
        compressed_bytes: Compressed JPG bytes
        
    Returns:
        float: Compression ratio (original_size / compressed_size)
    """
    if original_image is None or compressed_bytes is None:
        return 0.0
    
    original_size = original_image.nbytes
    compressed_size = len(compressed_bytes)
    
    if compressed_size == 0:
        return 0.0
    
    return original_size / compressed_size


def compress_frame_dict(frame_dict: dict, quality: int = 85) -> dict:
    """
    Compress all images in a frame dictionary to JPG bytes.
    
    Args:
        frame_dict: Dictionary with camera names as keys and image data as values
        quality: JPG compression quality (1-100)
        
    Returns:
        dict: Dictionary with same structure but JPG bytes instead of numpy arrays
    """
    compressed_dict = {}
    
    for camera_name, camera_data in frame_dict.items():
        if isinstance(camera_data, dict):
            # Handle nested structure (e.g., {'color': image, 'depth': image})
            compressed_dict[camera_name] = {}
            for stream_type, image in camera_data.items():
                if isinstance(image, np.ndarray):
                    compressed_dict[camera_name][stream_type] = compress_image_to_jpg(image, quality)
                else:
                    compressed_dict[camera_name][stream_type] = image
        elif isinstance(camera_data, np.ndarray):
            # Handle direct image array
            compressed_dict[camera_name] = compress_image_to_jpg(camera_data, quality)
        else:
            # Pass through other data types unchanged
            compressed_dict[camera_name] = camera_data
    
    return compressed_dict


def decompress_frame_dict(compressed_dict: dict) -> dict:
    """
    Decompress all JPG bytes in a frame dictionary back to numpy arrays.
    
    Args:
        compressed_dict: Dictionary with JPG bytes
        
    Returns:
        dict: Dictionary with numpy arrays instead of JPG bytes
    """
    decompressed_dict = {}
    
    for camera_name, camera_data in compressed_dict.items():
        if isinstance(camera_data, dict):
            # Handle nested structure
            decompressed_dict[camera_name] = {}
            for stream_type, data in camera_data.items():
                if isinstance(data, bytes):
                    decompressed_dict[camera_name][stream_type] = decompress_jpg_to_image(data)
                else:
                    decompressed_dict[camera_name][stream_type] = data
        elif isinstance(camera_data, bytes):
            # Handle direct bytes
            decompressed_dict[camera_name] = decompress_jpg_to_image(camera_data)
        else:
            # Pass through other data types unchanged
            decompressed_dict[camera_name] = camera_data
    
    return decompressed_dict