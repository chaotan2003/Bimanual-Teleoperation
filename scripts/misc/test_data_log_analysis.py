#!/usr/bin/env python3
"""
XRoboToolkit Teleop Data Log Analysis Script

This script analyzes the structure of .pkl files in the logs directory to understand
the teleoperation data format, including robot state data and camera images.

Usage:
    python test_data_log_analysis.py [pkl_file_path]
    
    If no file path is provided, it will analyze the first .pkl file found in logs/.
"""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import cv2

from xrobotoolkit_teleop.utils.image_utils import decompress_jpg_to_image


class TeleopDataAnalyzer:
    """Analyzes teleoperation pickle log files to understand their structure."""
    
    def __init__(self, pkl_file_path: str):
        """
        Initialize the analyzer with a pickle file path.
        
        Args:
            pkl_file_path (str): Path to the pickle file to analyze
        """
        self.pkl_file_path = Path(pkl_file_path)
        self.data: Optional[List[Dict]] = None
        self.analysis_results: Dict[str, Any] = {}
    
    def load_data(self) -> bool:
        """
        Safely load the pickle file data.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            print(f"Loading pickle file: {self.pkl_file_path}")
            print(f"File size: {self.pkl_file_path.stat().st_size / (1024*1024):.2f} MB")
            
            with open(self.pkl_file_path, 'rb') as f:
                self.data = pickle.load(f)
                
            print(f"✅ Successfully loaded {len(self.data)} data entries")
            return True
            
        except Exception as e:
            print(f"❌ Error loading pickle file: {e}")
            return False
    
    def get_data_type_info(self, value: Any) -> str:
        """
        Get detailed type information for a value.
        
        Args:
            value: The value to analyze
            
        Returns:
            str: Descriptive type information
        """
        if isinstance(value, np.ndarray):
            return f"numpy.ndarray(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, tuple):
            if len(value) == 0:
                return "tuple(empty)"
            elif len(value) < 10:
                return f"tuple(length={len(value)}, types={[type(x).__name__ for x in value]})"
            else:
                return f"tuple(length={len(value)}, sample_types={[type(value[i]).__name__ for i in [0, len(value)//2, -1]]})"
        elif isinstance(value, list):
            if len(value) == 0:
                return "list(empty)"
            elif len(value) < 10:
                return f"list(length={len(value)}, types={[type(x).__name__ for x in value]})"
            else:
                return f"list(length={len(value)}, sample_types={[type(value[i]).__name__ for i in [0, len(value)//2, -1]]})"
        elif isinstance(value, dict):
            return f"dict(keys={list(value.keys())[:5]}{'...' if len(value) > 5 else ''})"
        elif isinstance(value, (int, float)):
            return f"{type(value).__name__}({value})"
        elif isinstance(value, str):
            return f"str(length={len(value)}, preview='{value[:50]}{'...' if len(value) > 50 else ''}')"
        elif isinstance(value, bytes):
            return f"bytes(length={len(value)}, compressed_image_data)"
        else:
            return f"{type(value).__name__}"
    
    def analyze_data_structure(self):
        """Analyze the overall structure of the loaded data."""
        if not self.data:
            print("❌ No data loaded")
            return
        
        print("\n" + "="*80)
        print("DATA STRUCTURE ANALYSIS")
        print("="*80)
        
        # Basic info
        self.analysis_results['total_entries'] = len(self.data)
        print(f"Total number of logged entries: {len(self.data)}")
        
        # Analyze first entry structure
        if self.data:
            first_entry = self.data[0]
            print(f"\nFirst entry has {len(first_entry)} fields:")
            
            self.analysis_results['fields'] = {}
            for key, value in first_entry.items():
                type_info = self.get_data_type_info(value)
                self.analysis_results['fields'][key] = type_info
                print(f"  '{key}': {type_info}")
        
        # Check data consistency across entries
        self._check_data_consistency()
    
    def _check_data_consistency(self):
        """Check if all entries have the same structure."""
        if len(self.data) < 2:
            return
        
        print(f"\n📊 CONSISTENCY CHECK (across {len(self.data)} entries):")
        
        first_keys = set(self.data[0].keys())
        inconsistent_entries = []
        
        for i, entry in enumerate(self.data[1:], 1):
            entry_keys = set(entry.keys())
            if entry_keys != first_keys:
                inconsistent_entries.append(i)
        
        if inconsistent_entries:
            print(f"⚠️  Found {len(inconsistent_entries)} entries with different structure")
            print(f"   Inconsistent entries: {inconsistent_entries[:10]}{'...' if len(inconsistent_entries) > 10 else ''}")
        else:
            print("✅ All entries have consistent structure")
    
    def analyze_robot_state_data(self):
        """Analyze robot state data (joint positions, velocities, etc.)."""
        if not self.data:
            return
        
        print("\n" + "="*80)
        print("ROBOT STATE DATA ANALYSIS")
        print("="*80)
        
        robot_keys = ['qpos', 'qvel', 'qpos_des', 'gripper_qpos', 'gripper_qpos_des', 'gripper_target']
        found_robot_keys = [key for key in robot_keys if key in self.data[0]]
        
        if not found_robot_keys:
            print("❌ No robot state data found")
            return
        
        print(f"✅ Found robot state fields: {found_robot_keys}")
        
        for key in found_robot_keys:
            print(f"\n📈 Analyzing '{key}':")
            self._analyze_robot_field(key)
    
    def _analyze_robot_field(self, field_name: str):
        """Analyze a specific robot state field."""
        sample_data = self.data[0][field_name]
        
        if isinstance(sample_data, dict):
            print(f"   Structure: Dictionary with arms: {list(sample_data.keys())}")
            
            for arm_name, arm_data in sample_data.items():
                if isinstance(arm_data, np.ndarray):
                    print(f"   '{arm_name}': {self.get_data_type_info(arm_data)}")
                    
                    # Check data range across all entries
                    all_values = []
                    for entry in self.data[:min(100, len(self.data))]:  # Sample first 100 entries
                        if field_name in entry and arm_name in entry[field_name]:
                            if entry[field_name][arm_name] is not None:
                                all_values.append(entry[field_name][arm_name])
                    
                    if all_values:
                        all_values = np.array(all_values)
                        print(f"      Range (first 100 entries): min={all_values.min():.4f}, max={all_values.max():.4f}")
                else:
                    print(f"   '{arm_name}': {self.get_data_type_info(arm_data)}")
        else:
            print(f"   Structure: {self.get_data_type_info(sample_data)}")
    
    def analyze_camera_data(self):
        """Analyze camera image data."""
        if not self.data:
            return
        
        print("\n" + "="*80)
        print("CAMERA DATA ANALYSIS")
        print("="*80)
        
        image_keys = ['image', 'images', 'camera_frames', 'frames']
        found_image_key = None
        
        for key in image_keys:
            if key in self.data[0]:
                found_image_key = key
                break
        
        if not found_image_key:
            print("❌ No camera/image data found")
            return
        
        print(f"✅ Found camera data field: '{found_image_key}'")
        
        sample_images = self.data[0][found_image_key]
        
        if isinstance(sample_images, dict):
            print(f"   Camera structure: Dictionary with cameras: {list(sample_images.keys())}")
            
            for camera_name, image_data in sample_images.items():
                print(f"   '{camera_name}': {self.get_data_type_info(image_data)}")
                
                if isinstance(image_data, np.ndarray) and len(image_data.shape) >= 2:
                    print(f"      Image dimensions: {image_data.shape}")
                    print(f"      Data type: {image_data.dtype}")
                    if len(image_data.shape) == 3:
                        print(f"      Color channels: {image_data.shape[2]}")
        else:
            print(f"   Structure: {self.get_data_type_info(sample_images)}")
    
    def analyze_timestamps(self):
        """Analyze timing information."""
        if not self.data:
            return
        
        print("\n" + "="*80)
        print("TIMESTAMP ANALYSIS")
        print("="*80)
        
        if 'timestamp' not in self.data[0]:
            print("❌ No timestamp field found")
            return
        
        print("✅ Found timestamp data")
        
        timestamps = [entry['timestamp'] for entry in self.data if 'timestamp' in entry]
        
        if len(timestamps) > 1:
            timestamps = np.array(timestamps)
            duration = timestamps[-1] - timestamps[0]
            avg_freq = len(timestamps) / duration if duration > 0 else 0
            
            print(f"   Recording duration: {duration:.2f} seconds")
            print(f"   Average frequency: {avg_freq:.1f} Hz")
            print(f"   First timestamp: {timestamps[0]:.3f}")
            print(f"   Last timestamp: {timestamps[-1]:.3f}")
            
            # Analyze frequency consistency
            if len(timestamps) > 2:
                dt_values = np.diff(timestamps)
                print(f"   Time step statistics:")
                print(f"      Mean dt: {dt_values.mean():.4f}s ({1/dt_values.mean():.1f} Hz)")
                print(f"      Std dt: {dt_values.std():.4f}s")
                print(f"      Min dt: {dt_values.min():.4f}s")
                print(f"      Max dt: {dt_values.max():.4f}s")
    
    def display_camera_images(self, entry_index: int = 0):
        """Display camera images from a specific entry using cv2."""
        if not self.data or entry_index >= len(self.data):
            print("❌ No data or invalid entry index")
            return
        
        entry = self.data[entry_index]
        
        if 'image' not in entry:
            print("❌ No image data found in entry")
            return
            
        image_data = entry['image']
        if not isinstance(image_data, dict):
            print("❌ Image data is not in expected dictionary format")
            return
        
        print(f"\n📸 Displaying camera images from entry {entry_index}")
        print("Press any key to close each image window")
        
        for cam_name, cam_data in image_data.items():
            try:
                # Handle different image data structures
                img_array = None
                
                if isinstance(cam_data, np.ndarray):
                    img_array = cam_data
                elif isinstance(cam_data, dict):
                    # Look for common image stream keys
                    for stream_key in ['color', 'rgb', 'image']:
                        if stream_key in cam_data:
                            stream_data = cam_data[stream_key]
                            if isinstance(stream_data, np.ndarray):
                                img_array = stream_data
                                break
                            elif isinstance(stream_data, bytes):
                                # Decompress JPG bytes to numpy array
                                img_array = decompress_jpg_to_image(stream_data)
                                break
                elif isinstance(cam_data, bytes):
                    # Handle direct compressed image bytes
                    img_array = decompress_jpg_to_image(cam_data)
                
                if img_array is not None:
                    # Ensure the image is in the right format for cv2
                    if len(img_array.shape) == 3:
                        # Convert RGB to BGR for cv2 display if needed
                        if img_array.shape[2] == 3:
                            img_display = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        else:
                            img_display = img_array
                    else:
                        img_display = img_array
                    
                    # Create window and display image
                    window_name = f"Camera: {cam_name} (Entry {entry_index})"
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(window_name, img_display)
                    
                    print(f"  📷 {cam_name}: {img_array.shape} - Press any key to continue")
                    cv2.waitKey(0)
                    cv2.destroyWindow(window_name)
                else:
                    print(f"  ⚠️  {cam_name}: No displayable image data found")
                    
            except Exception as e:
                print(f"  ❌ Error displaying {cam_name}: {e}")
        
        # Clean up any remaining windows
        cv2.destroyAllWindows()
        print("✅ Image display complete")

    def show_sample_data(self, num_samples: int = 3):
        """Show sample data from first, middle, and last entries."""
        if not self.data:
            return
        
        print("\n" + "="*80)
        print(f"SAMPLE DATA (first, middle, and last entries)")
        print("="*80)
        
        # Determine which entries to show
        if len(self.data) == 1:
            sample_indices = [0]
        elif len(self.data) == 2:
            sample_indices = [0, 1]
        else:
            middle_idx = len(self.data) // 2
            sample_indices = [0, middle_idx, len(self.data) - 1]
        
        for i in sample_indices:
            print(f"\n📋 Entry {i} (of {len(self.data) - 1}):")
            entry = self.data[i]
            
            for key, value in entry.items():
                if key == 'timestamp':
                    print(f"   {key}: {value:.6f}")
                elif key == 'image' and isinstance(value, dict):
                    print(f"   {key}: dict with cameras: {list(value.keys())}")
                    for cam_name, cam_data in value.items():
                        if isinstance(cam_data, np.ndarray):
                            img_shape = cam_data.shape
                            print(f"      {cam_name}: {img_shape} (H×W×C: {img_shape[0]}×{img_shape[1]}×{img_shape[2] if len(img_shape) > 2 else 'N/A'})")
                        elif isinstance(cam_data, dict):
                            # Handle nested camera structure (e.g., {'color': image_data})
                            if any(isinstance(v, np.ndarray) for v in cam_data.values()):
                                print(f"      {cam_name}: dict with streams: {list(cam_data.keys())}")
                                for stream_type, stream_data in cam_data.items():
                                    if isinstance(stream_data, np.ndarray):
                                        img_shape = stream_data.shape
                                        print(f"        {stream_type}: {img_shape} (H×W×C: {img_shape[0]}×{img_shape[1]}×{img_shape[2] if len(img_shape) > 2 else 'N/A'})")
                                    elif isinstance(stream_data, bytes):
                                        print(f"        {stream_type}: bytes(length={len(stream_data)}, compressed_jpg)")
                                    elif stream_data is None:
                                        print(f"        {stream_type}: None")
                                    else:
                                        print(f"        {stream_type}: {type(stream_data).__name__}")
                            else:
                                # Handle dict that might contain other metadata
                                print(f"      {cam_name}: {self.get_data_type_info(cam_data)}")
                        elif cam_data is None:
                            print(f"      {cam_name}: None")
                        else:
                            print(f"      {cam_name}: {type(cam_data).__name__} - {self.get_data_type_info(cam_data)}")
                elif isinstance(value, dict):
                    print(f"   {key}: dict({list(value.keys())})")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            dimensions = f" dimensions: {sub_value.shape}"
                            print(f"      {sub_key}: {self.get_data_type_info(sub_value)}{dimensions}")
                        elif sub_value is None:
                            print(f"      {sub_key}: None")
                        elif isinstance(sub_value, list) and sub_value and isinstance(sub_value[0], (int, float)):
                            print(f"      {sub_key}: list(length={len(sub_value)}, values={sub_value})")
                        elif isinstance(sub_value, tuple):
                            if sub_value and isinstance(sub_value[0], (int, float)):
                                print(f"      {sub_key}: tuple(length={len(sub_value)}, values={sub_value})")
                            else:
                                print(f"      {sub_key}: {self.get_data_type_info(sub_value)}")
                        else:
                            print(f"      {sub_key}: {type(sub_value).__name__}")
                elif isinstance(value, np.ndarray):
                    dimensions = f" dimensions: {value.shape}"
                    print(f"   {key}: {self.get_data_type_info(value)}{dimensions}")
                elif isinstance(value, list):
                    if value and isinstance(value[0], (int, float)):
                        print(f"   {key}: list(length={len(value)}, values={value})")
                    else:
                        print(f"   {key}: {self.get_data_type_info(value)}")
                elif isinstance(value, tuple):
                    if value and isinstance(value[0], (int, float)):
                        print(f"   {key}: tuple(length={len(value)}, values={value})")
                    else:
                        print(f"   {key}: {self.get_data_type_info(value)}")
                else:
                    print(f"   {key}: {type(value).__name__}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        if not self.data:
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("="*80)
        
        print(f"📄 File: {self.pkl_file_path}")
        print(f"📊 Total entries: {len(self.data)}")
        print(f"💾 File size: {self.pkl_file_path.stat().st_size / (1024*1024):.2f} MB")
        
        if 'timestamp' in self.data[0] and len(self.data) > 1:
            duration = self.data[-1]['timestamp'] - self.data[0]['timestamp']
            print(f"⏱️  Duration: {duration:.2f} seconds")
            print(f"📈 Average logging frequency: {len(self.data) / duration:.1f} Hz")
        
        print(f"\n🔧 Data fields found:")
        for field, type_info in self.analysis_results.get('fields', {}).items():
            print(f"   • {field}: {type_info}")
        
        # Check for key teleoperation data types
        key_data_types = {
            'Robot Joint Data': ['qpos', 'qvel', 'qpos_des'],
            'Gripper Data': ['gripper_qpos', 'gripper_target', 'gripper_qpos_des'],
            'Camera Data': ['image', 'images', 'frames'],
            'Timing Data': ['timestamp']
        }
        
        print(f"\n📋 Data type summary:")
        for category, keys in key_data_types.items():
            found_keys = [key for key in keys if key in self.data[0]]
            if found_keys:
                print(f"   ✅ {category}: {found_keys}")
            else:
                print(f"   ❌ {category}: Not found")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("🤖 XRoboToolkit Teleop Data Log Analyzer")
        print("="*80)
        
        if not self.load_data():
            return False
        
        self.analyze_data_structure()
        self.analyze_robot_state_data()
        self.analyze_camera_data()
        self.analyze_timestamps()
        self.show_sample_data()
        self.generate_summary_report()
        
        # Display camera images from sample entries
        if 'image' in self.data[0]:
            print("\n" + "="*80)
            user_input = input("Display camera images from sample entries (first, middle, last)? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                # Use same logic as show_sample_data for entry selection
                if len(self.data) == 1:
                    sample_indices = [0]
                elif len(self.data) == 2:
                    sample_indices = [0, 1]
                else:
                    middle_idx = len(self.data) // 2
                    sample_indices = [0, middle_idx, len(self.data) - 1]
                
                for idx in sample_indices:
                    print(f"\n--- Displaying images from Entry {idx} (of {len(self.data) - 1}) ---")
                    self.display_camera_images(idx)
        
        return True


def find_first_pkl_file(logs_dir: str = "logs") -> Optional[str]:
    """Find the first .pkl file in the logs directory."""
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        print(f"❌ Logs directory '{logs_dir}' not found")
        return None
    
    # Search for .pkl files recursively
    pkl_files = list(logs_path.rglob("*.pkl"))
    
    if not pkl_files:
        print(f"❌ No .pkl files found in '{logs_dir}'")
        return None
    
    return str(pkl_files[0])


def main():
    """Main function to run the analysis."""
    # Determine which file to analyze
    if len(sys.argv) > 1:
        pkl_file_path = sys.argv[1]
    else:
        pkl_file_path = find_first_pkl_file()
        if not pkl_file_path:
            return 1
    
    # Check if file exists
    if not Path(pkl_file_path).exists():
        print(f"❌ File not found: {pkl_file_path}")
        return 1
    
    # Run analysis
    analyzer = TeleopDataAnalyzer(pkl_file_path)
    success = analyzer.run_full_analysis()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)