import os
import pickle
from datetime import datetime


class DataLogger:
    """
    A simple data logger that collects data entries and saves them to a pickle file.
    """

    def __init__(self, log_dir="logs"):
        """
        Initializes the logger.

        Args:
            log_dir (str): The directory where log files will be stored. If None, logging is disabled.
        """
        self.log_data = []
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.count = 0
        os.makedirs(log_dir, exist_ok=True)

    def add_entry(self, data_entry):
        """
        Adds a data entry to the log. A timestamp is automatically added.

        Args:
            data_entry (dict): A dictionary containing the data to log for the current timestep.
        """
        self.log_data.append(data_entry)

    def save(self):
        """
        Saves the collected log data to a pickle file.
        """
        if not self.log_data:
            print("No data to save.")
            return
        self.count += 1
        self.log_file = os.path.join(self.log_dir, f"teleop_log_{self.timestamp}_{self.count}.pkl")

        print(f"Saving {len(self.log_data)} data points to {self.log_file}...")
        try:
            with open(self.log_file, "wb") as f:
                pickle.dump(self.log_data, f)
            print(f"Data successfully saved to {self.log_file}")
        except IOError as e:
            print(f"Error saving data: {e}")

    def reset(self):
        """
        Resets the logger, clearing all collected data.
        """
        self.log_data = []
        self.log_file = None

        print("Logger has been reset.")
