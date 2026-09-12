from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

# Protocol version
PROTOCOL_VERSION = 2.0

DYNAMIXEL_DEGREE_PER_UNIT = 0.0879
DEFAULT_DEVICE_NAME = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 4000000

# Control table addresses (common to most Dynamixel motors)
ADDR_LED_RED = 65
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132


class DynamixelController:
    """Generic Dynamixel motor controller for basic operations."""

    def __init__(
        self,
        device_name: str = DEFAULT_DEVICE_NAME,
        baudrate: int = DEFAULT_BAUDRATE,
        protocol_version: float = PROTOCOL_VERSION,
    ):
        self.device_name = device_name
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(protocol_version)
        self.is_closed = True

        # Open port
        if not self.port_handler.openPort():
            raise Exception(f"Failed to open port {device_name}")
        self.is_closed = False
        print(f"Successfully opened port {device_name}")

        # Set baudrate
        if not self.port_handler.setBaudRate(baudrate):
            self.close()
            raise Exception(f"Failed to set baudrate to {baudrate}")
        print(f"Successfully set baudrate to {baudrate}")

    def enableTorque(self, motor_id: int):
        """Enable torque for a specific motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Torque enable failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Torque enable error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}")

    def disableTorque(self, motor_id: int):
        """Disable torque for a specific motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 0
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Torque disable failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Torque disable error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}")

    def turnOnLED(self, motor_id: int):
        """Turn on LED for a specific motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, ADDR_LED_RED, 1)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"LED turn on failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"LED turn on error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}")

    def turnOffLED(self, motor_id: int):
        """Turn off LED for a specific motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, ADDR_LED_RED, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"LED turn off failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"LED turn off error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}")

    def setGoalPosition(self, motor_id: int, position: int):
        """Set goal position for a specific motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id, ADDR_GOAL_POSITION, position
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(
                f"Failed to write position for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
            return False
        elif dxl_error != 0:
            print(f"Error in writing position for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}")
            return False
        return True

    def getPresentPosition(self, motor_id: int):
        """Get current position of a specific motor."""
        dxl_present_position, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
            self.port_handler, motor_id, ADDR_PRESENT_POSITION
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to read position for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
            return None
        elif dxl_error != 0:
            print(f"Error in reading position for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}")
            return None
        return dxl_present_position

    def close(self):
        """Closes the port and performs cleanup."""
        if not self.is_closed:
            print("Closing Dynamixel controller...")
            try:
                if hasattr(self.port_handler, "is_using") and self.port_handler.is_using:
                    self.port_handler.closePort()
                    print("Dynamixel port closed.")
            except Exception as e:
                print(f"Error during port close: {e}")
            finally:
                self.is_closed = True

    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.close()
