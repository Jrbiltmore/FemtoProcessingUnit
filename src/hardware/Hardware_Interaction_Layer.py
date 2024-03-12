# fpu/src/hardware/Hardware_Interaction_Layer.py

import time
from .Device_Drivers import SerialDeviceDriver, USBDeviceDriver

class HardwareController:
    """
    HardwareController manages interactions with various hardware components,
    abstracting the complexities involved in communicating with different types of devices.
    """
    def __init__(self):
        # Initialize device drivers as needed
        self.serial_devices = {}  # Dictionary to hold serial devices
        self.usb_devices = {}     # Dictionary to hold USB devices

    def add_serial_device(self, device_name, port, baudrate=9600):
        """
        Adds a serial device to the controller.
        """
        if device_name not in self.serial_devices:
            self.serial_devices[device_name] = SerialDeviceDriver(port, baudrate)
            print(f"Serial device '{device_name}' added.")
        else:
            print(f"Serial device '{device_name}' already exists.")

    def add_usb_device(self, device_name, vid, pid):
        """
        Adds a USB device to the controller.
        """
        if device_name not in self.usb_devices:
            self.usb_devices[device_name] = USBDeviceDriver(vid, pid)
            print(f"USB device '{device_name}' added.")
        else:
            print(f"USB device '{device_name}' already exists.")

    def communicate_with_device(self, device_type, device_name, data_to_send=None):
        """
        Facilitates communication with a specified device. Can send data to or receive data from the device.
        """
        if device_type == "serial" and device_name in self.serial_devices:
            device = self.serial_devices[device_name]
            device.connect()
            if data_to_send:
                device.send_data(data_to_send)
                print(f"Data sent to {device_name}: {data_to_send}")
            received_data = device.receive_data()
            device.disconnect()
            return received_data
        elif device_type == "usb" and device_name in self.usb_devices:
            device = self.usb_devices[device_name]
            device.connect()
            if data_to_send:
                device.send_data(data_to_send)
                print(f"Data sent to {device_name}: {data_to_send}")
            received_data = device.receive_data()
            device.disconnect()
            return received_data
        else:
            print(f"Device '{device_name}' of type '{device_type}' not found.")
            return None

# Example usage
if __name__ == "__main__":
    hc = HardwareController()
    hc.add_serial_device("Serial1", "/dev/ttyUSB0", 115200)
    hc.add_usb_device("USB1", 0x1234, 0x5678)

    # Example of sending data to and receiving data from a serial device
    response = hc.communicate_with_device("serial", "Serial1", "Ping")
    print(f"Response from Serial1: {response}")

    # Example of sending data to and receiving data from a USB device
    response = hc.communicate_with_device("usb", "USB1", b"Hello")
    print(f"Response from USB1: {response}")
