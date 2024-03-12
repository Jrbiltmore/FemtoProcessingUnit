# fpu/src/hardware/Device_Drivers.py

import serial
import usb.core
import usb.util

class SerialDeviceDriver:
    """
    A simple serial device driver to handle serial communications.
    """
    def __init__(self, port, baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None

    def connect(self):
        """
        Establishes a serial connection to the device.
        """
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate)
            print(f"Connected to {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")

    def disconnect(self):
        """
        Closes the serial connection.
        """
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print(f"Disconnected from {self.port}.")

    def send_data(self, data):
        """
        Sends data over the serial connection.
        """
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.write(data.encode())
        else:
            print("Serial connection is not established.")

    def receive_data(self):
        """
        Receives data from the serial connection.
        """
        if self.serial_connection and self.serial_connection.is_open:
            return self.serial_connection.readline().decode().strip()
        else:
            print("Serial connection is not established.")
            return None

class USBDeviceDriver:
    """
    A driver to interact with USB devices.
    """
    def __init__(self, vid, pid):
        self.vid = vid
        self.pid = pid
        self.device = None

    def connect(self):
        """
        Finds and attaches to the USB device.
        """
        self.device = usb.core.find(idVendor=self.vid, idProduct=self.pid)
        if self.device is None:
            print("Device not found.")
        else:
            print(f"Device found: {self.device}")
            if self.device.is_kernel_driver_active(0):
                try:
                    self.device.detach_kernel_driver(0)
                    print("Kernel driver detached.")
                except usb.core.USBError as e:
                    print(f"Could not detach kernel driver: {e}")

    def disconnect(self):
        """
        Releases the USB device.
        """
        usb.util.dispose_resources(self.device)
        print("Device released.")

    def send_data(self, data):
        """
        Sends data to the USB device.
        """
        if self.device:
            # Example endpoint and data for demonstration
            endpoint = self.device[0][(0,0)][0]
            try:
                self.device.write(endpoint.bEndpointAddress, data, timeout=1000)
                print("Data sent to device.")
            except usb.core.USBError as e:
                print(f"Error sending data: {e}")
        else:
            print("USB device is not connected.")

    def receive_data(self):
        """
        Receives data from the USB device.
        """
        if self.device:
            # Example endpoint for demonstration
            endpoint = self.device[0][(0,0)][0]
            try:
                data = self.device.read(endpoint.bEndpointAddress, endpoint.wMaxPacketSize, timeout=1000)
                print("Data received from device.")
                return data
            except usb.core.USBError as e:
                print(f"Error receiving data: {e}")
                return None
        else:
            print("USB device is not connected.")
            return None

# Example usage
if __name__ == "__main__":
    # Serial device example
    serial_driver = SerialDeviceDriver(port='/dev/ttyUSB0', baudrate=115200)
    serial_driver.connect()
    serial_driver.send_data("Hello from serial!")
    response = serial_driver.receive_data()
    print(f"Received: {response}")
    serial_driver.disconnect()

    # USB device example
    usb_driver = USBDeviceDriver(vid=0x1234, pid=0x5678)
    usb_driver.connect()
    usb_driver.send_data(b'Hello from USB!')
    response = usb_driver.receive_data()
    print(f"Received: {response}")
    usb_driver.disconnect()
