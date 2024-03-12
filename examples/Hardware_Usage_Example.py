# fpu/examples/Hardware_Usage_Example.py

from fpu.src.hardware.Device_Drivers import SerialDeviceDriver, USBDeviceDriver

def serial_device_example():
    serial_port = '/dev/ttyUSB0'  # Example serial port, adjust according to your system
    baud_rate = 9600  # Common baud rate for serial communication
    
    # Initialize and connect to the serial device
    serial_device = SerialDeviceDriver(port=serial_port, baudrate=baud_rate)
    serial_device.connect()
    
    # Send data to the serial device
    data_to_send = "Hello, serial device!"
    serial_device.send_data(data_to_send)
    print(f"Sent data to the serial device: {data_to_send}")
    
    # Receive data from the serial device
    received_data = serial_device.receive_data()
    print(f"Received data from the serial device: {received_data}")
    
    # Disconnect from the serial device
    serial_device.disconnect()

def usb_device_example():
    vid = 0x1234  # Vendor ID, replace with your device's Vendor ID
    pid = 0x5678  # Product ID, replace with your device's Product ID
    
    # Initialize and connect to the USB device
    usb_device = USBDeviceDriver(vid=vid, pid=pid)
    usb_device.connect()
    
    # Send data to the USB device
    data_to_send = b"Hello, USB device!"
    usb_device.send_data(data_to_send)
    print(f"Sent data to the USB device.")
    
    # Receive data from the USB device (assuming the device sends data back)
    received_data = usb_device.receive_data()
    if received_data:
        print(f"Received data from the USB device: {received_data}")
    
    # Disconnect from the USB device
    usb_device.disconnect()

if __name__ == "__main__":
    serial_device_example()
    usb_device_example()
