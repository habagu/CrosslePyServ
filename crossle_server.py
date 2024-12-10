import socket
import struct
import json
import threading
import time
from PIL import Image
import io
import os

from analyze import analyze_image

received_images = []
open_sockets = []

def receive_data(client_socket, data_size):
    buffer = b""
    while len(buffer) < data_size:
        data = client_socket.recv(data_size - len(buffer))
        if not data:
            break
        buffer += data
    return buffer

def start_server():
    server_ip = socket.gethostbyname(socket.gethostname())  # Get the device's IP address
    server_port = 12345    # Port to bind to

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(1)
    print("Server listening on port", server_port)
    print("Server IP address:", server_ip)

    while True:
        client_socket, client_address = server_socket.accept()
        open_sockets.append(client_socket)
        print(f"Connection from {client_address}")
        
        try:
            image_name = f"received_image_from_{client_address}.png"
            received_images.append(image_name)
            # Receive Image Data
            image_size = struct.unpack(">I", receive_data(client_socket, 4))[0]
            image_bytes = receive_data(client_socket, image_size)
            image = Image.open(io.BytesIO(image_bytes))
            image.save(image_name)
            print("Image received and saved.")
            
            # Receive JSON Data
            json_size = struct.unpack(">I", receive_data(client_socket, 4))[0]
            json_bytes = receive_data(client_socket, json_size)
            json_data = json.loads(json_bytes.decode("utf-8"))
            print("JSON received:", json_data)
            
            processed_data = analyze_image(image_name, json_data, client_socket)
            
            send_response(client_socket, processed_data)
            
        except Exception as e:
            print("Error:", e)
        finally:
            open_sockets.remove(client_socket)
            client_socket.close()
            os.remove(image_name)
            received_images.remove(image_name)
            print(f"Deleted image file: {image_name}")

def send_response(client_socket, response_dict):
    print("Sending JSON:", json.dumps(response_dict))
    json_string = json.dumps(response_dict)
    json_string = json_string.replace('"', '')
    # Convert the dictionary to a JSON string and encode it
    response_bytes = json_string.encode("utf-8")
    
    # Send the length of the JSON string as a 4-byte integer
    client_socket.sendall(struct.pack(">I", len(response_bytes)))
    
    # Send the actual JSON string
    client_socket.sendall(response_bytes)

def send_status_updates(msg: str, percentage: int, client_socket):
    status = {
        "type": "status_update",
        "message": msg,
        "percentage": percentage
    }
    
    send_response(client_socket, str(status))
    

def process_data(image_path: str, data: dict, client_socket) -> str:
    send_status_updates("Processing data...", 0, client_socket)
    for i in range(11):
        send_status_updates(f"Processing data...", i * 10, client_socket)
        time.sleep(1)
    status = {
        "type": "finished",
        "data": {}
    }
    return str(status)

def handle_console_input():
    while True:
        command = input()
        if command.strip().lower() == "exit":
            print("Shutting down the server...")
            # Add any necessary cleanup code here
            for image in received_images:
                if os.path.exists(image):
                    os.remove(image)
                    print(f"Deleted image file: {image}")
            for sock in open_sockets:
                try:
                    sock.close()
                    print("Closed socket:", sock)
                except Exception as e:
                    print("Error closing socket:", e)
            os._exit(0)

if __name__ == "__main__":
    console_thread = threading.Thread(target=handle_console_input)
    console_thread.start()
    start_server()
    console_thread.join()
