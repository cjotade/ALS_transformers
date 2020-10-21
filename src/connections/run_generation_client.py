import socket
import sys

HOST = "localhost"  # The server's hostname or IP address
PORT = 6234        # The port used by the server

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = (HOST, PORT)
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)

try:
    # Send data
    message = b'Tengo sed dame'
    print(f'Sending {message}')

    # Send length message and message
    l = len(message).to_bytes(4, byteorder='little')
    sock.send(l)
    sock.send(message)
    
    # Retrieve how many predictions
    n_preds = int.from_bytes(sock.recv(4), byteorder='little')

    # Display results retrieved from server
    for _ in range(n_preds):
        l_result = int.from_bytes(sock.recv(4), byteorder='little')
        data = sock.recv(l_result)
        print(f'Received {data}')

finally:
    print('closing socket')
    sock.close()