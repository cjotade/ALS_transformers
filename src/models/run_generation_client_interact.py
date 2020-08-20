# import curses
# import curses.textpad
# import socket
# import sys

# def main(stdscr):
#     # Screen
#     stdscr = curses.initscr() 
#     curses.cbreak()
#     curses.noecho()
#     stdscr.clear()
#     stdscr.refresh()
#     # Main
#     sentence = ""
#     while True:
#         c = stdscr.getch()
#         if c == 27:
#             break
#         else:
#             if c == ord(" "):
#                 sentence += chr(c)
#                 results = send_and_receive_message(sentence)
#                 for i, result in enumerate(results):
#                     add_str = "{}\n".format(result)
#                     stdscr.addstr(5+i, 0, add_str)
#             else:
#                 if c == curses.KEY_ENTER:
#                     sentence = ""
#                 elif c == curses.KEY_BACKSPACE:
#                     sentence = sentence[:-1]
#                 else:
#                     sentence += chr(c)
#                 stdscr.addstr(0, 0, sentence)
#             stdscr.refresh()

# def send_and_receive_message(message):
#     try:
#         # Create a TCP/IP socket
#         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         sock.connect(server_address)
#         # Send data
#         message = str.encode(message)

#         # Send length message and message
#         l = len(message).to_bytes(4, byteorder='little')
#         sock.send(l)
#         sock.send(message)
        
#         # Retrieve how many predictions
#         n_preds = int.from_bytes(sock.recv(4), byteorder='little')

#         # Display results retrieved from server
#         results = []
#         for _ in range(n_preds):
#             l_result = int.from_bytes(sock.recv(4), byteorder='little')
#             data = sock.recv(l_result)
#             results.append(data)
#         sock.close()
#         return results
#     except:
#         sock.close()
        

# if __name__ == "__main__":
#     HOST = "localhost"  # The server's hostname or IP address
#     PORT = 6234        # The port used by the server

#     # Connect the socket to the port where the server is listening
#     server_address = (HOST, PORT)

#     curses.wrapper(main)
