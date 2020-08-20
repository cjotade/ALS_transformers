""" from .GenerativeModel import GenerativeModel
from ..utils import do_parse_args
import socket

def server(model):
	HOST = "localhost"
	PORT = 6234

	s  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	input_address = (HOST, PORT)
	s.bind(input_address)
	s.listen(1)

	try: 
		while True:
			# Wait for a connection
			print('Waiting for a Request...')
			connection, client_address = s.accept()
			print('Request recived from %s:%d' % client_address)

			try:
				# Receive the lengh of the sentence request sended as byte(little)sequence and readed as int
				l = int.from_bytes(connection.recv(4), byteorder='little')
				print('String Length : {:d}'.format(l))

				# Receive the sentence
				sentence = str(connection.recv(l), 'utf-8')
				print(f'Sentence : {sentence}')

				# EXECUTE THE MODEL
				predicted_tokens = model.run(sentence)
				print(f'Predicted tokens:{predicted_tokens}')
				
				# Send how many predictions to client
				n_preds = len(predicted_tokens).to_bytes(4, byteorder='little')
				connection.send(n_preds)

				# For each predicted token send to client
				for pred_token in predicted_tokens:
					# Calculate the length of the string to send
					l_result = len(pred_token.encode()).to_bytes(4, byteorder='little')
					# Sending the length of the string to be sended
					connection.send(l_result)
					# Send the string
					connection.send(pred_token.encode())
				print("Tokens Predicted!")
				print()
			finally:
				# Clean up the connection
				connection.close()
	except KeyboardInterrupt:
		print('exiting.')
	finally:
		s.shutdown(socket.SHUT_RDWR)
		s.close()

if __name__ == "__main__":
	args = do_parse_args()
	model = GenerativeModel(args)
	server(model) """

import socketserver
#import socket
import json

response = {
  "sentences": ["John ef ","ann ere ","ssdaa erer", "saa er er"],
  "words": ["palabra1","palabra2","palabra3"],
  "error": False
}

class MyTCPHandler(socketserver.BaseRequestHandler):

	def handle(self):
		while True:
			# escucha longitud de mensaje
			self.msgl = self.request.recv(1024).strip().decode("utf-8")
			print("largo: ",self.msgl)
			
			# escucha mensaje
			self.msg = json.loads(self.request.recv(int(self.msgl)).strip().decode("utf-8"))
			print("{}:".format(self.client_address[0]))
			print(self.msg)

			# envia longitud de respuesta
			l = len(json.dumps(response))
			print("- l: ",bytes(str(l), 'utf-8'))
			self.request.send(bytes(str(l), 'utf-8'))

			# envia respuesta
			self.request.sendall(bytes(json.dumps(response), 'utf-8'))
	


if __name__ == "__main__":
	HOST, PORT = "192.168.1.199", 11000
	print(HOST)
	print(PORT)
	server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
	server.serve_forever()