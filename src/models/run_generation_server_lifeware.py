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