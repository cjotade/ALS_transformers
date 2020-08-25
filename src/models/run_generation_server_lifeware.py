import socketserver
#import socket
import json
from .GenerativeModel import GenerativeModel
from ..utils import do_parse_args

response = {
  "sentences": [],
  "words": [],
  "response": "",
  "error": False
}
model_sentences = None
model_words = None

def mapping(language,context):

	return

def set_parameters(args,modelType,modelNameOrPath,length,numReturnSequences):
	args.model_type = modelType
	args.model_name_or_path = modelNameOrPath
	args.length = length
	args.num_return_sequences = numReturnSequences
	print('----')
	print('model_type =', args.model_type)
	print('model_name_or_path =', args.model_name_or_path)
	print('length =', args.length)
	print('num_return_sequences =', args.num_return_sequences)
	print('no_cuda =', args.no_cuda)
	return args

def get_sentences(text):
	return model_sentences.run(text)

def get_words(text):
	return model_words.run(text)

def set_model(model,args):

	return


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
			if "text" in self.msg.keys():
				response["sentences"] = get_sentences(self.msg["text"])
				response["words"] = get_words(self.msg["text"])
			else:
				print("no text")

			if "language" in self.msg.keys() and "context" in self.msg.keys():
				print("language and context")
			else:
				print("no language and context")
			# envia longitud de respuesta
			l = len(json.dumps(response))
			print("- l: ",bytes(str(l), 'utf-8'))
			self.request.send(bytes(str(l), 'utf-8'))

			# envia respuesta
			self.request.sendall(bytes(json.dumps(response), 'utf-8'))
	


if __name__ == "__main__":
	args = do_parse_args()
	# print('model_type =', args.model_type)
	# print('model_name_or_path =', args.model_name_or_path)
	# print('length =', args.length)
	# print('num_return_sequences =', args.num_return_sequences)
	# print('no_cuda =', args.no_cuda)
	
	args_sentences = set_parameters(args,"gpt2","gpt2",5,2)
	args_words = set_parameters(args,"gpt2","gpt2",4,1)
	




	model_sentences = GenerativeModel(args_sentences)
	model_words = GenerativeModel(args_words)
	HOST, PORT = "127.0.0.1", 11000
	print(HOST)
	print(PORT)
	server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
	server.serve_forever()