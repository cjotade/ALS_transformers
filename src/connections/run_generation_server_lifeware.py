import socketserver
import socket
import json
from ..models import GenerativeModel
from ..utils import do_parse_args
import copy

model_sentences = None
model_words = None
args = do_parse_args()

class MyTCPHandler(socketserver.BaseRequestHandler):

	def setup(self):
		self.language = None
		self.context = None
		self.model_sentences = None
		self.model_words = None
		self.args_sentences = copy.copy(args)
		self.args_words = copy.copy(args)
	
	def mapping(self):
		#defaul
		mt_sentences = "gpt2"
		mt_words = "bert"
		mop_sentences = "gpt2"
		mop_words = "bert-base-uncased"
		if self.language == "EnglishUS":
			mop_sentences = "distilgpt2"
			mop_words = "bert-base-uncased"
			if self.context == "papers":
				mop_sentences = "weights/gpt2-en/EN_papers_distil"
		if self.language == "SpanishSpain":
			mop_sentences = "distilgpt2" # cambiar por peso en español
			mop_words = "weights/bert-es/colloquial"
			if self.context == "lessons":
				mop_sentences = "weights/gpt2-es/ES_clases_distil"
			if self.context == "formal":
				mop_sentences = "weights/gpt2-es/ES_formal_distil"
		return mt_sentences, mt_words, mop_sentences, mop_words

	def set_parameters(self, args, modelType, modelNameOrPath, length, numReturnSequences):
		args.model_type = modelType
		args.model_name_or_path = modelNameOrPath
		args.length = length
		args.num_return_sequences = numReturnSequences
		print(args)
		return args

	def set_models(self):
		mt_sentences, mt_words, mop_sentences, mop_words = self.mapping()
		self.args_sentences = self.set_parameters(args = self.args_sentences, modelType = mt_sentences, modelNameOrPath=mop_sentences, length=5, numReturnSequences=4)
		self.args_words = self.set_parameters(args = self.args_words, modelType = mt_words, modelNameOrPath=mop_words, length=1, numReturnSequences=4)
		self.model_sentences = GenerativeModel(self.args_sentences)
		self.model_words = GenerativeModel(self.args_words)

	def get_sentences(self, text):
		if self.model_sentences != None:
			return self.model_sentences.run(text)
		else:
			return []
	
	def get_words(self, text):
		if self.model_words != None:
			return self.model_words.run(text)
		else:
			return []

	def handle(self):
		print("conected")
		while True:
			print("---------------------------------------------------")
			response = {"sentences": [],"words": [],"error":False}
			# escucha mensaje
			req = self.request.recv(1024).strip()
			req = req.decode("utf-8")
			print(req)
			try:
				self.msg = json.loads(req)
			except:
				print("error decoder")
				response["error"] = True
				self.request.sendall(bytes(json.dumps(response), 'utf-8'))
				continue
			print("{}:".format(self.client_address[0]))
			print(self.msg)
			if "text" in self.msg.keys() and self.msg["text"] != None:
				response["sentences"] = self.get_sentences(self.msg["text"])
				response["words"] = self.get_words(self.msg["text"])
				self.request.sendall(bytes(json.dumps(response), 'utf-8'))
			if "language" in self.msg.keys() and "context" in self.msg.keys() and self.msg["language"] != None and self.msg["context"] != None:
				print("language and context")
				if self.msg["language"] != self.language or self.msg["context"] != self.context:
					self.language = self.msg["language"]
					self.context = self.msg["context"]
					print("Si se cambia el contexto")
					self.set_models()
				self.request.sendall(bytes(json.dumps(response), 'utf-8'))
			if "close" in self.msg.keys() and self.msg["close"] == True:
				print("close")
				response["error"] = False
				self.request.sendall(bytes(json.dumps(response), 'utf-8'))
				break
			print(response)
			print("**----------------------------------------------**")
	
	def finish(self):
		print("disconected")


if __name__ == "__main__":
	HOST, PORT = "127.0.0.1", 11000
	print(HOST)
	print(PORT)
	server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
	try:
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			print("StartServer")
			s.connect((HOST, 11001))
			s.sendall(b'serverStart')
	except Exception as e:
		print("no detecta cliente")
	server.serve_forever()
		

