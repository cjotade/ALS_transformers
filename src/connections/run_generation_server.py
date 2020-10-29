import logging
from ..models import GenerativeModel
from ..utils import do_parse_args
import socket

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
			logger.info('Waiting for a Request...')
			connection, client_address = s.accept()
			logger.info('Request recived from %s:%d' % client_address)

			try:
				# Receive the lengh of the sentence request sended as byte(little)sequence and readed as int
				l = int.from_bytes(connection.recv(4), byteorder='little')
				logger.info('String Length : {:d}'.format(l))

				# Receive the sentence
				sentence = str(connection.recv(l), 'utf-8')
				logger.info(f'Sentence : {sentence}')

				# EXECUTE THE MODEL
				predicted_tokens = model.run(sentence)
				logger.info(f'Predicted tokens:{predicted_tokens}')
				
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
				logger.info("Tokens Predicted!")
			finally:
				# Clean up the connection
				connection.close()
	except KeyboardInterrupt:
		logger.error('exiting.')
	finally:
		s.shutdown(socket.SHUT_RDWR)
		s.close()

if __name__ == "__main__":
	args = do_parse_args()
	model = GenerativeModel(args)
	server(model)
