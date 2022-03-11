import os
import traceback
import sys
from flask import Flask, jsonify, request
from executors.campus_inferrer import CampusInferrer

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

app = Flask(__name__)

APP_ROOT = os.getenv('APP_ROOT', '/infer')
HOST = "127.0.0.1"
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 5000))

c_net = CampusInferrer()


@app.route(APP_ROOT, methods=["POST"])
def infer():
    data = request.json
    image = data['image']
    c_net.Load_classes()
    c_net.inference_engine()
    return c_net.final_prediction(image)


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == '__main__':
    app.run(host=HOST, port=PORT_NUMBER)
