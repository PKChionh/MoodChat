from flask import Flask, jsonify, request
import json
import pandas as pd
from string import punctuation
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
  return "Predict"

@app.route("/", methods=['GET', 'POST'])
def home():
  return "Hello"

@app.errorhandler(500)
def internal_error(error):
   return "500 error"

@app.errorhandler(404)
def not_found(error):
   return "404 error",404

if __name__ == "__main__":
  print("Running server...")
  app.run(host="0.0.0.0", port=80)

