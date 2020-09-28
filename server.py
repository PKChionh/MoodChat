from flask import Flask, jsonify, request, render_template
import json
import pandas as pd
from string import punctuation
import numpy as np
import torch
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim

from wtforms import Form, TextField, SubmitField, validators

app = Flask(__name__)

class Sen_Analy_LSTM(nn.Module):

  def __init__(self, n_vocab = 5401, n_embed = 50, n_hidden = 100, n_output = 1, n_layers = 2, drop_p = 0.8):
    super().__init__()

    self.n_vocab = n_vocab
    self.n_layers = n_layers
    self.n_hidden = n_hidden

    self.embedding = nn.Embedding(n_vocab, n_embed)
    self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first = True, dropout = drop_p)
    self.dropout = nn.Dropout(drop_p)
    self.fc = nn.Linear(n_hidden, n_output)
    self.sigmoid = nn.Sigmoid()

  def init_hidden (self, batch_size):
    device = "cpu"
    weights = next(self.parameters()).data
    h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
          weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))

    return h

  def forward (self, input_words):

    # Temporary fix: Unable to access batch_size in this method
    batch_size = 1

    embedded_words = self.embedding(input_words)
    lstm_out, h = self.lstm(embedded_words)
    lstm_out = self.dropout(lstm_out)
    lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)
    fc_out = self.fc(lstm_out)
    sigmoid_out = self.sigmoid(fc_out)
    sigmoid_out = sigmoid_out.view(batch_size, -1)
    sigmoid_last = sigmoid_out[:, -1]

    return sigmoid_last, h

# end of class

class InputForm(Form):

  userInput1 = TextField("Enter a sentence", validators=[validators.InputRequired()])

  submit = SubmitField("Enter")


def preprocess_review(review):

  with open('models/word_to_int_dict.json') as handle:
    word_to_int_dict = json.load(handle)

  review = review.translate(str.maketrans('', '', punctuation)).lower().rstrip()
      
  tokenized = word_tokenize(review)

  if len(tokenized) >= 50:
    review = tokenized[:50]
  else:
    review= ['0']*(50-len(tokenized)) + tokenized

  final = []

  for token in review:
    try:
      final.append(word_to_int_dict[token])
    except:
      final.append(word_to_int_dict[''])

  return final  


@app.route("/", methods=['GET', 'POST'])
def home2():

  form = InputForm(request.form)

  if request.method == 'POST' and form.validate():
    txtUserInput = request.form['userInput1']
    print(f"if: {txtUserInput}")

    if txtUserInput:
      batch_size = 1

      # Vocab = 5401
      # Embed = 50
      # Hidden = 100
      # Output = 1
      # Layers = 2
      model = Sen_Analy_LSTM(5401, 50, 100, 1, 2)

      model.load_state_dict(torch.load("models/model_nlp.pkl"))
      model.eval()
      words = np.array([preprocess_review(review=txtUserInput)])
      padded_words = torch.from_numpy(words)
      pred_loader = DataLoader(padded_words, batch_size = 1, shuffle = True)
      for x in pred_loader:
        output = model(x)[0].item()

      print(f"Prediction: {output}")
    else:
      print("No output")
  else:
    print("elif")

  return render_template('index.html', form=form)

@app.route("/temp", methods=['GET', 'POST'])
def home():

  form = InputForm(request.form)

  print("Home")

  if request.method == 'POST' and form.validate():
    txtUserInput = request.form['userInput1']

    if txtUserInput:
      #-- Original code: Get sentence input via JSON ---
      #  request_json = request.get_json()
      #  i = request_json['input']

      batch_size = 1

      # Vocab = 5401
      # Embed = 50
      # Hidden = 100
      # Output = 1
      # Layers = 2
      model = Sen_Analy_LSTM(5401, 50, 100, 1, 2)

      model.load_state_dict(torch.load("models/model_nlp.pkl"))
      model.eval()
      words = np.array([preprocess_review(review=i)])
      padded_words = torch.from_numpy(words)
      pred_loader = DataLoader(padded_words, batch_size = 1, shuffle = True)
      for x in pred_loader:
        output = model(x)[0].item()

      print(f"Output: {output}")
    else:
      print("No output")
  
  print("End home()")

  return render_template('index.html', form=form)
#  return "Hello"

@app.errorhandler(500)
def internal_error(error):
  return "500 error"

@app.errorhandler(404)
def not_found(error):
  return "404 error",404

if __name__ == "__main__":
  print("Running server...")
  app.run(host="0.0.0.0", port=80)

