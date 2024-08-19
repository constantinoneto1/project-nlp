import nltk

from collections import Counter

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

import pickle
import re

from connection import MySQLCRUD

from model import iterate_minibatches, clean_text

vocab_size = 119968
embedding_dim = 100
hidden_dim = 128
output_dim = 5

TARGET_COLUMN = 'stars'
BATCH_SIZE = 128

class LSTMPredictorTest(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix = None):
        # Nesse caso o construtor não precisa dos pesos pre-treinados do GloVe, eles já estarão no state_dict salvo durante o treinamento
        super(LSTMPredictorTest, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size= hidden_dim, batch_first= True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x):
        text = x['text']

        text_emb = self.embedding(text)
        out_lstm, (hidden, cel) = self.lstm(text_emb)
        feat = hidden[-1]
        out_fc = self.fc(feat)
        # output = F.relu(out_fc)

        return out_fc


def calculate_metrics(model, token_to_id, data, device, batch_size= BATCH_SIZE, **kw):
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for batch in iterate_minibatches(data, token_to_id, batch_size= batch_size, shuffle= False, device= device, **kw):

            batch_pred = model(batch)
            batch_label = batch[TARGET_COLUMN]

            preds = torch.argmax(batch_pred, dim= 1)
            
            all_preds.append(preds)
            all_labels.append(batch_label)

    return torch.cat(all_preds), torch.cat(all_labels)


if __name__ == '__main__':

    dict_path = 'C:/Users/Rafael (Aluízio)/Documents/GitHub/project-nlp/token_to_id.txt'
    # json_path = 'D:/Documentos/Estudos/Projeto-NLP/dataset/yelp_academic_dataset_review.json'
    test_path = 'C:/Users/Rafael (Aluízio)/Documents/GitHub/project-nlp/dataset/test_set/test_set.csv'
    model_path = 'C:/Users/Rafael (Aluízio)/Documents/GitHub/project-nlp/model/lstm_model_30_epochV2.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_size = 120031
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 5

    with open(dict_path, 'rb') as file:
        token_to_id = pickle.loads(file.read())

    df_test = pd.read_csv(test_path, index_col= 0)

    df_test['text'] = df_test['text'].apply(clean_text)
    df_test = df_test.reset_index().drop('index', axis= 1)
    
    model = LSTMPredictorTest(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path))

    all_preds, all_labels = calculate_metrics(model, token_to_id, df_test, device, BATCH_SIZE)

    print(all_preds.shape)

    # torch.save(all_preds, f'C:/Users/Rafael (Aluízio)/Documents/GitHub/project-nlp/values/test_result_values/all-preds.pt')
    # torch.save(all_labels, f'C:/Users/Rafael (Aluízio)/Documents/GitHub/project-nlp/values/test_result_values/all-labels.pt')