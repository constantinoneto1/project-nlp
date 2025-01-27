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

TARGET_COLUMN = 'stars'
device = "cuda" if torch.cuda.is_available() else "cpu"
UNK, PAD = "UNK", "PAD"
UNK_IX = 0
PAD_IX = 1

data_sol = False

MODEL_PATH = 'C:/Users/Rafael (Aluízio)/Documents/GitHub/project-nlp/model/'

BATCH_SIZE = 128
EPOCHS = 10

class LSTMPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix, dropout= 0.3):
        super(LSTMPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype= torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_size= hidden_dim, batch_first= True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        text = x['text']

        text_emb = self.embedding(text)
        out_lstm, (hidden, cel) = self.lstm(text_emb)
        feat = self.dropout(hidden[-1])
        
        out_fc = self.fc(feat)

        # output = F.relu(out_fc)

        return out_fc

def print_metrics(model, data, batch_size=BATCH_SIZE, name="", token_to_id= None, device=torch.device('cpu'), criterion= None, **kw):
    total_loss = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in iterate_minibatches(data, token_to_id, batch_size=batch_size, shuffle=False, device=device, **kw):
            batch_pred = model(batch)
            loss = criterion(batch_pred, batch[TARGET_COLUMN])
            total_loss += loss.item() * batch[TARGET_COLUMN].size(0)
            total += batch[TARGET_COLUMN].size(0)
            # squared_error += torch.sum(torch.square(batch_pred - batch[TARGET_COLUMN]))
            # abs_error += torch.sum(torch.abs(batch_pred - batch[TARGET_COLUMN]))

            # Corrigido
            # squared_error += torch.sum(torch.square(batch_pred.squeeze() - batch[TARGET_COLUMN]))
            # abs_error += torch.sum(torch.abs(batch_pred.squeeze() - batch[TARGET_COLUMN]))
    avg_loss = total_loss / total
    print("Cross-entropy loss: %.5f" % avg_loss)

    return avg_loss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time the validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def to_tensors(batch, device):
    batch_tensors = dict()
    for key, arr in batch.items():
        if key == 'text':
            batch_tensors[key] = torch.tensor(arr, device= device, dtype= torch.int64)
        else:
            batch_tensors[key] = torch.tensor(arr, device= device, dtype= torch.long)

    return batch_tensors

def iterate_minibatches(data, token_to_id, batch_size=256, shuffle=True, cycle=False, device=device, **kwargs):
    """ iterates minibatches of data in random order """
    while True:
        indices = np.arange(len(data))
        if shuffle:
            indices = np.random.permutation(indices)

        for start in range(0, len(indices), batch_size):
            batch = make_batch(data.iloc[indices[start : start + batch_size]], token_to_id, device=device, **kwargs)
            yield batch
        
        if not cycle: break

def apply_word_dropout(matrix, keep_prop, replace_with= UNK_IX, pad_ix= PAD_IX,):
    dropout_mask = np.random.choice(2, np.shape(matrix), p=[keep_prop, 1 - keep_prop])
    dropout_mask &= matrix != pad_ix
    return np.choose(dropout_mask, [matrix, np.full_like(matrix, replace_with)])

def make_batch(data, token_to_id, max_len=None, word_dropout=0, device=device):
    """
    Creates a keras-friendly dict from the batch data.
    :param word_dropout: replaces token index with UNK_IX with this probability
    :returns: a dict with {'title' : int64[batch, title_max_len]
    """
    batch = {}
    batch["text"] = as_matrix(data["text"].values, token_to_id, max_len)
    
    if word_dropout != 0:
        batch["text"] = apply_word_dropout(batch["text"], 1. - word_dropout)
    
    if TARGET_COLUMN in data.columns:
        batch[TARGET_COLUMN] = data[TARGET_COLUMN].values - 1
    
    return to_tensors(batch, device)

def as_matrix(sequences, token_to_id, max_len=None, UNK_IX= 0, PAD_IX= 1):

    """ Convert a list of tokens into a matrix with padding """
    if isinstance(sequences[0], str):
        sequences = list(map(str.split, sequences))

    max_len = min(max(map(len, sequences)), max_len or float('inf'))
    
    matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))
    for i,seq in enumerate(sequences):
        row_ix = [token_to_id.get(word, UNK_IX) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix
    
    return matrix

def make_vocab(data, tokenizer, embedding_dim):

    glove_path = 'C:/Users/Rafael (Aluízio)/Documents/GitHub/project-nlp/dataset/glove.twitter.27B.100d.txt'

    all_text = ' '.join(data['text'].values)
    all_tokens = tokenizer.tokenize(all_text)
    token_counts = Counter(all_tokens)

    tokens =  sorted(t for t, c in token_counts.items() if c >= 10)
    tokens = [UNK, PAD] + tokens

    token_to_id = {k: i for i, k in enumerate(tokens)}

    embedding_index = {}

    with open(glove_path, 'r', encoding= 'utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in token_to_id:
                coefs = np.asarray(values[1:], dtype= 'float32')
                embedding_index[word] = coefs

    
    tokens_size = len(token_to_id)

    embedding_matrix = np.zeros((tokens_size, embedding_dim))

    UNK_VEC = np.array(list(embedding_index.values())).mean(axis= 0)
    PAD_VEC = np.zeros(100)

    embedding_matrix[0] = UNK_VEC
    embedding_matrix[1] = PAD_VEC

    for word, i in token_to_id.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = UNK_VEC

    return tokens, token_to_id, embedding_matrix

def clean_text(text):
    text = re.sub(r"[^\w\s']", '', str(text))
    return ' '.join(nltk.tokenize.WordPunctTokenizer().tokenize(str(text))).lower()

if __name__ == '__main__':

    print(f'Rodando em {device}')
    config_path = 'D:/Documentos/Estudos/Projeto-NLP/config.json'
    json_path = 'C:/Users/Rafael (Aluízio)/Documents/GitHub/project-nlp/dataset/yelp_academic_dataset_review.json'

    model_name = input('Selecione o nome do modelo (final .pth): ')

    MODEL_PATH = MODEL_PATH + model_name

    val_loss = []

    if data_sol:
        db_conn = MySQLCRUD(config_path)
        table_name = 'YELP_REVIEW'

        len_dataset = db_conn.select_values(f"SELECT COUNT(*) FROM {table_name}")

        chunk_size = 10000

        df = pd.DataFrame()

        for offset in range(0, len_dataset, chunk_size):
            query = f'SELECT stars, text FROM YELP_REVIEW LIMIT {chunk_size} OFFSET {offset}'
            chunk = db_conn.select_values(query)
            df_chunks = pd.DataFrame(chunk, columns= ['stars', 'text'])
            df = pd.concat([df, df_chunks])
    else:
        df = pd.DataFrame()
        for chunk in pd.read_json(json_path, lines= True, chunksize= 100000):
            df = pd.concat([df, chunk[['stars', 'text']]])

    df['text'] = df['text'].apply(clean_text)

    # random_state para garantir reproducibilidade
    data_val, data_test = train_test_split(df, test_size= 0.1, random_state= 42)
    data_train, data_val = train_test_split(data_val, test_size= 0.222, random_state= 42)

    data_test.to_csv('C:/Users/Rafael (Aluízio)/Documents/GitHub/project-nlp/dataset/test_set/test_set.csv')

    tokenizer = nltk.tokenize.WordPunctTokenizer()
    
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 5

    tokens, token_to_id, embedding_matrix = make_vocab(data_train, tokenizer, embedding_dim)

    token_size = len(token_to_id)

    model = LSTMPredictor(token_size, embedding_dim, hidden_dim, output_dim, embedding_matrix).to(device)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= 1e-4)

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch}")
        model.train()
        for i, batch in tqdm(enumerate(
            iterate_minibatches(data_train, token_to_id= token_to_id, batch_size= BATCH_SIZE, device= device)),
            total = len(data_train) // BATCH_SIZE
        ):
            pred = model(batch)

            optimizer.zero_grad()

            loss = criterion(pred, batch[TARGET_COLUMN]) 
            loss.backward()
            optimizer.step()
        avg_loss = print_metrics(model, data_val, device= device, token_to_id= token_to_id, criterion= criterion)  
        val_loss.append(avg_loss)
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print('Early Stopping acionado!!')
            break

    try:
        with open('val_values.txt', 'wb') as file:
            pickle.dump(val_loss, file)
    except:
        pass

    torch.save(model.state_dict(), MODEL_PATH)