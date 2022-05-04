import io
import os
import zipfile
from textwrap import wrap
from pathlib import Path
from itertools import zip_longest
from collections import defaultdict
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse

from tqdm import tqdm
import time

from movielens_dataset import MovieDataset

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.optim import lr_scheduler 
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.optim as optim


def download_data(args):
    """
    Download the data as zip format from the url and save it to the download_path
    Args:
        url (string): url of the dataset
        download_path (string): Path object
    """

    try:
        r = urlopen(args.data_url)
    except URLError as e:
        print(f'Cannot download the data. Error: {e}')
        return
    assert r.status == 200
    data = r.read()

    with zipfile.ZipFile(io.BytesIO(data)) as arch:
        arch.extractall(args.data_path)

    print(f'Data is downloaded to: {args.data_path}')




def create_dataset(args):
    """
    Create ratings and movie dataframes from dataset

    Args:
        args (data_url): get folder name from url
        args (data_path): read files from data_path

    Returns:
        ratings (DataFrame): user ratings
        movies (DataFrame): movie information
        X (DataFrame): user-movie matrix
        y (DataFrame): ratings
        user_to_index (dict): user id to index
        movie_to_index (dict): movie id to index
        n_users (int): number of users
        n_movies (int): number of movies

    """
    archive_name = os.path.basename(args.data_url)
    folder_name, _ = os.path.splitext(archive_name)
    path = args.data_path / folder_name
    files = {}
    for filename in path.glob('*'):
        if filename.suffix == '.csv':
            files[filename.stem] = pd.read_csv(filename)
        elif filename.suffix == '.dat':
            if filename.stem == 'ratings':
                columns = ['userId', 'movieId', 'rating', 'timestamp']
            else:
                columns = ['movieId', 'title', 'genres']
            data = pd.read_csv(filename, sep='::', names=columns, engine='python', encoding='latin-1')
            files[filename.stem] = data
    
    ratings, movies = files['ratings'], files['movies']

    unique_users = ratings.userId.unique()
    unique_movies = ratings.movieId.unique()
 
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)
    
    movie_to_index = {old: new for new, old in enumerate(unique_movies)}
    new_movies = ratings.movieId.map(movie_to_index)
    
    n_users = unique_users.shape[0]
    n_movies = unique_movies.shape[0]
    X = pd.DataFrame({'user_id': new_users, 'movie_id': new_movies})
    y = ratings['rating'].astype(np.float32)

    return ratings, movies, X, y, user_to_index, movie_to_index, n_users, n_movies


def prepare_data(args, bs=None):

    _, _, X, y, _, _, _, _ = create_dataset(args)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    
    print(f'Train Data : {X_train.shape}')
    print(f'Test Data : {X_valid.shape}')
    
    train_set = MovieDataset(X_train["user_id"], X_train["movie_id"], y_train)
    test_set = MovieDataset(X_valid["user_id"], X_valid["movie_id"], y_valid)

    if args.wandb:
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    return {"train":train_loader, "test":test_loader, "dataset_size":{"train":len(train_set), "test":(len(test_set))}}





class EmbeddingsNet(nn.Module):
    """
    
    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, n_users, n_movies, layers, dropout=0.5):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, layers[0])
        self.movie_embed = nn.Embedding(n_movies, layers[0])
        self.layers = nn.ModuleList([nn.Linear(n_in*2, n_out*2) for n_in, n_out in zip(layers[:-1], layers[1:])])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(layers[-1]*2, 1)
    
    def forward(self, users, movies):

        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        x = torch.cat([user_embeds, movie_embeds], dim=1)

        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output(x)
        return x


def fetch_optimizer(model, args):
    
    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
   
    return optimizer

def fetch_scheduler(optimizer, args):

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
    elif args.scheduler == "reduce":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
    else:
        raise ValueError(f"Scheduler {args.scheduler} not found")
    return scheduler






def recommend_movie(args):
    """
    Args:
        args (user_id): user to be recommended 5 films
        

    Returns:
        list: a list of 5 movies with highest predicted rating
    """

    user_id = args.user_id
    ratings, movies, X, y, user_to_index, movie_to_index, n_users, n_movies = create_dataset(args)

    unique_users = ratings.userId.unique()
    unique_movies = ratings.movieId.unique()
    
    #movies watched by the user
    usr_watched_film_ids = X["movie_id"][X["user_id"]==user_id]
    
    #movie ids
    movie_real_ids=unique_movies[usr_watched_film_ids]
        
    #list of the films that user haven't watched
    usr_not_watched_film_ids = unique_movies[~(np.isin(unique_movies, movie_real_ids))]
    
    index_converter = lambda x: movie_to_index[x]
    
    #modele user id ve izlenmemiş film id yi fixlenmiş idler üzerinden vereceğimiz 
    # için real id bilgilerini tekrardan fixlenmiş id bilgisine geçtik
    user_not_watched_film_fixed_ids = [index_converter(i) for i in usr_not_watched_film_ids]
    
    #model inputs 
    user = torch.LongTensor(len(user_not_watched_film_fixed_ids)*[user_id])
    films_to_rate = torch.LongTensor(user_not_watched_film_fixed_ids)
    
    model = EmbeddingsNet(n_users, n_movies, args.model_layers)
    checkpoint = torch.load("best_epoch.pt")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    #model outputs
    predictions = model(user,films_to_rate)
    
    #sort the ratings
    v = torch.sort(predictions, 0)    
    
    #get the top 5
    recommended_ids = unique_movies[v[1][-5:]]
    recommended_movies = movies[movies["movieId"].isin(recommended_ids[:,0])]

    return recommended_movies