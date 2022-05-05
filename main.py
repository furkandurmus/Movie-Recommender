# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 01:36:20 2022

@author: furkan
"""

import os
import copy
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import time
import torch
from torch import nn
from torch.optim import lr_scheduler 
from utils import download_data, create_dataset, prepare_data, EmbeddingsNet, fetch_optimizer, fetch_scheduler, recommend_movie
from torch.utils.tensorboard import SummaryWriter
from hp_tuner import run_wandb




timestr = time.strftime("%Y%m%d-%H%M%S")
output_directory = os.getcwd()
tensorboard_output = os.path.join(os.getcwd(), timestr)

#tensorboard will watch the train progress
writer = SummaryWriter(tensorboard_output)



device = 'cuda' if torch.cuda.is_available() else 'cpu'




def train_model(args):
    """
    Train the model by given arguments

    Args:
        args (data_url): link of the dataset
        args (model_layers): set hidden layers of the model
        args (optimizer): set optimizer
        args (scheduler): set scheduler
        args (epochs): set epochs
        args (batch_size): set batch size
        args (weight_decay): set weight decay
    Return: Saves the best weights with the lowest loss
    """
    model = EmbeddingsNet(n_users, n_movies, args.model_layers).to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=np.inf
    criterion = nn.MSELoss()
    optimizer = fetch_optimizer(model, args)
    scheduler = fetch_scheduler(optimizer, args)
    dataloaders = prepare_data(args)

    
    for epoch in tqdm(range(args.epoch_number)):
        
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
                dataloader = dataloaders["train"]
            
            else:
                model.eval()
                dataloader = dataloaders["test"] 
                
            
            
            running_loss = 0.0
        
            for user, movie, rating in tqdm(dataloader):
                user = user.type(torch.LongTensor)
                movie = movie.type(torch.LongTensor)
                rating = rating.view(-1,1)
                
                user = user.to(device)
                movie = movie.to(device)
                rating = rating.to(device)

            
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(user, movie)

                    loss = criterion(outputs, rating)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #backpropagate and get gradients 
                        loss.backward()
                        
                        #update weights with optimizer function
                        optimizer.step()

                # sum up each loss of batch
                running_loss += loss.item() * user.size(0)

            # divide by image size to obtain overall epoch loss
            epoch_loss = running_loss / dataloaders["dataset_size"][phase]
            writer.add_scalars("Loss", {phase:epoch_loss}, epoch)

             # update the lr based on the epoch loss
            if phase == "test": 

                # keep best model weights to use later on inference phase
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    best_epoch_loss = epoch_loss
                    print("Found a better model")

                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("LR", lr, epoch)
                scheduler.step(epoch_loss) 
            

            print('Epoch:\t  %d |Phase: \t %s | Loss:\t\t %.4f'
                      % (epoch, phase, epoch_loss))
    
    save_model(best_model_wts, best_epoch, best_epoch_loss)            

def save_model(model_weights,  best_epoch,  best_epoch_loss):

    torch.save({
        "epoch": best_epoch,
        "model_state_dict": model_weights,
        "loss": best_epoch_loss,
    },  output_directory + "/best_epoch.pt")


def calc_rmse(args):
    checkpoint = torch.load("best_epoch.pt")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ground_truth = []
    predictions = []
    with torch.no_grad():
        for user, movie, rating in dataloaders["test"]:
            user = user.type(torch.LongTensor)
            movie = movie.type(torch.LongTensor)
            rating = rating.view(-1,1)
            
            user = user.to(device)
            movie = movie.to(device)
            rating = rating.to(device)

            outputs = model(user, movie)
            ground_truth.extend(rating.tolist())
            predictions.extend(outputs.tolist())

    ground_truth = np.asarray(ground_truth).ravel()
    predictions = np.asarray(predictions).ravel()

    final_loss = np.sqrt(np.mean((predictions - ground_truth)**2))
    print(f'Final RMSE: {final_loss:.4f}')


writer.close()



if __name__ == '__main__':
    

    arg = argparse.ArgumentParser()
    arg.add_argument('--data_path', type=str, default=Path.home(), 
                    help='Path to the data directory')
    arg.add_argument('--data_url', type=str, default='http://files.grouplens.org/datasets/movielens/ml-1m.zip')
    arg.add_argument('--user_id', type=int, default=1, 
                    help='User ID to recommend movies for')
    arg.add_argument('--model_layers', type=list, default=[128 ,64, 32], 
                    help='Number of layers in each hidden layer')
    arg.add_argument('--epoch_number', type=int, default=2, 
                    help='Number of epochs to train')
    arg.add_argument('--batch_size', type=int, default=4096)
    arg.add_argument('--learning_rate', type=float, default=0.001)
    arg.add_argument('--weight_decay', type=float, default=0.001)
    arg.add_argument('--dropout_rate', type=float, default=0.5)
    arg.add_argument('--opt', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
    arg.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'reduce'])
    arg.add_argument('--wandb', type=bool, default=False, 
                    help='Whether to use wandb tool for hyperparameter tuning')
    arg.add_argument('--hp_run_number', type=int, default=2,
                    help='Number of hyperparameter tuning runs')
    
    args = arg.parse_args()
    download_data(args)
    ratings, movies, X, y, user_to_index, movie_to_index, n_users, n_movies = create_dataset(args)
    print(f'Movielens Dataset: {n_users} users, {n_movies} movies')
    model = EmbeddingsNet(n_users, n_movies, args.model_layers).to(device)
    dataloaders = prepare_data(args)


    #if wandb True, hyperparameter tuning will be started. To run training set "wandb" flag False
    if args.wandb:
        run_wandb(args)
    else:
        train_model(args)
    
    #Calculate RMSE metric that will be calculated with the best model weights.
    calc_rmse(args)
    
    recommended_movies = recommend_movie(args)
    print("------"*10)
    print(f"Recommended Top 5 Films for user {args.user_id}")
    print("\n")
    for i in recommended_movies.itertuples():
        print(i.title, ":", i.genres)
        print("\n")
    print("------"*10)
    


