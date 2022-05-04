import wandb
import os
import copy
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import time
import torch
from torch import nn
from utils import create_dataset, prepare_data, EmbeddingsNet, recommend_movie
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
import torch.optim as optim



class run_wandb():
    """
     Hyperparameter tuning with wandb tool. parameters_dict is a dictionary that contains the hyperparameters.
     Tuning will be done with random search.
     """
    def __init__(self, args):
        self.args = args

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        wandb.login()

        self.metric = {
            'name': 'test_loss',
            'goal': 'minimize'
        }


        self.parameters_dict = {
            'optimizer': {'values': ['sgd', 'adam', 'rmsprop']},
            'learning_rate': {'values': [0.001, 0.0005, 0.0001, 0.00005]},
            'epochs': {'values': [50]},
            'model_layers': {'values': [[256, 128, 64, 32], [128, 64, 32], [64, 32]]},
            'scheduler': {'values': ['CosineAnnealingLR', 'ReduceLROnPlateau']},
            "batch_size": {'values': [512, 1024, 2048, 4096]},
            "weight_decay": {'values': [0., 1e-5, 1e-3]}

        }
        self.sweep_config = {'method': 'random', 'metric': self.metric, 'parameters': self.parameters_dict}

        self.sweep_id = wandb.sweep(self.sweep_config, project='Huawei')
        wandb.agent(self.sweep_id, function=self.train_model_wandb, count=args.hp_run_number)
    
    def train_model_wandb(self):
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
        with wandb.init() as run:
            config=wandb.config
            model = self.prepare_model(config.model_layers)
            best_loss=np.inf
            best_model_wts = copy.deepcopy(model.state_dict())
            optimizer, scheduler = self.build_optimizer(model, config.optimizer, config.learning_rate, config.scheduler, config.weight_decay)            
            criterion = nn.MSELoss()
            dataloaders = prepare_data(self.args, config.batch_size)
    

        
            for epoch in tqdm(range(config.epochs)):
                
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
                        
                        user = user.to(self.device)
                        movie = movie.to(self.device)
                        rating = rating.to(self.device)

                    
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
                    
                    if phase=="train":
                        train_loss = epoch_loss
                        wandb.log({'epoch': epoch, 'train loss': train_loss})
                    else:
                        test_loss = epoch_loss
                        wandb.log({'epoch': epoch, 'test loss': test_loss})



                    # update the lr based on the epoch loss
                    if phase == "test": 

                        # keep best model weights to use later on inference phase
                        if epoch_loss < best_loss:
                            best_model_wts = copy.deepcopy(model.state_dict())
                            best_epoch = epoch
                            best_epoch_loss = epoch_loss
                            print("Found a better model")

                        # lr = optimizer.param_groups[0]['lr']
                        scheduler.step(epoch_loss) 
                    

                    print('Epoch:\t  %d |Phase: \t %s | Loss:\t\t %.4f'
                            % (epoch, phase, epoch_loss))
    




    def build_optimizer(self, model, optimizer, learning_rate, scheduler, weight_decay):
        
        if optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer} not found")


        if scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
        elif scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", verbose=True)    
        else:
            raise ValueError(f"Scheduler {scheduler} not found")
        
        return optimizer, scheduler

    def prepare_model(self, model_layers):
        _, _, _, _, _, _, n_users, n_movies = create_dataset(self.args)

        return EmbeddingsNet(n_users, n_movies, model_layers)



    


    

