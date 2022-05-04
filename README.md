# Movie-Recommender
A deep learning based movie recommender system.
# Deep Learning Based Movie Recommendation System

Get movie recommendations for specific user from recommendation model that trained on MovieLens-1M dataset.

## Requirements

First clone this repo and change directory into this folder. It is highly recommended to use the conda virtual environment to set up an environment with required libraries and dependencies. 

- For systems without GPU:
```bash
conda env create -f requirements_cpu.yaml
```
- For systems with GPU:
```bash
conda env create -f requirements_gpu.yaml
```

## How to Use?
To train with default parameters run;
```bash
python main.py
```

Additionaly you can play around with below arguments. If youset "wandb" argument as True,  [Wandb](https://wandb.ai)
```python
    arg.add_argument('--data_path', type=str, default=Path.home(), 
                    help='Path to the data directory')
    arg.add_argument('--data_url', type=str, default='http://files.grouplens.org/datasets/movielens/ml-1m.zip')
    arg.add_argument('--user_id', type=int, default=1, 
                    help='User ID to recommend movies for')
    arg.add_argument('--model_layers', type=list, default=[128 ,64, 32], 
                    help='Number of layers in each hidden layer with any desired depth')
    arg.add_argument('--epoch_number', type=int, default=50, 
                    help='Number of epochs to train')
    arg.add_argument('--batch_size', type=int, default=4096)
    arg.add_argument('--learning_rate', type=float, default=0.001)
    arg.add_argument('--weight_decay', type=float, default=0.001)
    arg.add_argument('--dropout_rate', type=float, default=0.5)
    arg.add_argument('--opt', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
    arg.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'reduce'])
    arg.add_argument('--wandb', type=bool, default=False, 
                    help='Whether to use wandb tool for hyperparameter tuning')
    arg.add_argument('--hp_run_number', type=int, default=200,
                    help='Number of hyperparameter tuning runs')
```
