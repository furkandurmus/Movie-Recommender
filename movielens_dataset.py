from torch.utils.data import Dataset, DataLoader


class MovieDataset(Dataset):
    """
    Dataloader for the MovieLens dataset

    Returns:
        user_id (int): user id
        movie_id (int): movie id
        rating (float): rating
        
    """
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
    
    def __len__(self):
        return self.users.shape[0]

    
    def __getitem__(self, index):
        user_id, movie_id = self.users[index], self.movies[index]
        return user_id, movie_id, self.ratings[index]