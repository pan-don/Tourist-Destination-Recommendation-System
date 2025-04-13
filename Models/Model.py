import torch
from torch import nn

class NeuralCollaborativeModel(nn.Module):
    def __init__(self, num_users, num_places, embedding_size):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_size)
        self.place_embedding = nn.Embedding(num_embeddings=num_places, embedding_dim=embedding_size)
        
        self.fc1 = nn.Sequential(
            nn.Linear(embedding_size * 2, 128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.linear = nn.Linear(32, 1)
        
    def forward(self, user_idx, place_idx):
        user_vec = self.user_embedding(user_idx)
        place_vec = self.place_embedding(place_idx)
        
        x = torch.cat([user_vec, place_vec], dim=1)
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        output = self.linear(x3)
        return output