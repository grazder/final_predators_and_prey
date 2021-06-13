import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from os import listdir
import numpy as np
from tqdm import tqdm


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_data_part(path: str):
    file_names = listdir(path)
    ids = np.array(list(set([int(file_name.split(":")[0]) for file_name in file_names])))

    indices = np.random.permutation(np.arange(len(ids)))
    ids = ids[indices]

    for id in ids:
        action_file_name = str(id) + ":actions.npy"
        state_file_name = str(id) + ":states.npy"

        X = torch.FloatTensor(np.load(f"{path}/{state_file_name}"))
        y = torch.FloatTensor(np.load(f"{path}/{action_file_name}"))

        yield X, y


def train(model: ActorNetwork, num_epochs: int, data_path: str, save_path: str):
    optimizer = Adam(params=model.parameters(), weight_decay=1e-4)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.
        num_parts = 0

        for X, y in tqdm(get_data_part(data_path)):

            num_batches = 0
            batch_total_loss = 0.
            train_dataloader = DataLoader(TensorDataset(X, y), batch_size=512, shuffle=True)

            for X_batch, y_batch in train_dataloader:
                optimizer.zero_grad()

                logits = model(X_batch)
                loss = mse_loss(logits, y_batch)
                loss.backward()
                optimizer.step()

                num_batches += 1
                batch_total_loss += loss.item()

            total_loss += (batch_total_loss / num_batches)
            num_parts += 1

        print(f"Epoch: {epoch}")
        print(f"Average MSE loss: {round(total_loss / num_parts, 4)}")
        torch.save(model.state_dict(), save_path)

SAVE_SIZE = 100_000
PREY_FEATURE_SIZE = 42
PREDATOR_FEATURE_SIZE = 52
HIDDEN_DIM = 64
NUM_EPOCHS = 50
DATA_PATH = 'imitation/prey'
SAVE_PATH = 'imitation/prey_model.pkl'

if __name__ == "__main__":
    model = ActorNetwork(PREY_FEATURE_SIZE, HIDDEN_DIM)
    train(model, num_epochs=NUM_EPOCHS, data_path=DATA_PATH, save_path=SAVE_PATH)
    print(f"Model successfully saved!")