import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
import pandas as pd

torch.manual_seed(42)

class ArtificialNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.net(x)

X = torch.randn(100, 3)
y = (X.sum(dim=1) > 0).float().unsqueeze(1)

Data = pd.DataFrame(X.numpy(), columns=['X1', 'X2', 'X3'])
Data['label'] = y.numpy()

X_train, X_test, y_train, y_test = train_test_split(
    Data[['X1', 'X2', 'X3']], Data['label'], test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test  = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

Model = ArtificialNeuralNetwork()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(Model.parameters(), lr=0.01)

epochs = 500

for epoch in range(epochs):
    Model.train()
    y_pred = Model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        Model.eval()
        with torch.inference_mode():
            test_pred = Model(X_test)
            test_loss = criterion(test_pred, y_test)
            acc = ((torch.sigmoid(test_pred) > 0.5) == y_test).float().mean()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Test Loss: {test_loss.item():.4f} | "
            f"Acc: {acc.item():.4f}"
        )

os.makedirs("models", exist_ok=True)
torch.save(Model.state_dict(), "models/ANN_model.pth")
print("\nModel saved to: models/ANN_model.pth")