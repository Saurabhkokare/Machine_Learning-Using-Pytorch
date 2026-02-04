import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# MODEL
# -------------------------------------------------
class ArtificialNeuralNetwork(nn.Module):
    def __init__(self, input_features, hidden_units, output_features):
        super().__init__()
        self.model_ = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )

    def forward(self, x):
        return self.model_(x)

# -------------------------------------------------
# DATA
# -------------------------------------------------
NUM_FEATURES = 3
NUM_CLASSES = 3

X, y = make_blobs(
    n_samples=1000,
    centers=NUM_CLASSES,
    n_features=NUM_FEATURES,
    random_state=42
)

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# TRAINING SETUP
# -------------------------------------------------
epochs = 100
model = ArtificialNeuralNetwork(
    input_features=NUM_FEATURES,
    hidden_units=8,
    output_features=NUM_CLASSES
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# -------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------
for epoch in range(epochs):
    # ---- Train ----
    model.train()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # ---- Evaluation ----
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_loss = loss_fn(test_logits, y_test)

            train_acc = (logits.argmax(dim=1) == y_train).float().mean()
            test_acc = (test_logits.argmax(dim=1) == y_test).float().mean()

        print(
            f"Epoch {epoch:04d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss.item():.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
torch.save(model.state_dict(), "models/ANN_multiclass_model.pth")
print("✅ Model saved")
