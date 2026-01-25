import torch
import math
from pathlib import Path

# =============================
# Gaussian Naive Bayes Model
# =============================
class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    # -----------------------------
    # Fit Model
    # -----------------------------
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        self.classes = torch.unique(y)

        for cls in self.classes:
            X_c = X[y == cls]

            self.mean[cls.item()] = X_c.mean(dim=0)
            self.var[cls.item()] = X_c.var(dim=0) + 1e-6  # numerical stability
            self.priors[cls.item()] = X_c.size(0) / X.size(0)

    # -----------------------------
    # Gaussian Probability Density
    # -----------------------------
    def _gaussian_pdf(self, x, mean, var):
        exponent = torch.exp(-((x - mean) ** 2) / (2 * var))
        return exponent / torch.sqrt(2 * math.pi * var)

    # -----------------------------
    # Predict
    # -----------------------------
    def predict(self, X: torch.Tensor):
        predictions = []

        for x in X:
            posteriors = []

            for cls in self.classes:
                cls = cls.item()

                prior = torch.log(torch.tensor(self.priors[cls]))

                likelihood = torch.sum(
                    torch.log(self._gaussian_pdf(x, self.mean[cls], self.var[cls]))
                )

                posterior = prior + likelihood
                posteriors.append(posterior)

            predictions.append(self.classes[torch.argmax(torch.tensor(posteriors))])

        return torch.tensor(predictions)

    # -----------------------------
    # Save Model
    # -----------------------------
    def save(self, path: Path):
        torch.save({
            "classes": self.classes,
            "mean": self.mean,
            "var": self.var,
            "priors": self.priors
        }, path)

    # -----------------------------
    # Load Model
    # -----------------------------
    def load(self, path: Path):
        checkpoint = torch.load(path)
        self.classes = checkpoint["classes"]
        self.mean = checkpoint["mean"]
        self.var = checkpoint["var"]
        self.priors = checkpoint["priors"]


# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    torch.manual_seed(42)

    # -----------------------------
    # Create Dataset
    # -----------------------------
    X = torch.empty(200, 1).uniform_(-50, 50)
    y = torch.where(X > 5, 1, 0)  # Binary classification

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # -----------------------------
    # Train Model
    # -----------------------------
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).float().mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # -----------------------------
    # Save Model
    # -----------------------------
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "gaussian_naive_bayes.pth"
    model.save(model_path)

    print(f"✅ Model saved at: {model_path}")

    # -----------------------------
    # Inference
    # -----------------------------
    sample = torch.tensor([[4.0], [6.0]])
    preds = model.predict(sample)

    print("\nPredictions:")
    for x_val, p in zip(sample, preds):
        print(f"x={x_val.item():.1f}, class={int(p.item())}")
