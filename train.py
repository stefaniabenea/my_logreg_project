import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from dataset import SimpleBinaryDataset
from model import LogisticRegressionModel
from utils import set_seed
import torch.nn as nn

def train_and_evaluate(batch_size, learning_rate):
    set_seed(42)

    dataset = SimpleBinaryDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = LogisticRegressionModel(input_dim=2)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(f"\nTraining with batch_size={batch_size}, learning_rate={learning_rate}")
    train_losses = []

    for epoch in range(1, 21):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_data.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch:02d} → Loss: {avg_loss:.4f}")

    plt.plot(train_losses, marker='o')
    plt.title(f"Loss over epochs (bs={batch_size}, lr={learning_rate})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            predictions = (outputs > 0.5).float()
            all_preds.append(predictions)
            all_labels.append(batch_labels.unsqueeze(1))
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy = (all_preds == all_labels).float().mean().item()

    print(f"Final Accuracy: {accuracy:.4f}")
    return model, accuracy

if __name__ == "__main__":
    batch_sizes = [2, 4, 10]
    learning_rates = [0.01, 0.1, 0.5]

    best_accuracy = 0.0
    best_model = None
    best_params = None

    for bs in batch_sizes:
        for lr in learning_rates:
            model, acc = train_and_evaluate(bs, lr)
            print(f"Batch size: {bs}, Learning rate: {lr:.3f} → Accuracy: {acc:.4f}")
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_params = (bs, lr)
                torch.save(best_model.state_dict(), "best_model.pt")
                print(f"Model saved as best_model.pt with accuracy {acc:.4f}")
