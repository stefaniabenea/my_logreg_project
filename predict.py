import torch
from model import LogisticRegressionModel

def load_and_predict(path, inputs):
    model = LogisticRegressionModel(input_dim=2)
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        outputs = model(inputs_tensor)
        preds = (outputs > 0.5).float()
        print(f"Predictions for {inputs} → {preds.squeeze().tolist()}")

if __name__ == "__main__":
    test_inputs = [[2.0, 3.0], [9.0, 2.0]]
    load_and_predict("best_model.pt", test_inputs)
