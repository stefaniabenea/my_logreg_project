
# Simple Logistic Regression with PyTorch

This project demonstrates a basic implementation of logistic regression using PyTorch. The goal is to classify simple 2D points into two classes.

---

## Features

- Custom `Dataset` for simple 2D points and binary labels
- Logistic regression model implemented as a PyTorch `nn.Module`
- Training loop with batch processing, loss calculation, and parameter updates
- Model evaluation on a test set with accuracy metric
- Saving the best model based on accuracy
- Reproducibility using fixed random seeds
- Visualization of loss over epochs with matplotlib
- Experimentation with different batch sizes and learning rates

---

### Requirements

- Python 3.7+
- PyTorch
- matplotlib
- numpy

Install dependencies via:

```bash
pip install -r requirements.txt
```

### Running Training

To train the model with different hyperparameters:

```bash
python train.py
```

Sample output:

```
Training with batch_size=2, learning_rate=0.01
Epoch 01 → Loss: 1.4102
Epoch 02 → Loss: 1.1314
Epoch 03 → Loss: 0.9609
...
Epoch 20 → Loss: 0.5271
Final Accuracy: 1.0000
Batch size: 2, Learning rate: 0.010 → Accuracy: 1.0000
Model saved as best_model.pt with accuracy 1.0000
```

### Loading and Predicting

To load the saved model and make predictions:

```bash
python predict.py
```

---

## Project Structure

```
my_logreg_project/
├── model.py           # Model definition
├── dataset.py         # Dataset class
├── train.py           # Training and evaluation script
├── predict.py         # Model loading and prediction script
├── requirements.txt   # Project dependencies
├── .gitignore         # Git ignore file
└── README.md          # This file
```

---

