import torch
import torch.onnx
from torch import nn
from model import NeuralNetwork, device
from dataloader import test_dataloader, train_dataloader
from datetime import datetime

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5f}/{size:5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Get current timestamp formatted as YYYY-MM-DD-HHMM
timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')

# Create dummy input tensor based on your expected input shape
dummy_input = torch.randn(1, 1, 28, 28)  # Adjust dimensions as needed
model = model.cpu()

# Export to ONNX
torch.onnx.export(model,               # model being run
                dummy_input,          # model input (or a tuple for multiple inputs)
                f"model_{timestamp}.onnx",         # where to save the model
                export_params=True,   # store the trained parameter weights inside the model file
                opset_version=11,     # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=['input'],    # the model's input names
                output_names=['output'],  # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                            'output' : {0 : 'batch_size'}})

torch.save(model.state_dict(), f"model_{timestamp}.pth")
print(f"Saved PyTorch Model State to model_{timestamp}.onnx")
