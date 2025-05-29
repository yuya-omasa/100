import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in tqdm(dataloader):
            inputs, labels = batch['input_ids'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['input_ids'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {correct / total:.4f}")
