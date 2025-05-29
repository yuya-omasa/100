import torch
from torch.utils.data import DataLoader
from config import *
from data import SSTDataset, collate_fn
from utils import load_pretrained_embeddings
from model import BoWClassifier
from train import train, evaluate
from model import CNNClassifier

def main():
    embedding_matrix, word2id, id2word = load_pretrained_embeddings("data/GoogleNews-vectors-negative300.bin", VOCAB_SIZE, EMBEDDING_DIM)

    train_dataset = SSTDataset("data/SST-2/train.tsv", word2id)
    dev_dataset = SSTDataset("data/SST-2/dev.tsv", word2id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    #model = BoWClassifier(embedding_matrix).to(DEVICE)
    
    model = CNNClassifier(embedding_matrix, freeze=False).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    train(model, train_loader, optimizer, criterion, DEVICE)
    evaluate(model, dev_loader, DEVICE)

if __name__ == "__main__":
    main()
