import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, embedding_matrix: torch.Tensor, num_classes=1, freeze=False):
        super(CNNClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=0)

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=5, padding=2)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(3 * 100, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)   # (batch_size, embed_dim, seq_len)

        conv1_out = F.relu(self.conv1(embedded))  # (batch_size, out_channels, seq_len)
        conv2_out = F.relu(self.conv2(embedded))
        conv3_out = F.relu(self.conv3(embedded))

        pool1 = F.max_pool1d(conv1_out, kernel_size=conv1_out.shape[2]).squeeze(2)  # (batch_size, out_channels)
        pool2 = F.max_pool1d(conv2_out, kernel_size=conv2_out.shape[2]).squeeze(2)
        pool3 = F.max_pool1d(conv3_out, kernel_size=conv3_out.shape[2]).squeeze(2)

        concat = torch.cat([pool1, pool2, pool3], dim=1)  # (batch_size, out_channels * 3)
        dropped = self.dropout(concat)
        logits = self.fc(dropped)
        return logits


class BoWClassifier(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        mean_embeds = embeds.mean(dim=1)
        logits = self.linear(mean_embeds)
        return self.sigmoid(logits)
