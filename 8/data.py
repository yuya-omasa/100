import torch
from torch.utils.data import Dataset
import csv

class SSTDataset(Dataset):
    def __init__(self, filepath, word2id):
        self.samples = []
        with open(filepath, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                tokens = row['sentence'].split()
                input_ids = [word2id.get(token) for token in tokens if token in word2id]
                if not input_ids:
                    continue
                label = torch.tensor([1.0]) if row['label'] == '1' else torch.tensor([0.0])
                self.samples.append({'text': row['sentence'], 'input_ids': torch.tensor(input_ids), 'label': label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['input_ids']), reverse=True)
    max_len = len(batch[0]['input_ids'])
    input_ids = [torch.cat([item['input_ids'], torch.zeros(max_len - len(item['input_ids']), dtype=torch.long)])
                 for item in batch]
    labels = [item['label'] for item in batch]
    return {'input_ids': torch.stack(input_ids).long(), 'label': torch.stack(labels)}
