from torch.utils.data import Dataset
import json

class CustomDataset(Dataset):
    def __init__(self, file_path=None):
        if file_path:
            self.data = self.load_jsonl(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample["text"]

    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line.strip())
                data.append(record)
        return data