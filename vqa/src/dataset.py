from torch.utils.data import Dataset
import json
from PIL import Image


class VQADataset(Dataset):
    def __init__(self, path, test=False):
        self.path = path
        self.test = test
        
        data = json.load(open(path, 'r'))
        self.questions = data['q']
        self.images = [Image.open(p) for p in data['p']]
        if not test:
            self.answers = data['a']
    
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        if self.test:
            return self.questions[i], self.images[i]
        return self.questions[i], self.answers[i], self.images[i]
