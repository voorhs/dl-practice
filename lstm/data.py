import torch
from torch.utils.data import Dataset
import re
import os
from collections import defaultdict
import nltk
nltk.download('stopwords')

STOPWORDS = nltk.corpus.stopwords.words('english')


def tokenize(text):
    """
    html_cleaner is taken from https://stackoverflow.com/a/12982689/20471250
    :param str text: Input text 
    :return List[str]: List of words
    """
    # lowercase
    text = text.lower()

    # remove all html tags and entities such as &nsbm
    html_cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(html_cleaner, '', text)

    # remove all except latin characters
    non_char_cleaner = _cleaner = re.compile('[^a-z ]')
    text = re.sub(non_char_cleaner, '', text)

    # split by whitespace
    text = text.split()

    # remove stopwords
    text = [w for w in text if w not in STOPWORDS]

    return text


def to_tensor(obj):
  return torch.tensor(obj, dtype=torch.long)


class LargeMovieReviewDataset(Dataset):
    def __init__(self, data_path, vocab, max_len, pad_sos=False, pad_eos=False):
        """
        :param str data_path: Path to folder with one of the data splits (train or test)
        :param torchtext.vocab.Vocab vocab: dictionary with lookup_indices method
        :param int max_len: Maximum length of tokenized text
        :param bool pad_sos: If True pad sequence at the beginning with <sos> 
        :param bool pad_eos: If True pad sequence at the end with <eos>         
        """
        super().__init__()
        
        self.pad_sos = pad_sos
        if self.pad_sos:
            self.sos_id = vocab['<sos>']
        self.pad_eos = pad_eos
        if self.pad_eos:
            self.eos_id = vocab['<eos>']
        
        self.vocab = vocab
        self.max_len = max_len
        self.data_path = data_path
        self.negative_path = os.path.join(data_path, 'neg')
        self.positive_path = os.path.join(data_path, 'pos')
        
        self.negative_paths = []
        self.positive_paths = []

        for file_path in os.listdir(self.negative_path):
            self.negative_paths.append(os.path.join(self.negative_path, file_path))

        for file_path in os.listdir(self.positive_path):
            self.positive_paths.append(os.path.join(self.positive_path, file_path))
        
        self.texts = []
        self.tokens = []
        self.ratings = []
        self.labels = [0] * len(self.negative_paths) + [1] * len(self.positive_paths)
        
        for path in self.negative_paths + self.positive_paths:
            # read file
            text = open(path, 'r', encoding='utf-8', errors='ignore').read().strip()
            self.texts.append(text)
            
            # convert to list of token ids
            token_ids = self.vocab.lookup_indices(tokenize(text))
            self.tokens.append(token_ids[:self.max_len])

            # extract rating by splitting path by / and _ and .
            # because typical path looks like '/content/aclImdb/train/neg/10003_1.txt')
            rating = re.split(r'/|_|\.', path)[-2]

            # -1 because we need 0..9, not 1..10
            self.ratings.append(int(rating) - 1)
        
    def __getitem__(self, idx):
        """
        :param int idx: index of object in dataset
        :return dict: Dictionary with all useful object data 
            {
                'text' str: unprocessed text,
                'label' torch.Tensor(dtype=torch.long): sentiment of the text (0 for negative, 1 for positive)
                'rating' torch.Tensor(dtype=torch.long): rating of the text
                'tokens' torch.Tensor(dtype=torch.long): tensor of tokens ids for the text
                'tokens_len' torch.Tensor(dtype=torch.long): number of tokens
            }
        """
        rating = to_tensor(self.ratings[idx])
        
        # pad with <sos> and <eos> if needed
        tokens = self.tokens[idx]
        if self.pad_sos:
          tokens = [self.sos_id] + tokens
        if self.pad_eos:
          tokens = tokens + [self.eos_id]
        
        # resulting dict ('text' and 'label' fields won't be used in training)
        res = {
            'text': self.texts[idx],
            # rating will be target for training
            'rating': rating,
            'label': rating > 5,
            # tokens will be fed to rnn
            'tokens': to_tensor(tokens),
            # tokens_len will be used for indexing
            # hidden state of last token which is not '<pad>'
            'tokens_len': to_tensor(len(tokens) - self.pad_sos - self.pad_eos)
        }
        return res
    
    def __len__(self):
        """
        :return int: number of objects in dataset 
        """
        return len(self.texts)


def collate_fn(batch, padding_value, batch_first=False):
    """
    :param List[Dict] batch: List of objects from dataset
    :param int padding_value: Value that will be used to pad tokens
    :param bool batch_first: If True resulting tensor with tokens must have shape [B, T] otherwise [T, B]
    :return dict: Dictionary with all data collated
        {
            'ratings' torch.Tensor(dtype=torch.long): rating of the text for each object in batch
            'labels' torch.Tensor(dtype=torch.long): sentiment of the text for each object in batch
            'texts' List[str]: All texts in one list
            'tokens' torch.Tensor(dtype=torch.long): tensor of tokens ids padded with @padding_value
            'tokens_lens' torch.Tensor(dtype=torch.long): number of tokens for each object in batch
        }
    """
    # to store result
    res = defaultdict(list)

    # collect from dicts to one dict
    for obj in batch:
      res['ratings'].append(obj['rating'])
      res['labels'].append(obj['label'])
      res['texts'].append(obj['text'])
      res['tokens'].append(obj['tokens'])
      res['tokens_lens'].append(obj['tokens_len'])
    
    # convert lists to tensors
    res['ratings'] = torch.stack(res['ratings'])
    res['labels'] = torch.stack(res['labels'])
    res['tokens_lens'] = torch.stack(res['tokens_lens'])
    
    # force all token sequences to have same length
    res['tokens'] = torch.nn.utils.rnn.pad_sequence(
        sequences=res['tokens'],
        batch_first=batch_first,
        padding_value=padding_value
    )

    return res