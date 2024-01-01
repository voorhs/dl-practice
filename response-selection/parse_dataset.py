import numpy as np
import json
from datasets import load_dataset
from tqdm import tqdm
import itertools as it


def dialogues_persona_chat(data, nickname):
    res = []
    conv_id = data['conv_id']
    history = data['history']
    for i, id in enumerate(tqdm(data['conv_id'][:-1], desc=f'Parsing Persona Chat {nickname}')):
        if id == conv_id[i + 1]:
            continue
        res.append(history[i])
    return res


def dialogues_imad(data, nickname):
    res = []
    dialogues = data['context']
    image_id = data['image_id']
    utter = data['utter']
    for ut, img_id, dia in tqdm(zip(utter, image_id, dialogues), desc=f'Parsing IMAD {nickname}'):
        res.append(dia+[(f'imad_images/{img_id}.jpg', ut)])
    return res


def parse(task):
    '''
    1. Load IMAD dataset
    2. Extract full dialogues ("train_dialogues.json")
    3. Compute run-length encodings of datasets ("train_rle.json")

    Params
    ------
        task: either 'imad' or 'persona_chat'
    '''
    directory = f'{task}_stuff'

    # extract full dialogues
    if task == 'imad':
        dialogues = dialogues_imad
        dataset = load_dataset("VityaVitalich/IMAD")
    elif task == 'persona_chat':
        dialogues = dialogues_persona_chat
        dataset = load_dataset('bavard/personachat_truecased')
        dataset['test'] = dataset['validation']
    else:
        raise ValueError(f'Not supported: {task}. Supported tasks: imad, persona_chat.')
    
    train_dialogues = dialogues(dataset['train'], 'train')
    test_dialogues = dialogues(dataset['test'], 'test')

    val_dialogues = train_dialogues[-len(test_dialogues):]
    train_dialogues = train_dialogues[:-len(test_dialogues)]

    # these will serve as convenient presentation of dataset
    json.dump(train_dialogues, open(f'{directory}/train_dialogues.json', 'w'))
    json.dump(test_dialogues, open(f'{directory}/test_dialogues.json', 'w'))
    json.dump(val_dialogues, open(f'{directory}/val_dialogues.json', 'w'))

    train_utterances = list(it.chain.from_iterable(train_dialogues))
    test_utterances = list(it.chain.from_iterable(test_dialogues))
    val_utterances = list(it.chain.from_iterable(val_dialogues))

    # these will be used in sentence_encoding.py (for further clustering) and in my_dff.py (for responding)
    json.dump(train_utterances, open(f'{directory}/train_utterances.json', 'w'))
    json.dump(test_utterances, open(f'{directory}/test_utterances.json', 'w'))
    json.dump(val_utterances, open(f'{directory}/val_utterances.json', 'w'))

    # run-length encode dialogues
    def rle(dialogues, nickname):
        res = []
        for dia in tqdm(dialogues, desc=f'RLE for {nickname}'):
            res.append(len(dia))
        return res
    
    train_rle = rle(train_dialogues, 'train')
    test_rle = rle(test_dialogues, 'test')
    val_rle = rle(val_dialogues, 'val')

    # these will be used for making contexts (for w2v in dgac_clustering.py and for catboost in linking_prediction.py)
    json.dump(train_rle, open(f'{directory}/train_rle.json', 'w'))
    json.dump(test_rle, open(f'{directory}/test_rle.json', 'w'))
    json.dump(val_rle, open(f'{directory}/val_rle.json', 'w'))


if __name__ == '__main__':
    import argparse
    
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', dest='task', required=True)
    args = ap.parse_args()

    parse(args.task)
