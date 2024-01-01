import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import json
import argparse
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch


def sbert_encoder(utterances, directory, split):
    """
    Encodes `utterances` with sentence_transformers library and saves embeddings to .npy file.

    Params
    ------
        utterances: list[str], all utterances from dataset
        path: str, where to save .npy file
    """
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1').to('cuda')
    embeddings = model.encode(utterances, show_progress_bar=True)
    np.save(f'{directory}/{split}_embeddings', embeddings)

def clip_encoder(utterances, directory, split, image_batch_size=8, text_batch_size=32):
    """
    Encodes `utterances` with sentence_transformers library and saves embeddings to .npy file.

    Params
    ------
        utterances: list[str], all utterances from dataset
        path: str, where to save .npy file
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="clip_feature_extractor",
        model_type="ViT-B-16",
        is_eval=True,
        device=device
    )

    # divide images, texts and last utterances
    images = []
    texts = []
    last_utters = []
    is_image = []
    for i, ut in enumerate(utterances):
        if isinstance(ut, list):
            is_image.append(True)
            img, lut = ut
            images.append(Image.open(img).convert('RGB'))
            last_utters.append(lut)
        else:
            is_image.append(False)
            texts.append(ut)
    is_image = np.array(is_image)

    pbar = tqdm(total=len(images) * 2 + len(texts))

    # encode them all
    image_embeddings = []
    for i in range(0, len(images), image_batch_size):
        inp = images[i:i+image_batch_size]
        preproc = torch.stack(list(map(vis_processors['eval'], inp)))
        embs = model.extract_features({'image': preproc.to(device)})
        image_embeddings.append(embs.detach().cpu().numpy())
        pbar.update(len(inp))
    image_embeddings = np.concatenate(image_embeddings, axis=0)
    
    text_embeddings = []
    for i in range(0, len(texts), text_batch_size):
        inp = texts[i:i+text_batch_size]
        preproc = txt_processors['eval'](inp)
        embs = model.extract_features({'text_input': preproc})
        text_embeddings.append(embs.detach().cpu().numpy())
        pbar.update(len(inp))
    text_embeddings = np.concatenate(text_embeddings, axis=0)

    last_utters_embeddings = []
    for i in range(0, len(texts), text_batch_size):
        inp = last_utters[i:i+text_batch_size]
        preproc = txt_processors['eval'](inp)
        embs = model.extract_features({'text_input': preproc})
        last_utters_embeddings.append(embs.detach().cpu().numpy())
        pbar.update(len(inp))
    last_utters_embeddings = np.concatenate(last_utters_embeddings, axis=0)
    
    # these will be used in dgac_clustering.py
    embeddings = np.empty((len(utterances), text_embeddings.shape[1]))
    embeddings[is_image] = image_embeddings
    embeddings[~is_image] = text_embeddings
    np.save(f'{directory}/{split}_embeddings', embeddings)

    # these will be used in response_selection.py and my_dff.py
    embeddings[is_image] = last_utters_embeddings
    np.save(f'{directory}/{split}_response_embeddings', embeddings)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', dest='task', required=True)
    args = ap.parse_args()

    directory = f'{args.task}_stuff'

    # load json files
    train_utterances = json.load(open(f'{directory}/train_utterances.json', 'r'))
    test_utterances = json.load(open(f'{directory}/test_utterances.json', 'r'))
    val_utterances = json.load(open(f'{directory}/val_utterances.json', 'r'))

    if args.task == 'persona_chat':
        encoder = sbert_encoder
    elif args.task == 'imad':
        encoder = clip_encoder
    else:
        raise ValueError(f'not supported: {args.task}')
    
    encoder(train_utterances, directory, 'train')
    encoder(test_utterances, directory, 'test')
    encoder(val_utterances, directory, 'val')