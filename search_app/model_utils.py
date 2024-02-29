import logging

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

LOGGER = logging.getLogger()

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}")

clip_model, clip_transform = clip.load("ViT-B/32", device=device)
clip_model.eval()


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or clip_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        return img


def embed_images_batch(dataloader, model=None):
    model = model or clip_model
    embeddings = []
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)
            image_embs = model.encode_image(images)
            embeddings.append(image_embs.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    return embeddings


def embed_query(text, model=None, tokenize_fn=None):
    model = model or clip_model
    tokenize_fn = tokenize_fn or clip.tokenize
    with torch.no_grad():
        texts = tokenize_fn([text]).to(device)
        text_embs = model.encode_text(texts)
    return text_embs[0].detach().cpu().numpy()
