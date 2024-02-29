import logging
import os

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from model_utils import ImageDataset, embed_images_batch

with open("search_app/config.yaml", "r") as file:
    config = yaml.safe_load(file)

LOGGER = logging.getLogger()

if not os.path.exists(config["resources_dir"]):
    os.makedirs(config["resources_dir"])

if __name__ == "__main__":
    all_paths = []
    for subdir in config["raw_data"]["subdirs"]:
        files = os.listdir(os.path.join(config["raw_data"]["dir"], subdir))
        all_paths += [os.path.join(config["raw_data"]["dir"], subdir, file) for file in files if file.endswith("jpg")]
    mapping = {i: path for i, path in enumerate(all_paths)}
    LOGGER.info(f"Number of images: {len(all_paths)}")

    dataset = ImageDataset(image_paths=all_paths)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    LOGGER.info("Computing embeddings...")
    embeddings = embed_images_batch(dataloader)
    LOGGER.info(f"Embeddings computed. Size: {embeddings.shape}")

    normalized_embeddings = normalize(embeddings, norm="l2", axis=1)

    np.savez(os.path.join(config["resources_dir"], config["vectors_path"]), vectors=normalized_embeddings)
    joblib.dump(mapping, os.path.join(config["resources_dir"], config["mapping_path"]))
    LOGGER.info("Successful")
