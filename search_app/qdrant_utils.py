import logging
import os

import numpy as np
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

with open("search_app/config.yaml", "r") as file:
    config = yaml.safe_load(file)

LOGGER = logging.getLogger()

client = QdrantClient("localhost", port=6333)


def qdrant_setup():
    client.delete_collection(config["qdrant"]["collection_name"])

    client.create_collection(
        collection_name=config["qdrant"]["collection_name"],
        vectors_config=VectorParams(size=config["qdrant"]["vector_size"], distance=Distance.COSINE),
    )

    LOGGER.info("Loading vectors")
    vectors = np.load(os.path.join(config["resources_dir"], config["vectors_path"]))["vectors"].tolist()

    LOGGER.info("Inserting vectors into Qdrant")
    qdrant_points = []
    for i, vector in enumerate(vectors):
        qdrant_points.append(PointStruct(id=i, vector=vector))
    client.upload_points(
        collection_name=config["qdrant"]["collection_name"],
        wait=True,
        points=qdrant_points,
    )

    LOGGER.info("Successful")


def qdrant_search(vector, limit=5):
    search_result = client.search(collection_name=config["qdrant"]["collection_name"], query_vector=vector, limit=limit)
    return [
        {
            "id": res.id,
            "score": res.score,
        }
        for res in search_result
    ]
