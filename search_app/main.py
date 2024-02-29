from typing import List

from fastapi import FastAPI, HTTPException

from search_app.model_utils import embed_query
from search_app.qdrant_utils import qdrant_search, qdrant_setup

qdrant_setup()

app = FastAPI()


def search(text: str) -> List[int]:
    query_vector = embed_query(text).tolist()
    ids = qdrant_search(query_vector)
    return ids


@app.get("/search/")
def search_endpoint(query: str):
    try:
        result = search(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
