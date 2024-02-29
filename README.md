# Text2Image Search with Qdrant

This repository contains an MVP for text2image vector search implemented using Qdrant.

A [pretrained CLIP model](https://github.com/openai/CLIP) is used for text2image similarity.

The solution is provided in a form of FastAPI service. 

## Quick start

**Step 1:** download and unzip the dataset
```
./load_data.sh
```

**Step 2:** install prerequisites
```
python -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

docker pull qdrant/qdrant
```

**Step 3:** precompute image vectors
```
python3 search_app/precompute_vectors.py
```

**Step 4:** start local Qdrant
```
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

**Step 5:** start FastAPI
```
uvicorn search_app.main:app --reload
```

Now the local text2image search engine is ready for use!

## Example usage
`/search` API endpoint has a single parameter `query` and returns IDs for top-5 relevant images together with the similarity scores.
```
curl http://localhost:8000/search/?query=car
```
Response:
```json
[
    {
        "id": 1124,
        "score": 0.28792995
    },
    {
        "id": 6597,
        "score": 0.27475026
    },
    {
        "id": 9374,
        "score": 0.27437174
    },
    {
        "id": 7990,
        "score": 0.27397966
    },
    {
        "id": 5501,
        "score": 0.27362114
    }
]
```
**For image visualization and evaluation, see `notebooks/Test_API.ipynb` notebook.**


![image](https://github.com/JandJane/qdrant_test/assets/23266443/434a07d4-4081-41f0-b32a-b3bf52f9dd39)


## Dataset
`Advertisement Image Dataset` is used in this MVP. See `notebooks/EDA.ipynb` for data analysis & report.

## Next steps
#### Engineering:ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKJtGrUiwplxylvfwDV9bL4QiZHt2DkPDtGS6V5z5Cu0 foggyjandjane@gmail.com
- Dockerize the API and vector precomputation pipleine. Deploy in the cloud.
- Make the number of results returned configurable. Tune minimal score threshold.
- Add tests.

#### Data/ML:
-  Collect some labels for evaluation.
    -  **Option 1: Label images.** Categorize images into different categories (a set of categories would need to be predefined). The category names and their variations can later be used as test queries. Using the collected labels, we can calculate Precision/Recall/other search metrics.
    -   **Option 2: Label query-image pairs**, evaluating how relevant an image is to a given search query (e.g., "relevant", "somewhat relevant", or "irrelevant"). Using the collected labels, we can calculate nDCG or some other search metric. The set of queries to label depends on a specific use case and business application.
    
    For both options, labels can be collected through crowdsourcing or with the help of internal assessors. 
	A couple of hundred labeled samples can be enough to start.
-   Try out other pre-trained models and approaches and compare their quality to CLIP (can use labeled evaluation data for this).
-   Fine-tune a pretrained model on our data or train from scratch (would require more labeled data).
