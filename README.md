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

![TODO IMAGE](https://ci3.googleusercontent.com/mail-img-att/AGAZnRrfncYS6ySJECTu_V17rzXYwYtmyIVCnT-8ACSPKo7uKy_nxHHW60vlVV0iYyfKK7F2Fx7jMXASUFzWCw9TASeom_9MPClecIlmL7YEiQHbVLQhrMjaYbyP-W16uyt28P-XqzpFtAgbMR1gItdQNvoKtatdfL5Gu96JLTg5ptbdOKja29DeoPNGzZBkwmHpHHBJpfNgtjvT20NTzgHwapPX7IXrBKpJV560M16RUjjDG9-6Bz9nNdc_NTbHlr2Xet7fO_F_tzK9YZ477t7u0l1g_u6wlG-4VzfvC0ACS5z5jZI9I1Mb2rE232_Qub7k9XMiTMFStI48HpFeZ50sLkc85SQQWIurr-5AoOxVNguVPx6dLrCaWebejAYn_RkAmKVkyt0EBvUz0DhVqwQHfhOelK2OIOzGJ7X_j1-cGPGZXv17Mat2trqUK5uXn6rqn9yywV_kPXmyvqFAQnhbDT-Zc_5elCO7HmFZ8aS7V8mwjRjB9LFhrquNHNrbV4qjeP5u0sEauzigwoaI-rRh1-d5_ZVGHOEdfSoNMEJz-EqpmkhdrPEpzcq0aE7idmQ1AS6eYl9_kE1Juv-YkbBN7d6Ppr3lLwbj9ia__NnROBdtV5_58iQU5f2jSET5dtt0LsMVpQ8HGPsEoACZe6FHmyeZdF5xFJ-vcHwTvWxzNPdQXBaG5m4wEwTHoCArmy3zQ4-jTu-8LMwADYYp9YaNHK_EonUAFOBYhs4UmTs9bnkmaVlwA8cvSsk7wTm3VLiZ-wXjBpT1NI4lsTG33MtJ9CtLykCtPEf4AgZ46H5bKX1q8-4aBEWSW9omR0sFw6K6Wgu7vGYubssJ84GirwQa6H0ZHAWd9aJsdXczweA0x3wlxi-IZyjqJpyM1KQm6XGR8_p-nMMKNuJkW9F0PArVYiBnO95osKVbMJddjDweK3ZQgMbXmtTcCiGFj6JTsX0ivLgBlpATaAKUCJfhfwSG-yYoATA2-r1maoLrJe9el4uONHIJDFXGyJ-CXPuzgFfcRn81atZsOT1xQjwm-4OsRYLVwX6bQCL-6DU1LwQPGzMyDo_cBvr1QBU=s0-l75-ft)

## Dataset
`Advertisement Image Dataset` is used in this MVP. See `notebooks/EDA.ipynb` for data analysis & report.

## Next steps
#### Engineering:
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