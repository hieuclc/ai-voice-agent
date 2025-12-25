import httpx
from fastmcp import FastMCP
import os
import dotenv
import time
import requests
import re
from bs4 import BeautifulSoup
import markdown as md

dotenv.load_dotenv(override=True)
mcp = FastMCP("mcp-server")

#helper functions ??
def semantic_chunk(text: str, max_len: int = 800):
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]

    chunks = []
    buf = ""

    for p in paragraphs:
        if len(buf) + len(p) < max_len:
            buf += " " + p
        else:
            chunks.append(buf.strip())
            buf = p

    if buf:
        chunks.append(buf.strip())

    return chunks

def crawl(urls):
    start = time.time()
    response = requests.post(
        os.getenv("CRAWL4AI_URL"),
        json={"urls": urls, "priority": 10}
    )
    end = time.time()
    print(end - start)
    if response.status_code == 200:
        print("Crawl job submitted successfully.")

    results_arr = []
    if "results" in response.json():
        results = response.json()["results"]
        for result in results:
            markdown_content = result.get("markdown").get("raw_markdown")
            results_arr.extend(semantic_chunk(markdown_content))
    return results_arr


def reranker(query, chunks, payload_chunk_size = 30, best_results = 3):
    url = os.getenv("RERANKER_URL")

    results = []

    for start_idx in range(0, len(chunks), payload_chunk_size):
        batch = chunks[start_idx : start_idx + payload_chunk_size]

        payload = {
            "query": query,
            "texts": batch
        }

        headers = {
            "Content-Type": "application/json"
        }

        start = time.time()
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        end = time.time()

        rerank_result = response.json()

        for item in rerank_result[:best_results]:
            item["index"] += start_idx

        results.extend(rerank_result)
        print(f"⏱️ batch {start_idx // payload_chunk_size}: {end - start:.3f}s")
    best_chunks_indexes = sorted(results[:best_results], key = lambda k: k.get("score"), reverse = True)
    print(best_chunks_indexes)
    best_chunks = [chunks[i.get("index")] for i in results[:best_results]]
    return best_chunks


def clean_markdown_plain(md_text: str) -> str:
    # 2. [text](url) -> text
    md_text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r"\1",
        md_text
    )
    # 3. remove numeric refs [1], [2][3]
    md_text = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", " ", md_text)

    # 4. remove citation needed
    md_text = re.sub(r"\[citation needed\]", " ", md_text, flags=re.I)

    # 5. remove raw urls
    md_text = re.sub(r"http\S+", " ", md_text)
    # 1. remove wikipedia edit UI
    md_text = re.sub(
        r"\[\s*sửa\s*\|\s*sửa\s*mã\s*nguồn\s*\]", " ", md_text, flags=re.I)

    # 6. markdown -> text
    html = md.markdown(md_text)
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")

    # 7. normalize spaces
    return re.sub(r"\s+", " ", text).strip()

## todo: new tool calling flow for google search
# start = time.time()
# search_results = await google_search("hello")
# urls = [i.get('link') for i in search_results]
# chunks = crawl(urls)
# best_chunks = reranker("hello là gì", chunks)
# best_chunks = [clean_markdown_plain(i) for i in best_chunks]
# end = time.time()
# -->> time = 8.17s, crawl time ~ 7.3s multiple websites (>5)

@mcp.tool()
async def google_search(query: str):
    """Search Google for the user's query
    
    Args:
        query: The search query
    """

    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": os.environ["GOOGLE_API_KEY"],
        "cx": os.environ["GOOGLE_SEARCH_ENGINE_ID"],
        "q": query,
        "hl": "vi",
        "gl": "vn",
    }
    search_results = []
    async with httpx.AsyncClient(timeout=10) as client:
        response = httpx.get(base_url, params = params)
        response.raise_for_status()

        for item in response.json().get("items"):
            search_results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link"),
            })
    urls = [i.get('link') for i in search_results[:3]]
    chunks = crawl(urls)
    best_chunks = reranker(query, chunks)
    best_chunks = [clean_markdown_plain(i) for i in best_chunks]

    result = ""
    for _, i in enumerate(best_chunks):
        context = f"[Source {i}:]\n{i}\n"
        result += context

    return result

if __name__ == "__main__":
    mcp.run(transport='http', port = 8000)



