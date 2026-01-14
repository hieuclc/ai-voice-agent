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

import google.genai as genai
import dotenv
dotenv.load_dotenv(override = True)
from google.genai.types import GenerateContentConfig

client = genai.Client() 

MODEL_ID = "gemini-2.5-flash"

@mcp.tool()
async def gemini_search(query: str):
    """Search Gemini for the user's query
    
    Args:
        query: The user's query. It should be reformatted as a question
    """
    # response = client.models.generate_content(
    #     model=MODEL_ID,
    #     contents=f"Tìm kiếm thông tin trên google và trả lời câu hỏi sau: {query}. Chỉ tập trung vào câu hỏi, không trả lời lan man.",
    #     config={"tools": [{"google_search": {}}]},
    # )
    system_instruction = """Bạn là một trợ lý tìm kiếm và tổng hợp thông tin
    QUY TẮC BẮT BUỘC:
    - Luôn gọi tool tìm kiếm google (google_search) cho yêu cầu của người dùng.
    """
    config = GenerateContentConfig(
            system_instruction=[system_instruction],
            tools = [{"google_search": {}}]
    )
    query = "Trường đại học công nghệ đại học quốc gia hà nội có hiệu trưởng là ai"
    response = client.models.generate_content(
            model=MODEL_ID,
            contents=f"{query}",
            config=config,
        )
    text_arr = []
    for part in response.candidates[0].content.parts:
        text = part.text
        text_arr.append(text)
    result = "".join(text_arr)
    return result
    

# @mcp.tool()
async def google_search(query: str, top_k = 3):
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
    wikipedia_list = []
    wikipedia_content = []
    if "wikipedia" not in query:
        async with httpx.AsyncClient(timeout=10) as client:
            wiki_params = {
                "key": os.environ["GOOGLE_API_KEY"],
                "cx": os.environ["GOOGLE_SEARCH_ENGINE_ID"],
                "q": query + " wikipedia",
                "hl": "vi",
                "gl": "vn",
            }
            response = await client.get(base_url, params = wiki_params)
            response.raise_for_status()
            wikipedia_list = [response.json().get("items")[0].get("link")]
        print(wikipedia_list)

        search_results = [i for i in search_results if "wikipedia" not in i.get("link")]
    if wikipedia_list:
        wikipedia_url = wikipedia_list[0]
        wikipedia_post = wikipedia_url.replace("https://vi.wikipedia.org/wiki/", "")
        wikipedia_base_url = f"https://vi.wikipedia.org/w/api.php?format=json&action=query&titles={wikipedia_post}&prop=extracts&explaintext=true"
        print(wikipedia_base_url)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        async with httpx.AsyncClient(timeout=10, headers = headers) as client:
            
            response = await client.get(wikipedia_base_url)
            response.raise_for_status()

            for key, content in response.json().get("query").get("pages").items():
                wikipedia_content.append(content.get("extract"))


    urls = [i.get('link') for i in search_results[:top_k]]
    chunks = crawl(urls)
    chunks.extend(semantic_chunk("".join(wikipedia_content)))
    for chunk in chunks:
        print(chunk)
    best_chunks = reranker(query, chunks)
    best_chunks = [clean_markdown_plain(i) for i in best_chunks]

    result = ""
    for _, i in enumerate(best_chunks):
        context = f"[Source {i}:]\n{i}\n"
        result += context

    return result

if __name__ == "__main__":
    mcp.run(transport='http', port = 8000)



