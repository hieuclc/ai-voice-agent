import httpx
from fastmcp import FastMCP
import httpx
import os
import dotenv

dotenv.load_dotenv(override=True)
mcp = FastMCP("mcp-server")


@mcp.tool()
async def google_search(query: str):
    """Search Google for the user's query
    
    Args:
        query: The user's query
    """

    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": os.environ["GOOGLE_API_KEY"],
        "cx": os.environ["GOOGLE_SEARCH_ENGINE_ID"],
        "q": query,
        "hl": "vi",
        "gl": "vn",
    }

    results = []
    async with httpx.AsyncClient(timeout=10) as client:
        response = httpx.get(base_url, params = params)
        response.raise_for_status()

        for item in response.json().get("items"):
            results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link"),
            })
    return results

if __name__ == "__main__":
    mcp.run(transport='http', port = 8000)



