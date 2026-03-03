"""
LangGraph-based RAG agent using ViLeXa-style retrieval and an OpenAI LLM.

This module is intentionally **framework-agnostic**: it exposes a Python class
that you can wire into any HTTP server. A thin FastAPI/OpenAI-compatible layer
can adapt OpenAI Chat Completions requests to this agent.

Key characteristics:
- Hybrid dense+sparse retrieval with Qdrant (via `langchain_qdrant`)
- GTE multilingual embeddings (dense + SPLADE-style sparse) from `rag_core`
- Simple 2-node LangGraph pipeline: `retrieve` -> `generate`
- OpenAI LLM via `langchain_openai.ChatOpenAI`
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from rag_core import GTEEmbedding, GTEDenseAdapter, GTESparseAdapter


class RAGSettings(BaseModel):
    """Environment-driven configuration for the RAG agent."""

    # OpenAI LLM configuration
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )
    openai_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    openai_temperature: float = Field(
        default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    )

    # Qdrant / retrieval configuration
    qdrant_host: str = Field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = Field(
        default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333"))
    )
    collection_name: str = Field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "laws")
    )
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = Field(
        default_factory=lambda: os.getenv("RETRIEVAL_MODE", "hybrid")
    )
    retrieval_k: int = Field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_K", "10"))
    )
    top_k: int = Field(default_factory=lambda: int(os.getenv("TOP_K", "3")))

    # Embedding model (GTE, mirroring ViLeXa default)
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME", "Alibaba-NLP/gte-multilingual-base"
        )
    )


class AgentState(TypedDict):
    """State for the simple RAG workflow."""

    messages: List[BaseMessage]
    context: str
    sources: List[Dict[str, Any]]


_RETRIEVAL_MODE_MAP = {
    "hybrid": RetrievalMode.HYBRID,
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
}


class OpenAIStyleRAGAgent:
    """
    OpenAI-compatible LangGraph RAG agent.

    This class exposes a `chat_completions`-like method that accepts an
    OpenAI-style messages list and returns a response dict shaped like
    `/v1/chat/completions`.
    """

    def __init__(self, settings: Optional[RAGSettings] = None) -> None:
        self.settings = settings or RAGSettings()
        self._validate_settings()

        self._llm: ChatOpenAI = self._build_llm()
        self._vector_store: QdrantVectorStore = self._build_vector_store()
        self._retriever = self._vector_store.as_retriever(
            search_kwargs={"k": self.settings.retrieval_k}
        )
        self._workflow = self._build_workflow()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _validate_settings(self) -> None:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIStyleRAGAgent.")
        if self.settings.retrieval_mode not in _RETRIEVAL_MODE_MAP:
            raise ValueError(
                f"Invalid RETRIEVAL_MODE={self.settings.retrieval_mode}. "
                f"Expected one of {list(_RETRIEVAL_MODE_MAP.keys())}."
            )

    def _build_llm(self) -> ChatOpenAI:
        kwargs: Dict[str, Any] = {
            "model": self.settings.openai_model,
            "temperature": self.settings.openai_temperature,
            "api_key": self.settings.openai_api_key,
        }
        if self.settings.openai_base_url:
            kwargs["base_url"] = self.settings.openai_base_url
        return ChatOpenAI(**kwargs)

    def _build_vector_store(self) -> QdrantVectorStore:
        # Instantiate GTE hybrid embeddings (dense + sparse)
        gte_engine = GTEEmbedding(model_name=self.settings.embedding_model_name)

        dense_adapter = None
        sparse_adapter = None

        if self.settings.retrieval_mode in ("hybrid", "dense"):
            dense_adapter = GTEDenseAdapter(gte_engine)
        if self.settings.retrieval_mode in ("hybrid", "sparse"):
            sparse_adapter = GTESparseAdapter(gte_engine)

        client = QdrantClient(
            host=self.settings.qdrant_host, port=self.settings.qdrant_port
        )

        return QdrantVectorStore(
            client=client,
            collection_name=self.settings.collection_name,
            embedding=dense_adapter,
            sparse_embedding=sparse_adapter,
            vector_name="dense",
            sparse_vector_name="sparse",
            retrieval_mode=_RETRIEVAL_MODE_MAP[self.settings.retrieval_mode],
        )

    # ------------------------------------------------------------------
    # LangGraph workflow
    # ------------------------------------------------------------------

    def _build_workflow(self):
        graph = StateGraph(AgentState)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    # Nodes ----------------------------------------------------------------

    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve documents for the latest user query."""
        last_user = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        query = last_user.content if last_user is not None else ""

        t0 = time.time()
        docs: List[Document] = self._retriever.invoke(query)
        t_elapsed = time.time() - t0

        top_docs = docs[: self.settings.top_k]
        context_text = "\n\n".join(d.page_content for d in top_docs)
        sources = [d.metadata for d in top_docs]

        # Minimal logging via print to avoid coupling to any logging config
        print(
            f"[RAG] Retrieved {len(docs)} docs ({len(top_docs)} used) "
            f"in {t_elapsed:.3f}s for query: {query!r}"
        )

        return {"context": context_text, "sources": sources}

    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate an answer using the retrieved context and conversation history."""
        system_prompt = (
            "Bạn là một trợ lý pháp lý tiếng Việt, trả lời dựa trên bối cảnh pháp luật được cung cấp.\n"
            "Luôn trả lời bằng tiếng Việt, trích dẫn và giải thích rõ ràng nhưng ngắn gọn."
        )

        context_header = f"\n\nBỐI CẢNH PHÁP LUẬT:\n{state.get('context', '')}\n"

        messages: List[BaseMessage] = []
        messages.append(SystemMessage(content=system_prompt + context_header))

        # Preserve prior conversation (excluding any old system prompts)
        for m in state["messages"]:
            if isinstance(m, SystemMessage):
                continue
            messages.append(m)

        t0 = time.time()
        result = self._llm.invoke(messages)
        t_elapsed = time.time() - t0

        print(f"[RAG] Generation time: {t_elapsed:.3f}s")

        # Append AI message back into the conversation state
        updated_messages = state["messages"] + [AIMessage(content=result.content)]

        return {"messages": updated_messages}

    # ------------------------------------------------------------------
    # OpenAI-style adapter
    # ------------------------------------------------------------------

    def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        OpenAI-style chat completions entrypoint (non-streaming).

        Args:
            messages: List of dicts with `role` and `content` (OpenAI format).
            stream: Streaming not yet implemented (must be False).
            model: Optional model override (ignored; uses settings).
        """
        if stream:
            raise NotImplementedError("Streaming is not implemented yet.")

        # Convert OpenAI-style messages into LangChain messages
        lc_messages: List[BaseMessage] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        initial_state: AgentState = {
            "messages": lc_messages,
            "context": "",
            "sources": [],
        }

        final_state = self._workflow.invoke(initial_state)

        final_messages: List[BaseMessage] = final_state["messages"]
        last_ai = next(
            (m for m in reversed(final_messages) if isinstance(m, AIMessage)), None
        )
        answer = last_ai.content if last_ai is not None else ""

        # Build minimal OpenAI-compatible response
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        used_model = model or self.settings.openai_model

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": used_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                }
            ],
            # Usage is optional; we omit token accounting here.
            "usage": None,
            # Non-standard but practical: surface RAG sources for callers that care.
            "rag_sources": final_state.get("sources", []),
        }


__all__ = ["RAGSettings", "OpenAIStyleRAGAgent"]

