"""Core RAG engine using LlamaIndex for indexing and ChromaDB as the vector store."""

import os
import logging
from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_HERE = Path(__file__).parent
KNOWLEDGE_BASE_PATH = str(_HERE / "knowledge_base")
CHROMA_DB_PATH = str(_HERE / "chroma_db")
COLLECTION_NAME = "mlops_knowledge_base"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5


class MLOpsRAGEngine:
    """Manages document indexing and retrieval for the MLOps knowledge base."""

    def __init__(
        self,
        knowledge_base_path: str = KNOWLEDGE_BASE_PATH,
        chroma_path: str = CHROMA_DB_PATH,
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.chroma_path = chroma_path
        self._setup_embeddings()
        self._setup_vector_store()
        self.index: Optional[VectorStoreIndex] = None

    def _setup_embeddings(self):
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME,
            embed_batch_size=32,
        )
        Settings.embed_model = self.embed_model
        Settings.llm = None  # We handle generation separately

    def _setup_vector_store(self):
        print(f"[DEBUG] ChromaDB path: {self.chroma_path}", flush=True)
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[DEBUG] ChromaDB collection '{COLLECTION_NAME}' size on init: {self.chroma_collection.count()}", flush=True)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def build_index(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """Build or load the vector index from the knowledge base documents."""
        existing_count = self.chroma_collection.count()
        print(f"[DEBUG] build_index called. existing ChromaDB count={existing_count}, force_rebuild={force_rebuild}", flush=True)

        if existing_count > 0 and not force_rebuild:
            print(f"[DEBUG] Reusing existing index ({existing_count} chunks)", flush=True)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embed_model,
            )
            return self.index

        print(f"[DEBUG] Knowledge base path: {self.knowledge_base_path}", flush=True)
        print(f"[DEBUG] Knowledge base path exists: {self.knowledge_base_path.exists()}", flush=True)

        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(f"Knowledge base path not found: {self.knowledge_base_path}")

        txt_files = sorted(self.knowledge_base_path.glob("*.txt"))
        print(f"[DEBUG] .txt files found: {[f.name for f in txt_files]}", flush=True)

        # Load .txt files directly — avoids the llama-index-readers-file dependency
        documents = []
        for txt_file in txt_files:
            text = txt_file.read_text(encoding="utf-8")
            print(f"[DEBUG]   loaded {txt_file.name} ({len(text)} chars)", flush=True)
            documents.append(Document(
                text=text,
                metadata={"file_name": txt_file.name},
                id_=txt_file.stem,
            ))

        if not documents:
            raise FileNotFoundError(f"No .txt files found in {self.knowledge_base_path}")
        print(f"[DEBUG] Total documents loaded: {len(documents)}", flush=True)

        parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            paragraph_separator="\n\n",
        )
        nodes = parser.get_nodes_from_documents(documents)
        print(f"[DEBUG] Total chunks created: {len(nodes)}", flush=True)

        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )
        final_count = self.chroma_collection.count()
        print(f"[DEBUG] Index built. ChromaDB collection size after indexing: {final_count}", flush=True)
        return self.index

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[NodeWithScore]:
        """Retrieve the most relevant document chunks for a query."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        logger.info(f"Retrieved {len(nodes)} nodes for query: '{query[:60]}...'")
        return nodes

    def get_node_text(self, node: NodeWithScore) -> str:
        """Extract text content from a retrieved node."""
        return node.node.get_content()

    def get_node_source(self, node: NodeWithScore) -> str:
        """Extract source filename from a retrieved node."""
        metadata = node.node.metadata
        file_name = metadata.get("file_name", metadata.get("filename", "unknown"))
        return Path(file_name).stem.replace("_", " ").title()

    def get_node_score(self, node: NodeWithScore) -> float:
        """Get similarity score for a retrieved node."""
        return float(node.score) if node.score is not None else 0.0

    def format_context(self, nodes: list[NodeWithScore]) -> tuple[str, list[dict]]:
        """Format retrieved nodes into a context string and citations list."""
        context_parts = []
        citations = []

        for i, node in enumerate(nodes, 1):
            text = self.get_node_text(node)
            source = self.get_node_source(node)
            score = self.get_node_score(node)

            context_parts.append(f"[Source {i}: {source}]\n{text}")
            citations.append({
                "index": i,
                "source": source,
                "score": round(score, 4),
                "snippet": text[:200] + "..." if len(text) > 200 else text,
            })

        return "\n\n".join(context_parts), citations
