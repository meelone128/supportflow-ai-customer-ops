"""Local document ingestion and sparse-vector retrieval for SupportFlow."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from math import log, sqrt
import os
from pathlib import Path
import re
from typing import Protocol

from dotenv import load_dotenv

from supportflow.domain import Evidence, TicketCategory


@dataclass(frozen=True)
class KnowledgeChunk:
    source_id: str
    title: str
    category: TicketCategory
    content: str
    version: str = "baseline"
    updated_at: str = ""


def _tokens(text: str) -> list[str]:
    """Tokenize English words and Chinese character n-grams without extra packages."""
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    chinese_sequences = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese_tokens: list[str] = []
    for sequence in chinese_sequences:
        chinese_tokens.extend(sequence)
        chinese_tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return words + chinese_tokens


class LocalVectorIndex:
    """A small TF-IDF cosine index; replaceable by Qdrant without changing callers."""

    def __init__(self, chunks: list[KnowledgeChunk]):
        self.chunks = chunks
        document_frequency: Counter[str] = Counter()
        self.term_counts = [Counter(_tokens(chunk.content)) for chunk in chunks]
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        total = len(chunks)
        self.idf = {term: log((total + 1) / (frequency + 1)) + 1 for term, frequency in document_frequency.items()}

    def search(self, query: str, category: TicketCategory, limit: int = 2) -> list[tuple[KnowledgeChunk, float]]:
        query_counts = Counter(_tokens(query))
        if not query_counts:
            return []
        query_vector = {term: count * self.idf.get(term, 0.0) for term, count in query_counts.items()}
        query_norm = sqrt(sum(weight * weight for weight in query_vector.values()))
        scored: list[tuple[KnowledgeChunk, float]] = []
        for chunk, counts in zip(self.chunks, self.term_counts):
            if chunk.category is not category:
                continue
            document_vector = {term: count * self.idf[term] for term, count in counts.items()}
            document_norm = sqrt(sum(weight * weight for weight in document_vector.values()))
            dot_product = sum(query_vector.get(term, 0.0) * weight for term, weight in document_vector.items())
            score = dot_product / (query_norm * document_norm) if query_norm and document_norm else 0.0
            if score > 0:
                scored.append((chunk, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class BailianEmbeddingProvider:
    """Alibaba Cloud Model Studio embedding adapter via its OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v4", timeout_seconds: float = 12):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class SemanticVectorIndex:
    """Lazily embeds knowledge chunks and uses cosine similarity at query time."""

    def __init__(self, chunks: list[KnowledgeChunk], provider: EmbeddingProvider):
        self.chunks = chunks
        self.provider = provider
        self.document_vectors: list[list[float]] | None = None

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(first * second for first, second in zip(left, right))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def search(self, query: str, category: TicketCategory, limit: int = 2) -> list[tuple[KnowledgeChunk, float]]:
        if self.document_vectors is None:
            self.document_vectors = self.provider.embed([chunk.content for chunk in self.chunks])
        query_vector = self.provider.embed([query])[0]
        scored = [
            (chunk, self._cosine(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.document_vectors)
            if chunk.category is category
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


class KnowledgeGrounder:
    """Ingest policy Markdown files and return evidence with source citations."""

    def __init__(
        self,
        knowledge_directory: Path | None = None,
        managed_directory: Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        default_directory = knowledge_directory or Path(__file__).parent / "knowledge_base"
        directories = [default_directory]
        if managed_directory:
            managed_directory.mkdir(parents=True, exist_ok=True)
            directories.append(managed_directory)
        chunks = self._load_chunks(directories)
        self.index = LocalVectorIndex(chunks)
        self.semantic_index = SemanticVectorIndex(chunks, embedding_provider) if embedding_provider else None

    @staticmethod
    def _load_chunks(directories: list[Path]) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for directory in directories:
            for document_path in sorted(directory.glob("*.md")):
                chunks.append(KnowledgeGrounder._parse_document(document_path))
        return chunks

    @staticmethod
    def _parse_document(document_path: Path) -> KnowledgeChunk:
            lines = document_path.read_text(encoding="utf-8").splitlines()
            metadata = {
                key.strip(): value.strip()
                for line in lines
                if ":" in line and not line.startswith("#")
                for key, value in [line.split(":", 1)]
            }
            content = "\n".join(line for line in lines if ":" not in line or line.startswith("#")).strip()
            return KnowledgeChunk(
                source_id=metadata["source_id"],
                title=metadata["title"],
                category=TicketCategory(metadata["category"]),
                content=content,
                version=metadata.get("version", "baseline"),
                updated_at=metadata.get("updated_at", ""),
            )

    def retrieve(self, query: str, category: TicketCategory) -> tuple[list[Evidence], str]:
        """Return evidence plus the retriever mode used for auditability."""
        if category is TicketCategory.UNKNOWN:
            return [], "not_applicable"
        try:
            results = self.semantic_index.search(query, category) if self.semantic_index else self.index.search(query, category)
            mode = "bailian_semantic" if self.semantic_index else "local_tfidf"
        except Exception:
            # Vector services are optional in local development. A retrieval outage
            # must not prevent the ticket from using the deterministic sparse index.
            results = self.index.search(query, category)
            mode = "local_tfidf_fallback"
        return [
            Evidence(chunk.source_id, chunk.title, chunk.content, round(score, 3), chunk.version)
            for chunk, score in results
        ], mode

    def find_evidence(self, query: str, category: TicketCategory) -> list[Evidence]:
        return self.retrieve(query, category)[0]


class ManagedKnowledgeRepository:
    """Persists operator-maintained knowledge as version-controlled-friendly Markdown."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.history_directory = directory / ".history"
        self.history_directory.mkdir(exist_ok=True)

    def save(self, source_id: str, title: str, category: TicketCategory, content: str) -> KnowledgeChunk:
        if not re.fullmatch(r"[A-Z0-9-]{3,64}", source_id):
            raise ValueError("source_id must use uppercase letters, digits, and hyphens")
        path = self.directory / f"{source_id}.md"
        previous_version = 0
        if path.exists():
            previous = KnowledgeGrounder._parse_document(path)
            previous_version = int(previous.version.removeprefix("v")) if previous.version.removeprefix("v").isdigit() else 1
            archive_directory = self.history_directory / source_id
            archive_directory.mkdir(parents=True, exist_ok=True)
            (archive_directory / f"v{previous_version}.md").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        version = f"v{previous_version + 1}"
        updated_at = datetime.now(timezone.utc).isoformat()
        document = "\n".join([
            f"source_id: {source_id}",
            f"title: {title}",
            f"category: {category.value}",
            f"version: {version}",
            f"updated_at: {updated_at}",
            "",
            f"# {title}",
            "",
            content.strip(),
            "",
        ])
        path.write_text(document, encoding="utf-8")
        return KnowledgeGrounder._parse_document(path)

    def list_documents(self) -> list[KnowledgeChunk]:
        return [KnowledgeGrounder._parse_document(path) for path in sorted(self.directory.glob("*.md"))]

    def list_versions(self, source_id: str) -> list[KnowledgeChunk]:
        if not re.fullmatch(r"[A-Z0-9-]{3,64}", source_id):
            raise ValueError("source_id must use uppercase letters, digits, and hyphens")
        paths = list((self.history_directory / source_id).glob("v*.md"))
        current = self.directory / f"{source_id}.md"
        if current.exists():
            paths.append(current)
        documents = [KnowledgeGrounder._parse_document(path) for path in paths]
        return sorted(
            documents,
            key=lambda document: int(document.version.removeprefix("v")) if document.version.removeprefix("v").isdigit() else 1,
            reverse=True,
        )

    def restore(self, source_id: str, version: str) -> tuple[KnowledgeChunk, KnowledgeChunk]:
        """Restore a historical document as a new version; never overwrite history."""
        if not re.fullmatch(r"v[1-9][0-9]*", version):
            raise ValueError("version must use the vN format")
        target = next((document for document in self.list_versions(source_id) if document.version == version), None)
        if target is None:
            raise FileNotFoundError(f"Version {version} does not exist for {source_id}")
        body = target.content.removeprefix(f"# {target.title}").strip()
        restored = self.save(source_id, target.title, target.category, body)
        return restored, target


def create_default_knowledge_grounder(managed_directory: Path | None = None) -> KnowledgeGrounder:
    """Use Bailian semantic embeddings when configured, otherwise use local TF-IDF."""
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".dev.env", override=False)
    load_dotenv(project_root / ".env", override=False)
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    base_url = (os.getenv("DASHSCOPE_BASE_URL") or "").strip()
    workspace_id = (os.getenv("DASHSCOPE_WORKSPACE_ID") or "").strip()
    # Operators often paste the whole OpenAI-compatible URL into the workspace
    # field. Accept that form as well as a bare workspace ID to avoid producing
    # an invalid double-prefixed endpoint.
    if not base_url and workspace_id:
        base_url = (
            workspace_id
            if workspace_id.startswith(("https://", "http://"))
            else f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        )
    retrieval_mode = os.getenv("SUPPORTFLOW_RETRIEVAL_MODE", "local").strip().lower()
    provider = BailianEmbeddingProvider(
        api_key=api_key,
        base_url=base_url,
        model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4"),
        timeout_seconds=float(os.getenv("SUPPORTFLOW_BAILIAN_TIMEOUT_SECONDS", "12")),
    ) if retrieval_mode == "semantic" and api_key and base_url else None
    return KnowledgeGrounder(managed_directory=managed_directory, embedding_provider=provider)
