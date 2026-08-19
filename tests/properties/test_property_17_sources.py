"""Property test for ordered, context-confined source provenance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from hypothesis import given, settings, strategies as st

from domain import AppConfig, RetrievedChunk, Source
from rag import INSUFFICIENT_ANSWER, RAGService, derive_sources


_CONTEXT_START = "<<<INÍCIO_CONTEXTO_NÃO_CONFIÁVEL_JSON>>>"
_CONTEXT_END = "<<<FIM_CONTEXTO_NÃO_CONFIÁVEL_JSON>>>"
_TOKEN = st.text(
    alphabet=st.sampled_from(tuple("abcdefghijklmnopqrstuvwxyz")),
    min_size=3,
    max_size=10,
)
_SOURCE_PAIR = st.tuples(_TOKEN, st.integers(min_value=1, max_value=200))


@dataclass(frozen=True, slots=True)
class _SourceCase:
    context: tuple[RetrievedChunk, ...]
    claimed_sources: tuple[tuple[str, int], ...]
    arbitrary_model_output: str


class _StaticRetrieval:
    def __init__(self, context: tuple[RetrievedChunk, ...]) -> None:
        self._context = context
        self.calls: list[str] = []

    def retrieve(self, question: str) -> tuple[RetrievedChunk, ...]:
        self.calls.append(question)
        return self._context


class _StaticGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls: list[tuple[str, int, int]] = []

    def generate(self, prompt: str, max_tokens: int, timeout_seconds: int) -> str:
        self.calls.append((prompt, max_tokens, timeout_seconds))
        return self._answer


def _config() -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused-property-model",
        chroma_path=Path("/unused/chroma"),
        chroma_collection="property_sources",
        upload_folder=Path("/unused/uploads"),
        embedding_model="unused-property-embedding",
        top_k=6,
        chunk_size=800,
        chunk_overlap=150,
        relevance_threshold=0.3,
        max_upload_bytes=1_048_576,
        max_zip_entries=10,
        max_zip_entry_bytes=1_048_576,
        max_uncompressed_bytes=2_097_152,
        max_compression_ratio=100.0,
        max_question_chars=2_000,
        ollama_timeout_seconds=120,
        max_answer_tokens=500,
        flask_host="127.0.0.1",
        flask_port=5_000,
        flask_debug=False,
    )


@st.composite
def _source_cases(draw) -> _SourceCase:
    distinct_pairs = draw(
        st.lists(
            _SOURCE_PAIR,
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    pair_order = list(
        draw(st.permutations(tuple(range(len(distinct_pairs)))))
    )
    repeated_pair = draw(st.sampled_from(pair_order))
    duplicate_position = draw(
        st.integers(min_value=0, max_value=len(pair_order))
    )
    pair_order.insert(duplicate_position, repeated_pair)

    reference_tokens = draw(
        st.lists(
            _TOKEN,
            min_size=len(pair_order),
            max_size=len(pair_order),
            unique=True,
        )
    )
    reference_pages = draw(
        st.lists(
            st.integers(min_value=1, max_value=500),
            min_size=len(pair_order),
            max_size=len(pair_order),
        )
    )

    chunks: list[RetrievedChunk] = []
    claimed_sources: list[tuple[str, int]] = []
    for ordinal, pair_index in enumerate(pair_order):
        document_token, source_page = distinct_pairs[pair_index]
        claimed_document = f"citacoes/{reference_tokens[ordinal]}.pdf"
        claimed_page = reference_pages[ordinal]
        claimed_sources.append((claimed_document, claimed_page))
        chunks.append(
            RetrievedChunk(
                chunk_id=f"chunk-{ordinal}-{reference_tokens[ordinal]}",
                text=(
                    f"o registro {reference_tokens[ordinal]} está disponível "
                    "para consulta na base de conhecimento."
                ),
                document_id=f"document-{ordinal}-{document_token}",
                display_name=f"manuais/manual-{document_token}.pdf",
                human_page=source_page,
                score=0.9,
            )
        )

    claimed_document, claimed_page = claimed_sources[0]
    arbitrary_prefix = draw(st.text(max_size=120))
    return _SourceCase(
        context=tuple(chunks),
        claimed_sources=tuple(claimed_sources),
        arbitrary_model_output=(
            f"{arbitrary_prefix}\n"
            f"a fonte {claimed_document} aponta a página {claimed_page}."
        ),
    )


def _expected_sources(
    context: tuple[RetrievedChunk, ...],
) -> tuple[Source, ...]:
    seen: set[tuple[str, int]] = set()
    expected: list[Source] = []
    for chunk in context:
        pair = (chunk.display_name, chunk.human_page)
        if pair in seen:
            continue
        seen.add(pair)
        expected.append(Source(document=pair[0], page=pair[1]))
    return tuple(expected)


def _sent_context(prompt: str) -> list[dict[str, object]]:
    encoded = prompt.split(_CONTEXT_START, 1)[1].split(_CONTEXT_END, 1)[0]
    payload = json.loads(encoded.strip())
    assert isinstance(payload, list)
    assert all(isinstance(item, dict) for item in payload)
    return payload


def _answer(
    case: _SourceCase,
    generated_answer: str,
) -> tuple[object, _StaticRetrieval, _StaticGenerator]:
    retrieval = _StaticRetrieval(case.context)
    generator = _StaticGenerator(generated_answer)
    result = RAGService(_config(), retrieval, generator).answer(
        "qual é a referência documentada?"
    )
    return result, retrieval, generator


@settings(max_examples=150, deadline=None)
@given(case=_source_cases())
def test_property_17_sources_are_ordered_unique_and_context_confined(
    case: _SourceCase,
) -> None:
    # Feature: erp-ai-support, Property 17: Fontes são deduplicadas, ordenadas e confinadas ao contexto
    # **Validates: Requirements 14.1–14.9, 15.11**
    expected = _expected_sources(case.context)
    context_pairs = tuple(
        (chunk.display_name, chunk.human_page) for chunk in case.context
    )
    source_pairs = tuple((source.document, source.page) for source in expected)

    assert len(case.context) >= 2
    assert len(set(context_pairs)) < len(context_pairs)
    assert derive_sources(case.context) == expected
    assert source_pairs == tuple(dict.fromkeys(context_pairs))
    assert 1 <= len(expected) <= _config().top_k
    assert all(pair in context_pairs for pair in source_pairs)
    assert set(source_pairs).isdisjoint(case.claimed_sources)

    for source, pair in zip(expected, source_pairs, strict=True):
        assert type(source) is Source
        assert asdict(source) == {"document": pair[0], "page": pair[1]}
        assert tuple(asdict(source)) == ("document", "page")

    grounded_results = []
    for generated_answer in (case.context[0].text, case.context[-1].text):
        result, retrieval, generator = _answer(case, generated_answer)
        grounded_results.append(result)

        assert result.answer == generated_answer
        assert result.sources == expected
        assert retrieval.calls == ["qual é a referência documentada?"]
        assert len(generator.calls) == 1

        sent = _sent_context(generator.calls[0][0])
        assert sent == [
            {
                "chunk_id": chunk.chunk_id,
                "document": chunk.display_name,
                "page": chunk.human_page,
                "text": chunk.text,
            }
            for chunk in case.context
        ]
        sent_pairs = tuple(
            (item["document"], item["page"]) for item in sent
        )
        assert tuple(dict.fromkeys(sent_pairs)) == source_pairs

    assert grounded_results[0].answer != grounded_results[1].answer
    assert grounded_results[0].sources == grounded_results[1].sources == expected

    unsupported_output = (
        f"{case.arbitrary_model_output}\n"
        "a fonte ZXQ-999999.pdf aponta a página 999999."
    )
    for generated_answer in (unsupported_output, INSUFFICIENT_ANSWER):
        result, retrieval, generator = _answer(case, generated_answer)

        assert retrieval.calls == ["qual é a referência documentada?"]
        assert len(generator.calls) == 1
        assert result.answer == INSUFFICIENT_ANSWER
        assert result.sources == ()
        assert ("ZXQ-999999.pdf", 999999) not in source_pairs
        assert derive_sources(case.context) == expected
