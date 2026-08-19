"""Property test for conservative grounding fallback and inert output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hypothesis import given, settings, strategies as st

from domain import AppConfig, RetrievedChunk
from rag import INSUFFICIENT_ANSWER, RAGService


Scenario = Literal[
    "unsupported_number",
    "unsupported_code",
    "unsupported_name",
    "unsupported_phrase",
    "supported_inert_text",
]
_TOKEN = st.text(
    alphabet=st.sampled_from(tuple("abcdefghijklmnopqrstuvwxyz")),
    min_size=4,
    max_size=10,
)


@dataclass(frozen=True, slots=True)
class _GroundingCase:
    scenario: Scenario
    context: tuple[RetrievedChunk, ...]
    generated_answer: str
    unsupported_fragment: str | None

    @property
    def should_be_grounded(self) -> bool:
        return self.scenario == "supported_inert_text"


class _StaticRetrieval:
    def __init__(self, context: tuple[RetrievedChunk, ...]) -> None:
        self._context = context
        self.calls = 0

    def retrieve(self, question: str) -> tuple[RetrievedChunk, ...]:
        assert question
        self.calls += 1
        return self._context


class _StaticGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int, timeout_seconds: int) -> str:
        assert prompt
        assert max_tokens > 0
        assert timeout_seconds > 0
        self.calls += 1
        return self._answer


def _config() -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused-property-model",
        chroma_path=Path("/unused/chroma"),
        chroma_collection="property_grounding_fallback",
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


def _chunk(
    ordinal: int,
    text: str,
    document_token: str,
    page: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{ordinal}-{document_token}",
        text=text,
        document_id=f"document-{ordinal}-{document_token}",
        display_name=f"manuais/{document_token}-{ordinal}.pdf",
        human_page=page,
        score=0.9,
    )


@st.composite
def _grounding_cases(draw):
    scenario = draw(
        st.sampled_from(
            (
                "unsupported_number",
                "unsupported_code",
                "unsupported_name",
                "unsupported_phrase",
                "supported_inert_text",
            )
        )
    )
    tokens = draw(st.lists(_TOKEN, min_size=5, max_size=5, unique=True))
    number = draw(st.integers(min_value=10, max_value=9_999))
    page = draw(st.integers(min_value=1, max_value=50))
    literal_name = f"{tokens[2].capitalize()} {tokens[3].capitalize()}"
    literal_code = f"{tokens[2].upper()}-{number}"

    unsupported_fragment: str | None
    if scenario == "unsupported_number":
        context_text = (
            f"o cadastro {tokens[0]} usa um prazo documentado em dias."
        )
        generated_answer = (
            f"o cadastro {tokens[0]} usa o prazo de {number} dias."
        )
        unsupported_fragment = str(number)
    elif scenario == "unsupported_code":
        context_text = f"o código do cadastro {tokens[0]} está documentado."
        generated_answer = (
            f"o código do cadastro {tokens[0]} é {literal_code}."
        )
        unsupported_fragment = literal_code
    elif scenario == "unsupported_name":
        context_text = f"a tela do cadastro {tokens[0]} está documentada."
        generated_answer = (
            f"a tela {literal_name} do cadastro {tokens[0]} está documentada."
        )
        unsupported_fragment = literal_name
    elif scenario == "unsupported_phrase":
        context_text = (
            f"o cadastro {tokens[0]} permite consulta do registro {tokens[1]}."
        )
        generated_answer = (
            f"o cadastro {tokens[0]} permite apagar o registro {tokens[2]}."
        )
        unsupported_fragment = tokens[2]
    else:
        context_text = (
            f"na tela {literal_name}, o código {literal_code} mantém o valor "
            f"{number} e <script>{tokens[4]}</script> permanece como texto."
        )
        generated_answer = context_text
        unsupported_fragment = None

    distractor = (
        f"o módulo auxiliar {tokens[4]} contém orientação complementar."
        if scenario != "supported_inert_text"
        else f"o módulo auxiliar {tokens[0]} contém orientação complementar."
    )
    context = (
        _chunk(1, context_text, tokens[0], page),
        _chunk(2, distractor, tokens[1], page + 1),
    )
    return _GroundingCase(
        scenario=scenario,
        context=context,
        generated_answer=generated_answer,
        unsupported_fragment=unsupported_fragment,
    )


@settings(max_examples=150, deadline=None)
@given(case=_grounding_cases())
def test_property_16_unsupported_output_converges_to_insufficiency(
    case: _GroundingCase,
) -> None:
    # Feature: erp-ai-support, Property 16: Saída não sustentada converge para insuficiência
    # **Validates: Requirements 13.14, 13.17**
    retrieval = _StaticRetrieval(case.context)
    generator = _StaticGenerator(case.generated_answer)
    result = RAGService(_config(), retrieval, generator).answer(
        "qual é a orientação documentada?"
    )

    assert retrieval.calls == 1
    assert generator.calls == 1
    assert type(result.answer) is str

    if case.should_be_grounded:
        assert case.generated_answer in tuple(chunk.text for chunk in case.context)
        assert result.answer == case.generated_answer
        assert "<script>" in result.answer
    else:
        assert case.generated_answer.strip()
        assert case.generated_answer != INSUFFICIENT_ANSWER
        assert case.unsupported_fragment is not None
        assert case.unsupported_fragment not in " ".join(
            chunk.text for chunk in case.context
        )
        assert result.answer == INSUFFICIENT_ANSWER
        assert result.sources == ()
