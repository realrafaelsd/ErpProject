"""Unit tests for stateless, context-confined RAG composition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from domain import AppConfig, PublicError, RetrievedChunk, Source
from rag import INSUFFICIENT_ANSWER, RAGService


_CONTEXT_START = "<<<INÍCIO_CONTEXTO_NÃO_CONFIÁVEL_JSON>>>"
_CONTEXT_END = "<<<FIM_CONTEXTO_NÃO_CONFIÁVEL_JSON>>>"
_QUESTION_START = "<<<INÍCIO_PERGUNTA_NÃO_CONFIÁVEL_JSON>>>"
_QUESTION_END = "<<<FIM_PERGUNTA_NÃO_CONFIÁVEL_JSON>>>"


class _RetrievalFake:
    def __init__(
        self,
        contexts: dict[str, tuple[RetrievedChunk, ...]],
    ) -> None:
        self.contexts = contexts
        self.calls: list[str] = []

    def retrieve(self, question: str) -> tuple[RetrievedChunk, ...]:
        self.calls.append(question)
        return self.contexts[question]


class _GeneratorFake:
    def __init__(self, answers: Sequence[object]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, int, int]] = []

    def generate(self, prompt: str, max_tokens: int, timeout_seconds: int) -> object:
        self.calls.append((prompt, max_tokens, timeout_seconds))
        return self.answers[len(self.calls) - 1]


def _config(root: Path) -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="local-test-generation",
        chroma_path=root / "chroma",
        chroma_collection="test_collection",
        upload_folder=root / "uploads",
        embedding_model="local-test-embedding",
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
        ollama_timeout_seconds=2,
        max_answer_tokens=500,
        flask_host="127.0.0.1",
        flask_port=5_000,
        flask_debug=False,
    )


def _chunk(
    chunk_id: str,
    text: str,
    document: str,
    page: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=f"document-{chunk_id}",
        display_name=document,
        human_page=page,
        score=0.9,
    )


def _json_block(prompt: str, start: str, end: str) -> object:
    return json.loads(prompt.split(start, 1)[1].split(end, 1)[0].strip())


def test_empty_context_returns_exact_insufficiency_without_generation(
    tmp_path: Path,
) -> None:
    retrieval = _RetrievalFake({"pergunta sem cobertura": ()})
    generator = _GeneratorFake(["não deve ser usado"])
    service = RAGService(_config(tmp_path), retrieval, generator)

    result = service.answer("pergunta sem cobertura")

    assert result.answer == INSUFFICIENT_ANSWER
    assert result.sources == ()
    assert retrieval.calls == ["pergunta sem cobertura"]
    assert generator.calls == []


def test_rag_sends_only_retrieved_chunks_and_derives_ordered_unique_sources(
    tmp_path: Path,
) -> None:
    question = "Como salvar um pedido?"
    context = (
        _chunk(
            "chunk-1",
            "Na tela Pedidos, clique em Salvar.",
            "manual.pdf",
            2,
        ),
        _chunk(
            "chunk-2",
            "O campo Status fica Ativo.",
            "manual.pdf",
            2,
        ),
        _chunk(
            "chunk-3",
            "O prazo padrão é de 30 dias.",
            "guias/pedidos.pdf",
            4,
        ),
    )
    retrieval = _RetrievalFake({question: context})
    generator = _GeneratorFake(["Na tela Pedidos, clique em Salvar."])
    config = _config(tmp_path)
    service = RAGService(config, retrieval, generator)

    result = service.answer(question)

    assert result.answer == "Na tela Pedidos, clique em Salvar."
    assert result.sources == (
        Source(document="manual.pdf", page=2),
        Source(document="guias/pedidos.pdf", page=4),
    )
    assert len(generator.calls) == 1
    prompt, max_tokens, timeout_seconds = generator.calls[0]
    assert (max_tokens, timeout_seconds) == (
        config.max_answer_tokens,
        config.ollama_timeout_seconds,
    )
    assert _json_block(prompt, _CONTEXT_START, _CONTEXT_END) == [
        {
            "chunk_id": chunk.chunk_id,
            "document": chunk.display_name,
            "page": chunk.human_page,
            "text": chunk.text,
        }
        for chunk in context
    ]
    assert _json_block(prompt, _QUESTION_START, _QUESTION_END) == question
    assert "documento-não-recuperado.pdf" not in prompt


@pytest.mark.parametrize(
    "answer",
    (
        "Na tela Pedidos, clique em Salvar.",
        "O campo Status fica Ativo.",
    ),
)
def test_sources_depend_on_sent_context_not_on_supported_model_wording(
    tmp_path: Path,
    answer: str,
) -> None:
    question = "Como funciona o pedido?"
    context = (
        _chunk(
            "chunk-1",
            "Na tela Pedidos, clique em Salvar.",
            "manual.pdf",
            2,
        ),
        _chunk(
            "chunk-2",
            "O campo Status fica Ativo.",
            "campos.pdf",
            5,
        ),
    )
    service = RAGService(
        _config(tmp_path),
        _RetrievalFake({question: context}),
        _GeneratorFake([answer]),
    )

    result = service.answer(question)

    assert result.answer == answer
    assert result.sources == (
        Source(document="manual.pdf", page=2),
        Source(document="campos.pdf", page=5),
    )


@pytest.mark.parametrize(
    "generated",
    (
        "A tela inventada exclui pedidos automaticamente.",
        "Fonte: documento-não-recuperado.pdf, página 99.",
        INSUFFICIENT_ANSWER,
    ),
)
def test_unsupported_or_insufficient_generation_has_no_sources(
    tmp_path: Path,
    generated: str,
) -> None:
    question = "O que devo fazer?"
    context = (
        _chunk(
            "chunk-1",
            "Na tela Pedidos, clique em Salvar.",
            "manual.pdf",
            2,
        ),
    )
    service = RAGService(
        _config(tmp_path),
        _RetrievalFake({question: context}),
        _GeneratorFake([generated]),
    )

    result = service.answer(question)

    assert result.answer == INSUFFICIENT_ANSWER
    assert result.sources == ()


def test_blank_generation_is_discarded_as_generation_failure(tmp_path: Path) -> None:
    question = "Como salvar?"
    context = (
        _chunk(
            "chunk-1",
            "Na tela Pedidos, clique em Salvar.",
            "manual.pdf",
            2,
        ),
    )
    service = RAGService(
        _config(tmp_path),
        _RetrievalFake({question: context}),
        _GeneratorFake(["   "]),
    )

    with pytest.raises(PublicError) as captured:
        service.answer(question)

    assert captured.value.code == "generation_failed"
    assert captured.value.http_status == 503


def test_consecutive_questions_do_not_share_question_context_answer_or_sources(
    tmp_path: Path,
) -> None:
    first_question = "Como funciona o fluxo Alfa?"
    second_question = "Como funciona o fluxo Beta?"
    first_context = (
        _chunk(
            "chunk-alpha",
            "O fluxo Alfa salva pedidos.",
            "alpha.pdf",
            1,
        ),
    )
    second_context = (
        _chunk(
            "chunk-beta",
            "O fluxo Beta cancela pedidos.",
            "beta.pdf",
            3,
        ),
    )
    retrieval = _RetrievalFake(
        {
            first_question: first_context,
            second_question: second_context,
        }
    )
    generator = _GeneratorFake(
        [
            "O fluxo Alfa salva pedidos.",
            "O fluxo Beta cancela pedidos.",
        ]
    )
    service = RAGService(_config(tmp_path), retrieval, generator)

    first = service.answer(first_question)
    second = service.answer(second_question)

    assert first.sources == (Source(document="alpha.pdf", page=1),)
    assert second.sources == (Source(document="beta.pdf", page=3),)
    assert retrieval.calls == [first_question, second_question]
    assert len(generator.calls) == 2
    first_prompt = generator.calls[0][0]
    second_prompt = generator.calls[1][0]
    assert _json_block(first_prompt, _QUESTION_START, _QUESTION_END) == first_question
    assert _json_block(second_prompt, _QUESTION_START, _QUESTION_END) == second_question
    assert _json_block(second_prompt, _CONTEXT_START, _CONTEXT_END) == [
        {
            "chunk_id": "chunk-beta",
            "document": "beta.pdf",
            "page": 3,
            "text": "O fluxo Beta cancela pedidos.",
        }
    ]
    assert first_question not in second_prompt
    assert "O fluxo Alfa salva pedidos." not in second_prompt
    assert "alpha.pdf" not in second_prompt
