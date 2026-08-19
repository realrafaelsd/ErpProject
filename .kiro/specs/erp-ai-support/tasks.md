# Implementation Plan: ERP AI Support

## Overview

Este plano implementa o MVP como uma aplicação monolítica Flask em Python 3.11+, com frontend em HTML, CSS e JavaScript puro. As tarefas constroem primeiro os contratos e a configuração, depois os pipelines de ingestão e persistência, o fluxo RAG, as rotas HTTP e a interface, terminando com documentação e validações automatizadas. Cada etapa produz código integrado às etapas anteriores; não há componentes órfãos nem serviços externos adicionais.

## Tasks

- [x] 1. Estabelecer a fundação do projeto, os contratos e a configuração central
  - [x] 1.1 Criar os manifestos e a estrutura local executável
    - Criar `requirements.txt` com versões exatas e compatíveis com Python 3.11 para Flask, ChromaDB, PyMuPDF, sentence-transformers, python-dotenv e toda dependência de produção importada diretamente.
    - Criar `requirements-dev.txt` com versões exatas de pytest, Hypothesis e ferramentas locais de teste de UI; configurar `pytest.ini` para excluir por padrão testes `real_models` e `browser`, sem excluir integrações com fakes locais.
    - Criar `.gitignore` e os diretórios versionáveis `documents/uploads` e `data/chroma`, mantendo artefatos de runtime, modelos, uploads e bases locais fora do controle de versão.
    - _Arquivos/componentes: `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.gitignore`, `documents/uploads/.gitkeep`, `data/chroma/.gitkeep`._
    - _Requirements: 1.5, 1.7, 1.8, 2.1, 2.3, 2.4, 2.5, 2.6, 2.13, 2.14, 2.16, 18.3, 20.14, 20.15_

  - [x] 1.2 Implementar os contratos de domínio e os erros públicos
    - Criar dataclasses imutáveis, enums/literais e protocolos para configuração, espaço vetorial, perfil de chunking, upload, ZIP, PDF, chunks, manifestos, plano de commit, recuperação, fontes e resultados.
    - Implementar `PublicError` sem dependências de Flask ou bibliotecas de I/O e manter `domain.py` livre de comportamento de negócio e dependências circulares.
    - _Arquivos/componentes: `domain.py` (`AppConfig`, DTOs, protocolos, `PublicError`)._
    - _Requirements: 2.7, 2.8, 2.15, 15.9, 15.10, 16.1, 16.13, 21.8_

  - [x] 1.3 Implementar o gerenciador de configuração centralizado e validado
    - Implementar em `config.py` o carregamento `defaults < .env < ambiente`, trim, conversões estritas, validações individuais e relações cruzadas de todas as variáveis do perfil aprovado.
    - Validar hosts loopback, URL Ollama sem credenciais, booleanos estritos, limites em MB via `Decimal`, criação/sondagem segura de diretórios e erros públicos sem caminho absoluto.
    - Expor uma única `AppConfig` imutável e funções puras para validar compatibilidade de espaço vetorial/perfil de chunking sem disponibilizar configuração parcial.
    - _Arquivos/componentes: `config.py` (`load_config`, parsers, validação de URL/diretórios/compatibilidade)._
    - _Requirements: 3.1, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 10.1, 12.10, 18.1, 18.2_

  - [x] 1.4 Criar o exemplo completo de ambiente
    - Criar `.env.example` sem segredos, com exatamente uma ocorrência de cada variável do perfil, os valores aprovados e comentários concisos sobre finalidade e limites.
    - Garantir alinhamento literal entre nomes/defaults do arquivo e `config.py`.
    - _Arquivos/componentes: `.env.example`._
    - _Requirements: 3.2, 3.3, 18.3, 20.6_

  - [x]* 1.5 Escrever testes unitários de domínio e configuração
    - Cobrir defaults, precedência, trim, tipos, fronteiras numéricas, relações cruzadas, booleanos, URLs locais, `.env` inválido, caminhos relativos/criação/permissões e ausência de retorno parcial em falhas.
    - Verificar que mensagens citam somente a variável afetada e não vazam caminhos; testar o contrato imutável dos DTOs e erros.
    - _Arquivos/componentes: `tests/unit/test_config.py`, `tests/unit/test_domain.py`._
    - _Requirements: 3.1–3.13, 18.1, 18.2, 18.8_

- [x] 2. Implementar validação e extração segura de arquivos ZIP
  - [x] 2.1 Implementar inspeção integral, plano seguro e extração limitada do ZIP
    - Implementar `ZipValidator.inspect()` em `ingest.py` para materializar todas as entradas antes da primeira escrita, rejeitar caminhos absolutos/traversal/prefixos de unidade/controles/duplicações e tipos especiais e calcular confinamento com caminhos resolvidos reais.
    - Validar quantidade, tamanho declarado por entrada, total declarado e razão de compressão; implementar extração por stream sem `extract()`/`extractall()`, com limites reais por entrada/total, criação exclusiva, modos `0700`/`0600` e publicação da lista somente após sucesso integral.
    - Mapear estrutura ZIP inválida, entrada insegura, categoria de limite e falha de extração para códigos públicos estáveis sem reproduzir nomes maliciosos.
    - _Arquivos/componentes: `ingest.py` (`ZipValidator`, `ArchivePlan`, contadores e helpers de confinamento)._
    - _Requirements: 5.5, 5.8, 5.9, 5.11, 6.1–6.14, 18.5, 19.3, 19.6_

  - [x]* 2.2 Escrever testes unitários e de integração do ZIP e filesystem
    - Cobrir ZIP válido, enumeração inválida, traversal POSIX/Windows, UNC, symlink/FIFO/device, destinos normalizados duplicados, CRC/truncamento, limites `N-1/N/N+1`, permissões e cleanup integral.
    - Instrumentar streams/sinks para provar que nenhum byte excedente é escrito e que nenhuma lista parcial chega ao extrator.
    - _Arquivos/componentes: `tests/unit/test_zip_validator.py`, `tests/integration/test_zip_filesystem.py`._
    - _Requirements: 5.5, 5.8, 5.11, 6.1–6.14, 18.5, 19.3, 19.6_

  - [x]* 2.3 Escrever property test do plano ZIP seguro
    - **Property 1: Confinamento e publicação segura do plano ZIP**
    - Gerar nomes e tipos de entrada seguros/maliciosos e verificar confinamento real, rejeição integral antes da escrita e publicação somente após extração completa.
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.11, 6.14**
    - _Arquivos/componentes: `tests/properties/test_property_01_zip_plan.py`._

  - [x]* 2.4 Escrever property test dos limites ZIP declarados
    - **Property 2: Limites ZIP declarados são invariantes de aceitação**
    - Gerar metadados de entradas e verificar exatamente os limites de quantidade, entrada, total e razão, inclusive tamanho positivo com tamanho compactado zero.
    - **Validates: Requirements 6.4, 6.5, 6.6, 6.7**
    - _Arquivos/componentes: `tests/properties/test_property_02_zip_declared_limits.py`._

  - [x]* 2.5 Escrever property test dos limites reais de extração
    - **Property 3: Extração nunca grava além dos limites reais**
    - Gerar sequências de blocos e verificar igualdade entre bytes gravados/contadores, interrupção no primeiro byte excedente e rejeição sem escrita além do limite.
    - **Validates: Requirements 6.8, 6.9**
    - _Arquivos/componentes: `tests/properties/test_property_03_zip_actual_limits.py`._

- [x] 3. Implementar descoberta de PDFs, extração textual, identidade e chunking
  - [x] 3.1 Implementar descoberta ordenada, nomes exibíveis e avisos
    - Percorrer `ExtractedEntry` na ordem declarada pelo ZIP, selecionar recursivamente somente regulares com sufixo `.pdf` case-insensitive e emitir um aviso por não PDF.
    - Normalizar nomes relativos para POSIX/NFC, remover controles e impedir exposição de staging, caminhos absolutos ou segmentos perigosos; implementar `WarningCollector` deduplicado por chave estrutural.
    - Rejeitar upload sem candidatos PDF antes de qualquer persistência.
    - _Arquivos/componentes: `ingest.py` (descoberta, `normalize_display_name`, `WarningCollector`)._
    - _Requirements: 7.1, 7.2, 7.3, 7.13, 16.6, 16.7, 18.9_

  - [x] 3.2 Implementar extração página a página com PyMuPDF e spool fiel
    - Abrir cada PDF, enumerar todas as páginas e extrair `get_text("text")`, associando página humana; preservar pontos de código em spool UTF-8 temporário e descartar integralmente um documento se qualquer etapa falhar.
    - Tratar páginas vazias/whitespace sem OCR, contar todas as páginas de documentos legíveis, emitir avisos acionáveis e continuar com PDFs válidos coexistentes; rejeitar quando nenhum candidato for legível.
    - _Arquivos/componentes: `ingest.py` (`PdfExtractor`, spool por página e cleanup documental)._
    - _Requirements: 2.3, 7.4–7.12, 7.14, 16.3, 16.4, 16.8, 19.4, 19.5, 21.6_

  - [x] 3.3 Implementar identidades determinísticas e chunking rastreável
    - Calcular SHA-256 dos bytes originais antes da extração textual e implementar `make_chunk_id` conforme `char-v1`, usando identidade, página humana e offset.
    - Implementar divisão por caracteres com stride `CHUNK_SIZE - CHUNK_OVERLAP`, sem trim/normalização, sem cruzar páginas e com metadados completos; páginas sem texto produzem zero chunks.
    - Garantir ordem, cobertura, overlap exato, parada no primeiro chunk que alcança o fim e determinismo para a mesma entrada/configuração.
    - _Arquivos/componentes: `ingest.py` (`sha256_file`, `make_chunk_id`, `ChunkingService`)._
    - _Requirements: 8.1, 8.2, 8.9, 9.1–9.11_

  - [x]* 3.4 Escrever testes unitários e de integração de PDF, spool, avisos e chunking
    - Gerar PDFs mínimos com PyMuPDF para páginas textuais, vazias e mistas; cobrir PDF corrompido junto de válido, todos ilegíveis, nomes sanitizados, ordem/deduplicação dos avisos e cleanup.
    - Cobrir chunking nas fronteiras, overlap zero, texto Unicode, página curta/longa e metadados/IDs esperados sem usar modelos.
    - _Arquivos/componentes: `tests/unit/test_pdf_chunking.py`, `tests/integration/test_pymupdf_pipeline.py`._
    - _Requirements: 7.4–7.14, 8.1, 8.9, 9.1–9.11, 16.3–16.8_

  - [x]* 3.5 Escrever property test de descoberta e nomes seguros
    - **Property 4: Descoberta e nomes de documentos são seguros e determinísticos**
    - Gerar sequências ordenadas de nomes Unicode e verificar seleção `.pdf`, avisos, ordem e nomes relativos/NFC sem controles ou segmentos perigosos.
    - **Validates: Requirements 7.1, 7.2, 7.13, 18.9**
    - _Arquivos/componentes: `tests/properties/test_property_04_pdf_discovery.py`._

  - [x]* 3.6 Escrever property test da identidade do documento
    - **Property 5: Identidade de documento depende somente dos bytes**
    - Gerar bytes e nomes/caminhos/uploads arbitrários e verificar SHA-256 estável exclusivamente pelo conteúdo.
    - **Validates: Requirements 8.1, 8.2**
    - _Arquivos/componentes: `tests/properties/test_property_05_document_identity.py`._

  - [x]* 3.7 Escrever property test da fidelidade do spool
    - **Property 6: Spool textual preserva exatamente o texto extraído**
    - Gerar texto Unicode, persistir/ler pelo spool e verificar igualdade exata de pontos de código sem transformação.
    - **Validates: Requirements 7.7**
    - _Arquivos/componentes: `tests/properties/test_property_06_text_spool.py`._

  - [x]* 3.8 Escrever property test da identidade de chunk
    - **Property 8: Identidade de chunk é determinística**
    - Gerar identidades, páginas, offsets e versão válidos e verificar repetibilidade do identificador.
    - **Validates: Requirements 8.9**
    - _Arquivos/componentes: `tests/properties/test_property_08_chunk_identity.py`._

  - [x]* 3.9 Escrever property test dos invariantes de chunking
    - **Property 9: Chunking preserva conteúdo, overlap, cobertura e origem**
    - Gerar textos/configurações válidas e verificar tamanho, slices, offsets, overlap, cobertura, metadados, determinismo e zero chunks para página sem texto.
    - **Validates: Requirements 7.9, 7.14, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10**
    - _Arquivos/componentes: `tests/properties/test_property_09_chunking_invariants.py`._

- [x] 4. Implementar embeddings locais, manifesto e armazenamento vetorial
  - [x] 4.1 Implementar o serviço local de embeddings
    - Carregar sentence-transformers de forma preguiçosa, uma vez e sob lock, com `local_files_only=True`, `trust_remote_code=False` e normalização para documentos e perguntas.
    - Obter/confirmar dimensão, validar quantidade, tipos, finitude e dimensão de cada vetor e mapear modelo ausente/falha sem persistir resultados parciais ou tentar download.
    - _Arquivos/componentes: `rag.py` (`LocalEmbeddingService`)._
    - _Requirements: 1.5, 2.4, 3.10, 10.1–10.7, 10.10, 18.4, 19.9, 19.12_

  - [x] 4.2 Implementar manifesto SQLite, locks e journal de recuperação
    - Implementar `ManifestStore` com schemas `vector_space`, `documents` e `ingestion_transactions`, parâmetros SQL, foreign keys, WAL e `synchronous=FULL`, sem armazenar perguntas, respostas ou texto dos chunks.
    - Implementar lock de ciclo de vida para uma instância, mutex não bloqueante de ingestão e lock de visibilidade compartilhado/exclusivo; recuperar journals incompletos antes de disponibilizar rotas.
    - Registrar somente documentos de transações `COMMITTED` como duplicatas e oferecer operações PREPARED/CHROMA_COMMITTED/COMMITTED/ABORTED.
    - _Arquivos/componentes: `rag.py` (`ManifestStore`, locks, journal e recovery)._
    - _Requirements: 8.3, 8.8, 8.10, 11.4, 11.6–11.10, 11.12, 15.17, 18.11, 21.2, 21.3_

  - [x] 4.3 Implementar o adaptador Chroma persistente com atomicidade observável
    - Implementar `ChromaVectorStore` com `PersistentClient`, coleção cosseno, embeddings fornecidos pela aplicação, metadados imutáveis, validação de compatibilidade antes de count/query/write, consulta columnar validada e conversão segura de erros.
    - Usar a API pública suportada pela versão exata pinada. Se transação condicional por coleção não existir ou for incompatível, não usar vários upserts desprotegidos: gravar journal PREPARED com os novos IDs, adquirir lock exclusivo de visibilidade, rejeitar colisões divergentes, pular IDs idênticos, aplicar somente novos registros, compensar esses IDs antes de liberar o lock em falha e removê-los no recovery de inicialização após queda.
    - Manter consultas sob lock compartilhado, bloquear chat/upload em `recovery_required` e preservar todos os registros anteriores; não adicionar servidor Chroma, coleção de negócio extra, worker ou microsserviço.
    - _Arquivos/componentes: `rag.py` (`ChromaVectorStore`, protocolo de commit/rollback/recovery e compatibilidade)._
    - _Requirements: 3.7, 10.8, 10.9, 11.1–11.13, 16.12, 16.14, 19.7, 21.2_

  - [x]* 4.4 Escrever testes unitários de embeddings, manifesto e compatibilidade
    - Usar fakes para carregamento/inferência e cobrir modelo ausente, NaN/infinito, dimensão/quantidade erradas, fingerprint, metadados incompatíveis, estados do journal e erros públicos.
    - Confirmar que nenhum teste padrão baixa modelo, usa internet ou grava pergunta/resposta.
    - _Arquivos/componentes: `tests/unit/test_embeddings.py`, `tests/unit/test_manifest_store.py`, `tests/unit/test_vector_compatibility.py`._
    - _Requirements: 3.7, 10.1–10.10, 11.3, 11.6, 11.7, 18.4, 19.7, 19.9, 19.12_

  - [x]* 4.5 Escrever property test de consistência vetorial
    - **Property 11: Todo vetor aceito pertence ao espaço configurado**
    - Gerar vetores finitos/não finitos e fingerprints e verificar que somente dimensão, componentes e espaço compatíveis chegam a consulta ou persistência.
    - **Validates: Requirements 10.3, 10.7, 10.10**
    - _Arquivos/componentes: `tests/properties/test_property_11_vector_space.py`._

  - [x]* 4.6 Escrever integração do Chroma pinado, round trip e capacidade transacional
    - Em diretórios temporários, criar/reabrir coleção, confirmar configuração cosseno, upsert/get/query, metadados, distância e persistência.
    - Exercitar a estratégia realmente implementada pela versão pinada: API condicional quando disponível ou journal + lock + compensação/recovery quando não disponível; falhar se uma escrita parcial puder ser observada.
    - _Arquivos/componentes: `tests/integration/test_chroma_store.py`._
    - _Requirements: 10.8, 10.9, 11.1–11.7, 11.11–11.13_

- [x] 5. Integrar a transação completa de ingestão
  - [x] 5.1 Implementar `IngestionService` do ZIP ao commit confirmado
    - Orquestrar inspeção/extração, descoberta, SHA-256, deduplicação no manifesto e no mesmo ZIP, PDF, chunking, embeddings em lote, plano de commit, contagens e avisos, sem persistir antes de todo processamento caro terminar.
    - Revalidar duplicatas e colisões sob mutex, confirmar manifesto/Chroma pelo protocolo atômico, responder 409 à segunda ingestão e garantir cleanup de staging/spools em `finally` em todos os caminhos.
    - Preservar documentos de mesmo nome com bytes diferentes, registrar exatamente um aviso por ocorrência duplicada, reconhecer documentos legíveis sem chunks e manter estado anterior em qualquer falha.
    - _Arquivos/componentes: `ingest.py` (`IngestionService`, preparação/commit e integração com protocolos de `rag.py`)._
    - _Requirements: 2.10, 5.7–5.11, 7.11, 7.12, 8.1–8.10, 9.11, 10.2, 10.6, 10.7, 11.5–11.13, 16.1–16.14_

  - [x]* 5.2 Escrever testes unitários da orquestração de ingestão
    - Usar stores/embeddings em memória para cobrir primeira ocorrência, duplicatas entre/nos uploads, mesmo nome com bytes diferentes, documento sem chunks, falhas por fase, colisão, contagens, avisos e cleanup.
    - Verificar zero embeddings/upserts para duplicatas e zero alteração confirmada em qualquer erro.
    - _Arquivos/componentes: `tests/unit/test_ingestion_service.py`._
    - _Requirements: 5.7–5.11, 7.11, 7.12, 8.3–8.10, 16.1–16.14_

  - [x]* 5.3 Escrever property test de idempotência da ingestão
    - **Property 7: Ingestão repetida é idempotente e preserva estado anterior**
    - Modelar estado confirmado e ocorrências para verificar reaplicação, primeira ocorrência válida, avisos, coexistência por hash e preservação em tentativas falhas.
    - **Validates: Requirements 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.10, 11.11**
    - _Arquivos/componentes: `tests/properties/test_property_07_ingestion_idempotency.py`._

  - [x]* 5.4 Escrever property test de contagens e avisos
    - **Property 10: Contagens e avisos são derivados do plano confirmado**
    - Gerar resultados documentais e verificar documentos, páginas inclusive vazias, novos chunks e uma ocorrência por evento, incluindo documento totalmente sem texto.
    - **Validates: Requirements 9.11, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8**
    - _Arquivos/componentes: `tests/properties/test_property_10_upload_result.py`._

  - [x]* 5.5 Escrever property test de atomicidade do estado
    - **Property 12: Confirmação ou falha preserva atomicidade do estado**
    - Gerar estado inicial, plano e pontos de falha para verificar rollback exato, commit completo, preservação preexistente, invisibilidade parcial e rejeição de colisão divergente.
    - **Validates: Requirements 11.5, 11.6, 11.7, 11.12, 11.13, 16.14**
    - _Arquivos/componentes: `tests/properties/test_property_12_ingestion_atomicity.py`._

  - [x]* 5.6 Escrever integrações de falha, recovery e concorrência da ingestão
    - Injetar falha antes da escrita, durante Chroma, entre Chroma/manifesto, no manifesto e na compensação; reiniciar e executar recovery antes de count/query.
    - Usar threads/barreiras para provar 409 na segunda ingestão e que chat observa somente estado anterior ou final, nunca parcial.
    - _Arquivos/componentes: `tests/integration/test_ingestion_transactions.py`._
    - _Requirements: 8.8, 8.10, 11.5–11.13, 16.11–16.14_

- [~] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implementar recuperação, prompt fundamentado, Ollama e composição RAG
  - [x] 7.1 Implementar recuperação semântica determinística
    - Implementar `RetrievalService` para embedar a pergunta no espaço validado, contar chunks, consultar exatamente `min(TOP_K, count)`, converter distância por `clamp(1-distance)` e ordenar estavelmente.
    - Filtrar exatamente `score >= RELEVANCE_THRESHOLD`, validar payload/metadados/distâncias e retornar somente os chunks recuperados; coleção vazia retorna contexto vazio sem erro.
    - _Arquivos/componentes: `rag.py` (`RetrievalService`, `cosine_distance_to_relevance`)._
    - _Requirements: 10.3, 12.1–12.6, 12.10, 12.11_

  - [x] 7.2 Implementar prompt seguro, validação conservadora e fontes
    - Implementar prompt em português com papel/regras imutáveis, insuficiência exata, estilo/procedimentos/nomes literais e pergunta/contexto serializados separadamente como dados não confiáveis na ordem recuperada.
    - Implementar validação determinística conservadora da saída como texto inerte, substituindo conteúdo não sustentado por insuficiência sem segunda chamada.
    - Derivar fontes somente dos chunks efetivamente enviados, com pares documento/página deduplicados na primeira ordem e campos exatos, ignorando citações do modelo.
    - _Arquivos/componentes: `rag.py` (`build_prompt`, `validate_generated_answer`, `derive_sources`)._
    - _Requirements: 1.2, 1.4, 12.9, 13.4–13.14, 13.17, 14.1–14.9, 18.6_

  - [x] 7.3 Implementar cliente Ollama local e `RAGService`
    - Implementar cliente HTTP da biblioteca padrão sem proxy/redirect, com resolução loopback, preflight `/api/tags`, modelo configurado, deadlines de conexão/geração, `stream=false` e limite `num_predict`.
    - Mapear indisponibilidade, modelo ausente, timeout/interrupção/JSON/vazio sem fallback externo ou resposta parcial; não manter contexto conversacional.
    - Integrar recuperação, curto-circuito de contexto vazio, prompt, geração, validação e fontes em `RAGService.answer()`.
    - _Arquivos/componentes: `rag.py` (`OllamaClient`, `RAGService`)._
    - _Requirements: 1.5, 1.6, 2.6, 2.7, 12.7–12.9, 13.1–13.3, 13.13–13.17, 15.13, 15.14, 15.17, 18.4, 19.1, 19.2, 19.10, 21.3_

  - [x]* 7.4 Escrever testes unitários de recuperação, prompt, geração e RAG
    - Cobrir cardinalidade `TOP_K`, distância/score, empate estável, threshold, contexto vazio, delimitadores/injeção, procedimentos, nomes literais, validador sustentado/não sustentado e fontes.
    - Usar fakes para embedding/store/generator e verificar zero chamada Ollama na insuficiência, nenhum histórico e nenhum conteúdo externo ao contexto.
    - _Arquivos/componentes: `tests/unit/test_retrieval.py`, `tests/unit/test_prompt_grounding.py`, `tests/unit/test_rag_service.py`._
    - _Requirements: 12.1–12.11, 13.4–13.17, 14.1–14.9, 15.17_

  - [x]* 7.5 Escrever property test de cardinalidade e limiar da recuperação
    - **Property 13: Recuperação respeita cardinalidade, score, ordem e limiar**
    - Gerar candidatos/distâncias finitas e verificar limite solicitado, clamp, estabilidade de empates, filtro e confinamento exato do contexto.
    - **Validates: Requirements 12.2, 12.3, 12.4, 12.5, 12.6, 12.9**
    - _Arquivos/componentes: `tests/properties/test_property_13_retrieval.py`._

  - [x]* 7.6 Escrever property test do curto-circuito de contexto vazio
    - **Property 14: Contexto vazio interrompe geração**
    - Gerar coleções/candidatos vazios ou abaixo do limiar e verificar insuficiência exata, fontes vazias e zero chamadas ao gerador.
    - **Validates: Requirements 12.7, 12.8, 12.11**
    - _Arquivos/componentes: `tests/properties/test_property_14_empty_context.py`._

  - [x]* 7.7 Escrever property test da separação do prompt
    - **Property 15: Prompt mantém regras fora dos dados não confiáveis**
    - Gerar perguntas/chunks com falsos delimitadores e instruções e verificar regras confiáveis, serialização separada, ordem e ausência de ferramenta/comando executável.
    - **Validates: Requirements 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.10, 13.11, 13.12**
    - _Arquivos/componentes: `tests/properties/test_property_15_prompt_boundaries.py`._

  - [x]* 7.8 Escrever property test da convergência para insuficiência
    - **Property 16: Saída não sustentada converge para insuficiência**
    - Gerar respostas/contextos e verificar rejeição de números, códigos, nomes ou frases sem evidência, fontes vazias e preservação de conteúdo aceito como texto.
    - **Validates: Requirements 13.14, 13.17**
    - _Arquivos/componentes: `tests/properties/test_property_16_grounding_fallback.py`._

  - [x]* 7.9 Escrever property test da proveniência de fontes
    - **Property 17: Fontes são deduplicadas, ordenadas e confinadas ao contexto**
    - Gerar sequências com pares repetidos e saídas arbitrárias do modelo; verificar deduplicação ordenada, campos exatos, independência da saída e lista vazia na insuficiência.
    - **Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.11**
    - _Arquivos/componentes: `tests/properties/test_property_17_sources.py`._

  - [x]* 7.10 Escrever integração do cliente Ollama com servidor fake local
    - Simular `/api/tags` e `/api/generate` para modelo presente/ausente, conexão recusada, timeout, interrupção, redirect, JSON inválido e whitespace; inspecionar `num_predict`, modelo, ausência de contexto anterior e descarte parcial.
    - Verificar que proxy/host externo não é usado e que a suíte não exige Ollama ou modelo real.
    - _Arquivos/componentes: `tests/integration/test_ollama_client.py`._
    - _Requirements: 1.6, 13.1–13.3, 13.13, 13.15, 13.16, 15.13, 15.14, 18.4, 19.1, 19.2, 19.10_

- [ ] 8. Implementar a aplicação Flask e os contratos HTTP
  - [x] 8.1 Implementar composição, ciclo de vida e tratamento uniforme de erros
    - Criar `create_app()`, `register_routes()` e `main()` em `app.py`, carregar/injetar uma instância de cada serviço, preparar diretórios/locks/recovery antes das rotas e executar somente por `python app.py` no host/porta/debug configurados.
    - Implementar handlers de `PublicError`, 413 e exceção genérica, com log local de stack trace e envelope exato `{success, code, message}` sem serializar exceção interna.
    - Configurar limite Flask apenas como defesa adicional e não ler ambiente fora de `config.py`.
    - _Arquivos/componentes: `app.py` (composition root, lifecycle e error handlers)._
    - _Requirements: 1.1, 1.3, 1.7, 1.8, 2.8, 2.9, 3.10, 11.10, 15.12–15.16, 16.12–16.14, 18.1, 18.2, 18.7, 18.8, 21.8, 21.9_

  - [-] 8.2 Implementar `GET /` e o contrato completo de `POST /chat`
    - Renderizar `templates/index.html` com status 200 e validar chat na ordem: media type, corpo JSON objeto, `question` string, comprimento Unicode antes do trim e não vazio após `strip()`.
    - Encerrar rejeições antes de serviços, encaminhar somente texto aparado e retornar exatamente `{answer, sources}` em sucesso/insuficiência, sem estado de sessão ou persistência de conversa.
    - _Arquivos/componentes: `app.py` (rotas `/` e `/chat`, `validate_chat_request`)._
    - _Requirements: 4.1, 15.1–15.17, 21.3, 21.4_

  - [~] 8.3 Implementar o contrato completo de `POST /upload` e cópia limitada
    - Validar multipart, ocorrência única do campo `file`, ausência de outros arquivos, nome, extensão e MIME na ordem normativa.
    - Criar staging exclusivo, copiar stream em blocos contando bytes sem gravar o byte excedente, chamar `IngestionService` somente após contrato válido e remover toda área em `finally`.
    - Retornar contrato exato de sucesso/contagens/avisos e mapear 400/409/413/415/422/503 com `success=false`, preservando zero alteração em rejeições.
    - _Arquivos/componentes: `app.py` (rota `/upload`, `validate_upload_contract`, `copy_upload_bounded`)._
    - _Requirements: 5.1–5.11, 16.1–16.14, 19.3, 19.6_

  - [ ]* 8.4 Escrever testes unitários e integração Flask dos contratos HTTP
    - Usar Flask test client e serviços fake para cobrir tabelas de casos de chat/upload, ordem de decisão, `N-1/N/N+1`, schemas exatos, status, cleanup e zero chamadas downstream em rejeições.
    - Cobrir handler global, stack trace apenas em log local, independência entre chats e mapeamentos de dependências 503.
    - _Arquivos/componentes: `tests/unit/test_app_validation.py`, `tests/integration/test_http_api.py`._
    - _Requirements: 4.1, 5.1–5.11, 15.1–15.17, 16.1–16.14, 18.7, 18.8_

  - [ ]* 8.5 Escrever property test da validação Unicode da pergunta
    - **Property 18: Validação da pergunta respeita Unicode e ordem de decisão**
    - Gerar strings Unicode e verificar 413 antes do trim, 400 para whitespace permitido e encaminhamento único somente do texto aparado nos demais casos.
    - **Validates: Requirements 15.5, 15.6, 15.7, 15.8**
    - _Arquivos/componentes: `tests/properties/test_property_18_question_validation.py`._

  - [ ]* 8.6 Escrever property test da não exposição em erros públicos
    - **Property 19: Erros públicos obedecem allowlist e não vazam dados**
    - Gerar erros/exceções com canários de segredo, caminho, texto, chunk, prompt e contexto e verificar envelope exato sem canário ou stack trace.
    - **Validates: Requirements 15.16, 16.13, 18.8, 18.10**
    - _Arquivos/componentes: `tests/properties/test_property_19_public_errors.py`._

- [ ] 9. Implementar a interface web segura e responsiva
  - [~] 9.1 Criar a estrutura HTML acessível da interface
    - Criar título/subtítulo, formulário de pergunta, áreas separadas de resposta/fontes, área de base, seletor `.zip`, importação, status, avisos e contadores nomeados com labels e regiões `aria-live`.
    - Não incorporar conteúdo documental no template nem usar framework frontend.
    - _Arquivos/componentes: `templates/index.html`._
    - _Requirements: 2.2, 2.12, 4.2–4.6, 17.14, 21.1, 22.2_

  - [~] 9.2 Implementar o layout CSS sem overflow em 1280 px
    - Criar container/grid responsivo com `box-sizing`, `minmax(0, 1fr)`, controles limitados ao viewport, quebra segura de texto e `white-space: pre-wrap` para respostas.
    - Garantir ausência de sobreposição, corte/rolagem horizontal e manter estados disabled/status visualmente identificáveis.
    - _Arquivos/componentes: `static/style.css`._
    - _Requirements: 2.12, 4.7, 17.1, 17.2, 17.7, 17.8, 17.16_

  - [~] 9.3 Implementar chat e upload com JavaScript puro e renderização literal
    - Implementar flags independentes, `fetch()` para ambas as rotas, estados `Consultando...`, `Processando...` e `Concluído`, desabilitação/reabilitação em `finally` e bloqueio de submit duplicado.
    - Validar schemas de sucesso/erro, preservar pergunta/arquivo em falhas, atualizar/limpar fontes, avisos e contadores e tratar falha de rede/corpo incompatível.
    - Inserir todo valor não confiável somente por `textContent`/`createTextNode`, nunca `innerHTML`, mantendo quebras de linha visuais.
    - _Arquivos/componentes: `static/script.js`._
    - _Requirements: 17.1–17.19, 18.9_

  - [ ]* 9.4 Escrever testes de contrato, XSS e layout da interface
    - Testar estados, flags, preservação de inputs, ordem/formato de fontes e avisos, contratos incompatíveis e payloads `<script>`/atributos/eventos sem nós executáveis.
    - Verificar `scrollWidth <= clientWidth` e ausência de interseções a 1280 px em teste de navegador marcado `browser`; manter os testes estáticos/contratuais na suíte padrão sem rede.
    - _Arquivos/componentes: `tests/frontend/test_ui_contract.py`, `tests/frontend/test_ui_browser.py`._
    - _Requirements: 4.2–4.7, 17.1–17.19, 18.9_

- [ ] 10. Completar documentação e verificações dos artefatos de entrega
  - [~] 10.1 Escrever o guia de operação completo do MVP
    - Criar `README.md` com pré-requisitos, venv/instalação, Ollama/modelos locais, preparação offline do embedding, `.env`, todas as variáveis/regras, estrutura, execução, uso, privacidade, limitações e troubleshooting acionável.
    - Incluir fluxo inicial com ZIP de exatamente dois PDFs, pergunta coberta, resultados esperados e procedimento final numerado com comandos exatos; documentar que obtenção inicial pode usar internet, mas a operação não.
    - Documentar a garantia de processo único e a estratégia atômica efetivamente implementada sem expor detalhes desnecessários ao usuário.
    - _Arquivos/componentes: `README.md`._
    - _Requirements: 1.8, 2.14, 18.11, 20.1–20.15, 21.1–21.9, 22.1–22.6_

  - [ ]* 10.2 Escrever testes dos manifestos, ambiente e documentação obrigatória
    - Verificar versões exatas, dependências diretamente importadas, Python suportado, uma ocorrência de cada variável/default em `.env.example`, estrutura obrigatória e seções/comandos/resultados normativos do README.
    - Importar os módulos em ambiente de teste instalado e falhar em dependência de produção ausente, sem baixar modelos nem acessar internet.
    - _Arquivos/componentes: `tests/unit/test_delivery_artifacts.py`._
    - _Requirements: 2.13, 2.14, 3.3, 20.1–20.15_

- [ ] 11. Integrar e validar o projeto executável completo
  - [ ]* 11.1 Escrever smoke ponta a ponta automatizado com dependências locais fake
    - Gerar ZIP com dois PDFs distintos, importar pela API, confirmar contagens, consultar pergunta coberta e pergunta insuficiente e validar respostas/fontes/UI usando embedding/store/generator determinísticos.
    - Executar sem modelo real, Ollama real, internet ou API paga e confirmar que nenhum histórico de pergunta/resposta é persistido.
    - _Arquivos/componentes: `tests/integration/test_end_to_end.py`._
    - _Requirements: 1.1–1.8, 21.1–21.9, 22.1–22.6_

  - [ ]* 11.2 Escrever testes estáticos e dinâmicos de segurança/privacidade
    - Falhar se produção usar `innerHTML`, `eval`, `exec`, subprocess/shell, `trust_remote_code=True`, `chromadb.HttpClient`, URL externa ou leitura de ambiente fora da configuração.
    - Bloquear sockets externos nos testes e verificar permissões, ausência de egress, ausência de perguntas/respostas/texto em manifesto/logs e redaction de canários.
    - _Arquivos/componentes: `tests/integration/test_security_guards.py`._
    - _Requirements: 1.3–1.7, 2.15, 2.16, 18.3–18.10, 21.9, 22.6_

  - [ ]* 11.3 Escrever smoke opt-in com modelos locais reais, sem download implícito
    - Criar teste marcado `real_models` que seja ignorado por padrão, exija opt-in explícito, use `local_files_only`, verifique dimensão/finitude e execute fluxo mínimo contra Ollama já iniciado.
    - Falhar com orientação acionável se o opt-in for solicitado sem artefatos locais; nunca tentar baixar modelo nem chamar host não loopback.
    - _Arquivos/componentes: `tests/smoke/test_offline_real_models.py`._
    - _Requirements: 1.5, 10.4–10.7, 13.1, 20.4, 20.15, 22.1, 22.4, 22.6_

- [~] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP; production and wiring tasks are never optional.
- Every property task must contain one Hypothesis test with at least 100 examples and the comment `Feature: erp-ai-support, Property {number}: {property_text}`.
- The standard test command must run without internet, real sentence-transformers models, a real Ollama instance or paid APIs. Tests marked `real_models` and `browser` are explicit opt-ins and never download artifacts implicitly.
- Tests use temporary directories, fakes and local disposable servers; no fixture may persist documents, questions or answers outside its temporary workspace.
- The Chroma adapter must target the exact pinned version. Absence of a conditional transaction API is not a blocker and must activate the documented process-single fallback using journal, exclusive visibility lock, compensation and startup recovery, without broadening the MVP.
- Requirement references are granular for traceability; checkpoints provide incremental validation without adding deployment, user-acceptance or approval work.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.4"] },
    { "id": 1, "tasks": ["1.3"] },
    { "id": 2, "tasks": ["1.5", "2.1", "4.1"] },
    { "id": 3, "tasks": ["3.1", "4.2"] },
    { "id": 4, "tasks": ["3.2", "4.3"] },
    { "id": 5, "tasks": ["3.3", "4.4", "4.5", "4.6"] },
    { "id": 6, "tasks": ["5.1"] },
    { "id": 7, "tasks": ["2.2", "2.3", "2.4", "2.5", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "5.2", "5.3", "5.4", "5.5", "5.6"] },
    { "id": 8, "tasks": ["7.1"] },
    { "id": 9, "tasks": ["7.2"] },
    { "id": 10, "tasks": ["7.3"] },
    { "id": 11, "tasks": ["7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10", "8.1"] },
    { "id": 12, "tasks": ["8.2"] },
    { "id": 13, "tasks": ["8.3"] },
    { "id": 14, "tasks": ["8.4", "8.5", "8.6", "9.1", "9.2", "9.3"] },
    { "id": 15, "tasks": ["9.4", "10.1", "11.1", "11.2", "11.3"] },
    { "id": 16, "tasks": ["10.2"] }
  ]
}
```
