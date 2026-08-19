# ERP AI Support

Assistente de suporte ao ERP totalmente local. Sem APIs externas, sem envio de
dados para fora da máquina.

---

## Pré-requisitos

| Componente | Versão mínima |
|------------|--------------|
| Python | 3.11 |
| [Ollama](https://ollama.com/) | instalado e em execução local |

Ollama deve estar acessível em `http://localhost:11434` (padrão) ou no endereço
configurado em `OLLAMA_URL`.

---

## Instalação

```bash
# 1. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# ou: .venv\Scripts\activate   # Windows

# 2. Instale as dependências
pip install -r requirements.txt
```

---

## Preparação offline dos modelos de embedding

O modelo de embeddings precisa ser baixado **uma única vez** com acesso à
internet. Após o download, a aplicação funciona completamente offline.

```bash
# Com a internet disponível (apenas na primeira vez):
python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("Modelo salvo no cache do Hugging Face.")
EOF
```

Na inicialização seguinte (sem internet), a aplicação carrega automaticamente
o modelo do cache local (`local_files_only=True`).

Para o modelo de geração, execute uma vez:

```bash
ollama pull qwen3:8b
```

---

## Configuração

Copie o arquivo de exemplo e ajuste apenas os valores que precisar alterar:

```bash
cp .env.example .env
```

As variáveis com seus valores padrão e restrições:

| Variável | Padrão | Descrição |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Endereço HTTP(S) do Ollama; host deve ser `localhost`, `127.0.0.1` ou `::1` |
| `OLLAMA_MODEL` | `qwen3:8b` | Modelo de geração instalado no Ollama |
| `CHROMA_PATH` | `./data/chroma` | Diretório da base vetorial persistente (criado automaticamente) |
| `CHROMA_COLLECTION` | `erp_ai_support` | Nome da coleção vetorial |
| `UPLOAD_FOLDER` | `./documents/uploads` | Diretório temporário de uploads (criado automaticamente) |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embeddings |
| `TOP_K` | `6` | Candidatos recuperados por pergunta (5–8) |
| `CHUNK_SIZE` | `800` | Tamanho máximo de cada chunk em caracteres (500–1000) |
| `CHUNK_OVERLAP` | `150` | Sobreposição entre chunks (0 a `CHUNK_SIZE - 1`) |
| `RELEVANCE_THRESHOLD` | `0.30` | Similaridade mínima aceita (-1 a 1) |
| `MAX_UPLOAD_MB` | `100` | Limite do arquivo ZIP compactado em MiB |
| `MAX_ZIP_ENTRIES` | `1000` | Quantidade máxima de entradas no ZIP |
| `MAX_ZIP_ENTRY_MB` | `100` | Limite por entrada descompactada em MiB |
| `MAX_UNCOMPRESSED_MB` | `500` | Limite total descompactado em MiB |
| `MAX_COMPRESSION_RATIO` | `100` | Razão máxima de compressão |
| `MAX_QUESTION_CHARS` | `2000` | Comprimento máximo da pergunta em pontos de código Unicode |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Timeout das chamadas ao Ollama (1–600 s) |
| `MAX_ANSWER_TOKENS` | `500` | Limite da resposta em tokens (64–2048) |
| `FLASK_HOST` | `127.0.0.1` | Host Flask local (`localhost`, `127.0.0.1` ou `::1`) |
| `FLASK_PORT` | `5000` | Porta Flask (1–65535) |
| `FLASK_DEBUG` | `false` | Modo de depuração (`true` ou `false`) |

---

## Execução

```bash
python app.py
```

Acesse `http://127.0.0.1:5000` no navegador (ou a porta configurada em
`FLASK_PORT`).

---

## Uso

### Importar documentos

1. Empacote os PDFs que descrevem o ERP em um arquivo `.zip`.
2. Na seção **Importar documentos**, clique em **Arquivo ZIP** e selecione o
   arquivo.
3. Clique em **Importar** e aguarde. Os contadores de documentos, páginas e
   chunks confirmam o sucesso.

Documentos duplicados (mesmo conteúdo já indexado) geram um aviso mas não
causam erro.

### Fazer perguntas

1. Na seção **Consultar base de conhecimento**, escreva sua pergunta em
   português.
2. Clique em **Consultar**.
3. A resposta é exibida na área de texto. As fontes (documento e página)
   aparecem abaixo da resposta.

---

## Privacidade

- Todos os modelos de IA rodam **localmente**.
- Nenhum dado (documentos, perguntas, respostas, embeddings) é enviado para
  fora da máquina.
- A base vetorial é um arquivo SQLite + Chroma em `CHROMA_PATH`.

---

## Limitações

- Somente documentos PDF são indexados. Outros formatos dentro do ZIP são
  ignorados com aviso.
- Páginas escaneadas (imagens sem camada de texto) não são indexadas. Aplique
  OCR antes de importar.
- O servidor Flask é de uso local (single-process). Não use em produção com
  múltiplos usuários simultâneos sem um servidor WSGI adequado.
- Apenas uma instância pode rodar ao mesmo tempo no mesmo `CHROMA_PATH`.

---

## Troubleshooting

| Sintoma | Causa provável | Solução |
|---|---|---|
| `embedding_model_missing` | Modelo de embeddings não baixado | Execute o download offline descrito acima |
| `ollama_unavailable` | Ollama não está rodando | Execute `ollama serve` |
| `ollama_model_missing` | Modelo de geração não instalado | Execute `ollama pull qwen3:8b` |
| `application_already_running` | Outra instância usa o mesmo `CHROMA_PATH` | Encerre a outra instância antes de iniciar |
| `recovery_required` | Processo interrompido durante indexação | Reinicie a aplicação; ela recupera automaticamente |
| `invalid_config` | Valor inválido no `.env` | Corrija o valor indicado na mensagem e reinicie |
| `data_path_invalid` | Diretório sem permissão de escrita | Verifique as permissões de `CHROMA_PATH` e `UPLOAD_FOLDER` |
