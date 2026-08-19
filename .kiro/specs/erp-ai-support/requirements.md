# Requirements Document

## Introduction

O ERP AI Support é uma aplicação web independente para suporte interno a um ERP legado. O MVP transforma documentos PDF internos, enviados exclusivamente em arquivos ZIP, em uma base de conhecimento vetorial local. Operadores de suporte podem formular perguntas em português do Brasil e receber respostas concisas, didáticas e fundamentadas somente nos trechos recuperados, acompanhadas pelos nomes reais dos PDFs e pelos números de página correspondentes.

O MVP não se integra ao ERP, não consulta o banco de dados do ERP, não executa código e não envia documentos, perguntas ou respostas a serviços externos. Depois da instalação das dependências e dos modelos, todo o fluxo deve funcionar localmente e sem conexão com a internet, com prioridade para Ubuntu/Linux.

## Glossary

- **MVP**: Primeira versão executável limitada à ingestão de PDFs por ZIP, indexação vetorial local, consulta semântica, geração local fundamentada e apresentação de fontes.
- **Ambiente_Local**: Computador no qual a aplicação, os modelos, os documentos e os dados persistentes são executados e armazenados sem serviço hospedado.
- **Operador_de_Suporte**: Pessoa que formula perguntas sobre o ERP na interface web.
- **Administrador_da_Base**: Pessoa que importa documentos internos para a base de conhecimento.
- **ERP_AI_Support**: Sistema completo definido neste documento.
- **Aplicacao_Web**: Componente Flask que expõe a interface e os endpoints HTTP do ERP_AI_Support.
- **Interface_Web**: Interface desktop construída com HTML, CSS e JavaScript puro.
- **Gerenciador_de_Configuracao**: Componente central que carrega, converte e valida as variáveis de ambiente.
- **Servico_de_Ingestao**: Componente que coordena validação do ZIP, leitura dos PDFs, chunking, embeddings e persistência.
- **Validador_de_Arquivo**: Componente que verifica formato, caminhos e limites de segurança de um arquivo ZIP antes e durante a extração.
- **Extrator_de_PDF**: Componente que abre PDFs com PyMuPDF e extrai texto página a página.
- **Servico_de_Chunking**: Componente que divide o texto extraído em trechos sobrepostos.
- **Servico_Local_de_Embeddings**: Componente sentence-transformers que produz vetores no Ambiente_Local.
- **Armazenamento_Vetorial_Local**: Componente ChromaDB persistente que armazena e consulta vetores e seus registros de origem.
- **Servico_de_Recuperacao**: Componente que transforma uma pergunta em embedding, pesquisa trechos e aplica o critério de relevância.
- **Servico_RAG**: Componente que monta o Contexto_Recuperado, solicita geração local e compõe a resposta da API.
- **Servico_Local_de_Geracao**: Componente que se comunica somente com uma instância local do Ollama.
- **Base_de_Conhecimento**: Conjunto persistente de Chunks e embeddings aceitos pelo processo de ingestão.
- **Upload**: Requisição multipart que contém um único arquivo ZIP no campo `file`.
- **Arquivo_ZIP**: Arquivo compactado válido no formato ZIP, usado como único contêiner aceito para importação.
- **Entrada_ZIP**: Arquivo ou diretório declarado dentro de um Arquivo_ZIP.
- **Entrada_Insegura**: Entrada_ZIP absoluta, com travessia `..`, prefixo de unidade, byte nulo, link simbólico, arquivo especial ou destino resolvido fora do diretório temporário.
- **Zip_Slip**: Escrita de uma Entrada_ZIP fora do diretório temporário autorizado por manipulação de caminho.
- **Zip_Bomb**: Arquivo_ZIP que excede os limites configurados de quantidade, tamanho descompactado ou razão de compressão.
- **Limites_de_Upload**: Limites configuráveis de bytes compactados, quantidade de entradas, bytes por entrada, bytes descompactados totais e razão de compressão.
- **Documento_PDF**: Arquivo com extensão `.pdf`, conteúdo que pode ser aberto pelo PyMuPDF e páginas enumeráveis.
- **Nome_Exibivel_do_Documento**: Caminho relativo do PDF dentro do ZIP, normalizado para exibição, sem caminhos do sistema operacional, segmentos perigosos ou caracteres de controle.
- **Pagina**: Página física de um Documento_PDF, identificada internamente por índice iniciado em zero.
- **Numero_Humano_da_Pagina**: Índice interno da Pagina acrescido de um, usado na API e na Interface_Web.
- **Pagina_Sem_Texto**: Pagina cujo texto extraído é vazio ou contém somente espaços em branco.
- **OCR**: Reconhecimento óptico de caracteres; capacidade explicitamente fora do MVP.
- **Texto_Extraido**: Sequência de caracteres devolvida pelo PyMuPDF para uma Pagina.
- **Chunk**: Substring contígua do Texto_Extraido de uma única Pagina, associada ao documento e à página de origem.
- **Tamanho_do_Chunk**: Quantidade máxima configurada de caracteres de um Chunk.
- **Sobreposicao_do_Chunk**: Quantidade configurada de caracteres repetidos entre Chunks adjacentes da mesma Pagina.
- **Identidade_do_Documento**: Hash SHA-256 calculado sobre os bytes originais de um Documento_PDF.
- **Identidade_do_Chunk**: Identificador determinístico derivado da Identidade_do_Documento, Numero_Humano_da_Pagina, posição do Chunk e versão do esquema de chunking.
- **Embedding**: Vetor numérico produzido localmente a partir de um texto.
- **Modelo_de_Embedding**: Modelo sentence-transformers configurado para produzir embeddings de documentos e perguntas.
- **Espaco_Vetorial**: Combinação do Modelo_de_Embedding, dimensão dos vetores, normalização e métrica de similaridade.
- **ChromaDB**: Banco vetorial embarcado e persistente usado pelo Armazenamento_Vetorial_Local.
- **Candidato_Recuperado**: Chunk retornado pela consulta vetorial antes da aplicação do limiar.
- **TOP_K**: Quantidade máxima configurada de Candidatos_Recuperados por pergunta.
- **Pontuacao_de_Relevancia**: Similaridade de cosseno entre a pergunta e um Chunk, na escala de -1 a 1, em que valores maiores indicam maior similaridade.
- **Limiar_de_Relevancia**: Menor Pontuacao_de_Relevancia aceita para incluir um Candidato_Recuperado no contexto.
- **Contexto_Recuperado**: Conjunto ordenado de Chunks que passaram pelo Limiar_de_Relevancia e que será fornecido ao modelo de geração.
- **Resposta_Fundamentada**: Resposta em português do Brasil cujas afirmações factuais são sustentadas exclusivamente pelo Contexto_Recuperado.
- **Resposta_de_Insuficiencia**: Texto exato `Não encontrei informação suficiente na base de conhecimento para responder a esta pergunta.`
- **Fonte**: Par `{document, page}` derivado de um Chunk do Contexto_Recuperado, em que `document` é o Nome_Exibivel_do_Documento e `page` é o Numero_Humano_da_Pagina.
- **Ollama**: Servidor local de modelos de linguagem acessado pelo Servico_Local_de_Geracao.
- **Modelo_de_Geracao_Configurado**: Modelo Ollama selecionado pela variável `OLLAMA_MODEL`.
- **Instrucao_Nao_Confiavel**: Texto presente em pergunta, documento ou contexto que tenta comandar o sistema, alterar regras ou provocar execução.
- **Transacao_de_Ingestao**: Conjunto de alterações no índice associado a um único Upload.
- **Aviso**: Mensagem em português do Brasil sobre uma condição recuperável, sem stack trace e sem caminho interno do sistema operacional.
- **Resposta_de_Erro**: Objeto JSON com código estável e mensagem em português do Brasil, sem stack trace, segredo ou detalhe interno.
- **Manifesto_de_Dependencias**: Arquivo `requirements.txt` com as dependências Python necessárias.
- **Exemplo_de_Ambiente**: Arquivo `.env.example` sem segredos e com todas as variáveis configuráveis documentadas.
- **Guia_de_Operacao**: Arquivo `README.md` com instalação, execução, uso, limitações e solução de problemas.

## Requirements

### Requisito 1: Escopo e operação local

**História do Usuário:** Como gestor de suporte, quero uma aplicação de conhecimento independente, para que a equipe consulte documentação do ERP sem criar dependência operacional com o ERP legado.

#### Critérios de Aceitação

1. THE ERP_AI_Support SHALL operar como uma aplicação web independente para suporte interno baseado em Documento_PDF.
2. WHILE respondendo a uma pergunta, THE ERP_AI_Support SHALL usar exclusivamente o Contexto_Recuperado da Base_de_Conhecimento como fonte de afirmações factuais.
3. WHILE executando o MVP, THE ERP_AI_Support SHALL operar sem estabelecer comunicação com qualquer aplicação, banco de dados, API ou ambiente de execução do ERP legado.
4. WHEN o ERP_AI_Support receber código-fonte, comando ou script como parte de um Documento_PDF ou de uma pergunta, THE ERP_AI_Support SHALL processar esse conteúdo exclusivamente como texto sem executá-lo.
5. WHILE as dependências e os modelos estiverem instalados no Ambiente_Local, THE ERP_AI_Support SHALL executar ingestão, recuperação e geração sem estabelecer comunicação com qualquer serviço fora do Ambiente_Local.
6. WHEN o ERP_AI_Support solicitar geração, THE ERP_AI_Support SHALL comunicar-se exclusivamente com o Ollama no endereço local configurado em `OLLAMA_URL`.
7. THE ERP_AI_Support SHALL armazenar documentos processados, embeddings e índice exclusivamente no Ambiente_Local.
8. THE ERP_AI_Support SHALL iniciar por comando Python sem exigir contêiner.

### Requisito 2: Stack e estrutura obrigatórias

**História do Usuário:** Como desenvolvedor responsável pelo MVP, quero uma stack simples e prescrita, para que a aplicação seja compreensível e executável em Ubuntu/Linux.

#### Critérios de Aceitação

1. THE Aplicacao_Web SHALL usar Python em versão igual ou superior a 3.11.0 e inferior a 4.0.0 com Flask.
2. THE Interface_Web SHALL usar exclusivamente HTML, CSS e JavaScript puro no navegador.
3. THE Extrator_de_PDF SHALL usar PyMuPDF para leitura página a página.
4. THE Servico_Local_de_Embeddings SHALL usar sentence-transformers.
5. THE Armazenamento_Vetorial_Local SHALL usar ChromaDB em modo persistente embarcado.
6. THE Servico_Local_de_Geracao SHALL usar Ollama no Ambiente_Local.
7. THE Servico_RAG SHALL integrar Servico_Local_de_Embeddings, Armazenamento_Vetorial_Local e Servico_Local_de_Geracao por meio de módulos do ERP_AI_Support, sem delegar a coordenação entre esses componentes a biblioteca ou componente externo ao ERP_AI_Support.
8. THE ERP_AI_Support SHALL operar como uma única Aplicacao_Web Flask composta por módulos Python internos, sem dividir esses módulos em aplicações independentes ou microsserviços e sem usar Kubernetes.
9. THE ERP_AI_Support SHALL fornecer `app.py` com criação da aplicação e rotas `GET /`, `POST /chat` e `POST /upload`.
10. THE ERP_AI_Support SHALL fornecer `ingest.py` com o fluxo do Servico_de_Ingestao desde a validação do Arquivo_ZIP até a persistência dos Chunks na Base_de_Conhecimento.
11. THE ERP_AI_Support SHALL fornecer `rag.py` com o Servico_RAG responsável pela recuperação de Chunks, pela montagem do Contexto_Recuperado, pela solicitação de geração ao Servico_Local_de_Geracao e pela composição das Fontes.
12. THE ERP_AI_Support SHALL fornecer `templates/index.html`, `static/style.css` e `static/script.js` para a Interface_Web.
13. THE ERP_AI_Support SHALL fornecer os diretórios `documents/uploads` e `data/chroma` para, respectivamente, armazenamento temporário de Arquivos_ZIP recebidos por Upload e persistência do Armazenamento_Vetorial_Local.
14. THE ERP_AI_Support SHALL fornecer Manifesto_de_Dependencias, Exemplo_de_Ambiente e Guia_de_Operacao.
15. THE ERP_AI_Support SHALL implementar o backend exclusivamente em Python, sem backend Node.js.
16. WHILE o ERP_AI_Support estiver em execução, THE ERP_AI_Support SHALL usar somente bibliotecas instaladas e componentes executados no Ambiente_Local, sem consumir APIs externas pagas.

### Requisito 3: Configuração centralizada e validada

**História do Usuário:** Como administrador local, quero controlar modelos, caminhos e limites por variáveis de ambiente, para que alterações operacionais não exijam edição de código.

#### Perfil de Configuração do MVP

| Variável | Valor de exemplo/padrão | Regra de validação |
|---|---:|---|
| `OLLAMA_URL` | `http://localhost:11434` | URL HTTP ou HTTPS cujo host seja `localhost`, `127.0.0.1` ou `::1` |
| `OLLAMA_MODEL` | `qwen3:8b` | texto não vazio |
| `CHROMA_PATH` | `./data/chroma` | caminho local não vazio |
| `CHROMA_COLLECTION` | `erp_ai_support` | nome não vazio |
| `UPLOAD_FOLDER` | `./documents/uploads` | caminho local não vazio |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | identificador não vazio |
| `TOP_K` | `6` | inteiro entre 5 e 8, inclusive |
| `CHUNK_SIZE` | `800` | inteiro entre 500 e 1000 caracteres, inclusive |
| `CHUNK_OVERLAP` | `150` | inteiro entre 0 e `CHUNK_SIZE - 1`, inclusive |
| `RELEVANCE_THRESHOLD` | `0.30` | número entre -1 e 1, inclusive |
| `MAX_UPLOAD_MB` | `100` | número positivo |
| `MAX_ZIP_ENTRIES` | `1000` | inteiro positivo |
| `MAX_ZIP_ENTRY_MB` | `100` | número positivo |
| `MAX_UNCOMPRESSED_MB` | `500` | número positivo e maior ou igual a `MAX_ZIP_ENTRY_MB` |
| `MAX_COMPRESSION_RATIO` | `100` | número maior ou igual a 1 |
| `MAX_QUESTION_CHARS` | `2000` | inteiro positivo |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | inteiro entre 1 e 600, inclusive |
| `MAX_ANSWER_TOKENS` | `500` | inteiro entre 64 e 2048, inclusive |
| `FLASK_HOST` | `127.0.0.1` | `localhost`, `127.0.0.1` ou `::1` |
| `FLASK_PORT` | `5000` | inteiro entre 1 e 65535, inclusive |
| `FLASK_DEBUG` | `false` | valor booleano |

#### Critérios de Aceitação

1. WHEN a Aplicacao_Web iniciar, THE Gerenciador_de_Configuracao SHALL carregar as variáveis do Perfil de Configuração do MVP a partir das variáveis de ambiente e do arquivo `.env` por meio de python-dotenv, com precedência das variáveis de ambiente sobre os valores do arquivo `.env`.
2. IF uma variável do Perfil de Configuração do MVP estiver ausente tanto das variáveis de ambiente quanto do arquivo `.env`, THEN THE Gerenciador_de_Configuracao SHALL aplicar o valor correspondente da coluna `Valor de exemplo/padrão` como valor padrão.
3. THE Exemplo_de_Ambiente SHALL declarar uma única vez cada variável do Perfil de Configuração do MVP com o valor correspondente da coluna `Valor de exemplo/padrão` e sem valor secreto.
4. WHEN o carregamento das variáveis terminar sem erro, THE Gerenciador_de_Configuracao SHALL disponibilizar a todos os componentes uma única configuração contendo cada valor sem espaços em branco nas extremidades, convertido para o tipo indicado e aprovado por todas as regras de validação aplicáveis.
5. IF uma variável não puder ser convertida para o tipo indicado, resultar vazia após a remoção de espaços em branco quando sua regra exigir valor não vazio ou violar qualquer outra regra de validação, THEN THE Gerenciador_de_Configuracao SHALL rejeitar integralmente a configuração por meio de uma Resposta_de_Erro que nomeie a variável inválida, sem disponibilizar a configuração aos componentes.
6. IF `CHUNK_OVERLAP` for maior ou igual a `CHUNK_SIZE`, THEN THE Gerenciador_de_Configuracao SHALL rejeitar a configuração antes de iniciar uma ingestão.
7. IF o Espaco_Vetorial configurado diferir do Espaco_Vetorial registrado em uma coleção existente, THEN THE Gerenciador_de_Configuracao SHALL bloquear ingestão e consulta nessa coleção por meio de uma Resposta_de_Erro que indique a incompatibilidade e oriente a recriação do índice, sem alterar a coleção existente.
8. IF o caminho configurado em `CHROMA_PATH` ou `UPLOAD_FOLDER`, interpretado a partir do diretório no qual a Aplicacao_Web foi iniciada no caso de valor relativo, não existir e o diretório pai existente mais próximo permitir escrita pelo processo, THEN THE Gerenciador_de_Configuracao SHALL criar o caminho configurado como diretório antes de disponibilizar a configuração aos componentes.
9. IF o caminho configurado em `CHROMA_PATH` ou `UPLOAD_FOLDER` existir sem ser um diretório, não puder ser criado como diretório ou não permitir leitura e escrita pelo processo, THEN THE Gerenciador_de_Configuracao SHALL rejeitar integralmente a configuração por meio de uma Resposta_de_Erro que nomeie a variável afetada sem expor o caminho absoluto, sem disponibilizar a configuração aos componentes.
10. THE ERP_AI_Support SHALL obter todos os valores definidos no Perfil de Configuração do MVP exclusivamente da configuração validada fornecida pelo Gerenciador_de_Configuracao.
11. WHEN o Gerenciador_de_Configuracao converter `FLASK_DEBUG`, THE Gerenciador_de_Configuracao SHALL aceitar como valor booleano somente `true` ou `false`, sem distinção entre letras maiúsculas e minúsculas.
12. THE Gerenciador_de_Configuracao SHALL interpretar cada unidade `MB` de `MAX_UPLOAD_MB`, `MAX_ZIP_ENTRY_MB` e `MAX_UNCOMPRESSED_MB` como exatamente 1.048.576 bytes.
13. IF o arquivo `.env` existir e não puder ser lido ou interpretado por python-dotenv, THEN THE Gerenciador_de_Configuracao SHALL rejeitar integralmente a configuração por meio de uma Resposta_de_Erro que indique a falha de carregamento sem expor o caminho absoluto do arquivo, sem disponibilizar a configuração aos componentes.

### Requisito 4: Interface e rota inicial

**História do Usuário:** Como Operador_de_Suporte, quero abrir uma página local simples, para que eu possa consultar a base e acompanhar importações sem ferramentas técnicas.

#### Critérios de Aceitação

1. WHEN um cliente solicitar `GET /`, THE Aplicacao_Web SHALL responder com a Interface_Web e status HTTP 200.
2. THE Interface_Web SHALL exibir o título `ERP AI Support`.
3. THE Interface_Web SHALL exibir o subtítulo `Suporte interno baseado em documentos`.
4. THE Interface_Web SHALL exibir uma área de pergunta com campo de texto e botão `Perguntar`.
5. THE Interface_Web SHALL exibir áreas separadas para Resposta_Fundamentada e Fontes.
6. THE Interface_Web SHALL exibir uma área `Base de conhecimento` com seletor de Arquivo_ZIP, botão `Importar`, indicador de status e três contadores identificados como `documents`, `pages` e `chunks`.
7. WHILE exibida em uma janela de navegador com área de conteúdo de 1280 pixels de largura, THE Interface_Web SHALL apresentar o título, o subtítulo, as áreas, os campos, os botões, o indicador de status e os contadores sem sobreposição entre elementos, sem corte horizontal e sem rolagem horizontal.

### Requisito 5: Contrato e validação inicial de upload

**História do Usuário:** Como Administrador_da_Base, quero importar exclusivamente arquivos ZIP válidos, para que a ingestão receba entradas previsíveis e limitadas.

#### Critérios de Aceitação

1. WHEN a Aplicacao_Web receber `POST /upload`, THE Aplicacao_Web SHALL aceitar como Upload somente uma requisição multipart contendo exatamente um arquivo no campo `file` e nenhum arquivo em outro campo.
2. IF a requisição recebida em `POST /upload` não for multipart, não contiver exatamente um arquivo no campo `file`, contiver qualquer arquivo em outro campo ou apresentar nome de arquivo ausente ou com zero caracteres, THEN THE Aplicacao_Web SHALL responder com status HTTP 400 e uma Resposta_de_Erro que indique a condição detectada.
3. IF o nome enviado não terminar em `.zip` sem distinção entre maiúsculas e minúsculas, THEN THE Aplicacao_Web SHALL responder com status HTTP 415 e uma Resposta_de_Erro.
4. IF o tipo MIME não pertencer à lista `application/zip`, `application/x-zip-compressed` ou `application/octet-stream`, THEN THE Aplicacao_Web SHALL responder com status HTTP 415 e uma Resposta_de_Erro.
5. IF o conteúdo do campo `file` não puder ser aberto como contêiner ZIP ou a enumeração integral de suas Entradas_ZIP não puder ser concluída sem erro, THEN THE Validador_de_Arquivo SHALL rejeitar a requisição antes de extrair qualquer Entrada_ZIP, com status HTTP 422 e uma Resposta_de_Erro que indique estrutura ZIP inválida.
6. IF a quantidade de bytes efetivamente recebidos no conteúdo do campo `file` exceder o produto de `MAX_UPLOAD_MB` por 1.048.576 bytes, THEN THE Aplicacao_Web SHALL interromper o recebimento, descartar o conteúdo recebido e responder com status HTTP 413 e uma Resposta_de_Erro que indique que o limite de tamanho compactado foi excedido.
7. WHEN as verificações de formato multipart, quantidade e nome de arquivo, extensão, tipo MIME, estrutura ZIP e tamanho compactado terminarem sem rejeição, THE Servico_de_Ingestao SHALL gravar o Arquivo_ZIP somente em uma área temporária exclusiva do Upload contida em `UPLOAD_FOLDER`.
8. WHEN uma Transacao_de_Ingestao terminar com sucesso ou erro, THE Servico_de_Ingestao SHALL remover integralmente a área temporária exclusiva associada, incluindo o Arquivo_ZIP, os arquivos extraídos e os diretórios temporários.
9. WHEN o Servico_de_Ingestao ler conteúdo de um Upload, THE Servico_de_Ingestao SHALL usar o conteúdo somente para descompactação, extração textual e indexação.
10. THE Aplicacao_Web SHALL aceitar somente Arquivo_ZIP como formato de importação do MVP.
11. IF uma requisição for rejeitada durante as verificações de formato multipart, quantidade ou nome de arquivo, extensão, tipo MIME, estrutura ZIP ou tamanho compactado, THEN THE Servico_de_Ingestao SHALL remover todo conteúdo temporário associado à requisição e preservar zero alterações na Base_de_Conhecimento.

### Requisito 6: Proteção de extração ZIP

**História do Usuário:** Como responsável pela segurança, quero validar integralmente os ZIPs antes da extração, para que arquivos maliciosos não escapem da área temporária nem esgotem recursos locais.

#### Critérios de Aceitação

1. WHEN o Validador_de_Arquivo abrir um Arquivo_ZIP, THE Validador_de_Arquivo SHALL inspecionar todas as Entradas_ZIP para detectar Entrada_Insegura e verificar, com base nos tamanhos declarados, a quantidade de entradas, o tamanho descompactado por entrada, o tamanho descompactado total e a razão de compressão antes de extrair a primeira entrada.
2. IF uma Entrada_Insegura estiver presente, THEN THE Validador_de_Arquivo SHALL rejeitar o Arquivo_ZIP inteiro antes de gravar conteúdo extraído.
3. WHEN o Validador_de_Arquivo calcular o destino resolvido de uma Entrada_ZIP, THE Validador_de_Arquivo SHALL confirmar que o destino é o próprio diretório temporário autorizado do Upload ou um caminho descendente desse diretório, sem considerar mera coincidência de prefixo entre caminhos como confinamento. *(Propriedade de correção: invariante de confinamento de caminho.)*
4. IF a quantidade de Entradas_ZIP exceder `MAX_ZIP_ENTRIES`, THEN THE Validador_de_Arquivo SHALL rejeitar o Upload com status HTTP 422.
5. IF o tamanho descompactado declarado de uma Entrada_ZIP exceder o produto de `MAX_ZIP_ENTRY_MB` por 1.048.576 bytes, THEN THE Validador_de_Arquivo SHALL rejeitar o Upload com status HTTP 422.
6. IF a soma dos tamanhos descompactados declarados exceder o produto de `MAX_UNCOMPRESSED_MB` por 1.048.576 bytes, THEN THE Validador_de_Arquivo SHALL rejeitar o Upload com status HTTP 422.
7. IF uma Entrada_ZIP declarar tamanho descompactado maior que 0 byte e tamanho compactado igual a 0 byte, ou se declarar tamanho compactado maior que 0 byte e o quociente do tamanho descompactado declarado pelo tamanho compactado declarado exceder `MAX_COMPRESSION_RATIO`, THEN THE Validador_de_Arquivo SHALL rejeitar o Upload com status HTTP 422.
8. WHILE extraindo uma Entrada_ZIP, THE Validador_de_Arquivo SHALL contabilizar separadamente os bytes descompactados efetivamente gravados para a entrada atual e o total cumulativo gravado para o Arquivo_ZIP.
9. IF os bytes descompactados disponíveis para gravação fizerem o total da Entrada_ZIP exceder o produto de `MAX_ZIP_ENTRY_MB` por 1.048.576 bytes ou o total cumulativo do Arquivo_ZIP exceder o produto de `MAX_UNCOMPRESSED_MB` por 1.048.576 bytes, THEN THE Validador_de_Arquivo SHALL interromper a extração antes de gravar qualquer byte acima do limite e rejeitar o Upload com status HTTP 422.
10. WHEN o Validador_de_Arquivo rejeitar um Zip_Bomb ou uma Entrada_Insegura, THE Aplicacao_Web SHALL retornar uma Resposta_de_Erro que indique rejeição por limite de segurança ou entrada insegura sem reproduzir qualquer caminho declarado no Arquivo_ZIP.
11. WHEN a extração integral de todas as Entradas_ZIP terminar sem falha ou rejeição, THE Validador_de_Arquivo SHALL disponibilizar ao Extrator_de_PDF somente arquivos regulares provenientes de Entradas_ZIP que não sejam Entrada_Insegura e cuja extração tenha respeitado os Limites_de_Upload.
12. IF a extração falhar ou for interrompida pela rejeição do Upload, THEN THE Servico_de_Ingestao SHALL remover todo conteúdo temporário já criado para o Upload.
13. IF uma falha ocorrer durante a extração, THEN THE Aplicacao_Web SHALL rejeitar o Upload com uma Resposta_de_Erro.
14. WHILE a extração integral do Arquivo_ZIP não tiver terminado sem falha ou rejeição, THE Validador_de_Arquivo SHALL disponibilizar zero arquivos ao Extrator_de_PDF.

### Requisito 7: Descoberta e extração de PDFs

**História do Usuário:** Como Administrador_da_Base, quero processar PDFs em qualquer subpasta do ZIP e receber avisos sobre documentos problemáticos, para que eu conheça a cobertura real da importação.

#### Critérios de Aceitação

1. WHEN um Arquivo_ZIP válido for extraído, THE Servico_de_Ingestao SHALL descobrir recursivamente, na raiz e em todas as subpastas, cada arquivo regular cujo nome termine em `.pdf`, sem distinção entre maiúsculas e minúsculas.
2. IF um arquivo regular extraído não tiver nome terminado em `.pdf`, sem distinção entre maiúsculas e minúsculas, THEN THE Servico_de_Ingestao SHALL ignorar o arquivo e adicionar um Aviso indicando que ele foi ignorado por não possuir a extensão `.pdf`.
3. IF nenhum arquivo regular com nome terminado em `.pdf`, sem distinção entre maiúsculas e minúsculas, estiver presente após a extração, THEN THE Aplicacao_Web SHALL responder com status HTTP 422, fornecer uma Resposta_de_Erro indicando que o Arquivo_ZIP não contém PDFs e preservar zero alterações do Upload na Base_de_Conhecimento.
4. WHEN o Extrator_de_PDF receber um arquivo regular descoberto cujo nome termine em `.pdf`, THE Extrator_de_PDF SHALL tentar abri-lo com PyMuPDF e enumerar suas Paginas antes de contabilizá-lo como Documento_PDF processado.
5. WHEN um Documento_PDF for aberto, THE Extrator_de_PDF SHALL extrair o Texto_Extraido separadamente para cada Pagina.
6. WHEN o Extrator_de_PDF registrar uma Pagina, THE Extrator_de_PDF SHALL associar o Numero_Humano_da_Pagina correspondente.
7. WHEN o Extrator_de_PDF produzir Texto_Extraido, THE Extrator_de_PDF SHALL preservar o texto devolvido por PyMuPDF como origem dos Chunks.
8. WHEN uma Pagina_Sem_Texto for detectada, THE Extrator_de_PDF SHALL adicionar um Aviso que identifique o Nome_Exibivel_do_Documento e o Numero_Humano_da_Pagina e informe que nenhum texto foi extraído e que o MVP não executa OCR.
9. WHEN uma Pagina_Sem_Texto for detectada, THE Servico_de_Chunking SHALL produzir zero Chunks para a Pagina_Sem_Texto.
10. IF o PyMuPDF não conseguir abrir um arquivo `.pdf` descoberto, enumerar todas as suas Paginas ou extrair o Texto_Extraido de qualquer Pagina devido a corrupção, criptografia sem acesso ou erro de leitura, THEN THE Extrator_de_PDF SHALL ignorar o arquivo inteiro, descartar todo Texto_Extraido já obtido dele e adicionar um Aviso que identifique o Nome_Exibivel_do_Documento e indique que o arquivo foi ignorado por falha de leitura.
11. WHEN um arquivo `.pdf` ignorado conforme o critério 10 coexistir com ao menos um Documento_PDF no mesmo Upload, THE Servico_de_Ingestao SHALL continuar a Transacao_de_Ingestao somente com os Documentos_PDF.
12. IF todos os arquivos regulares descobertos com nome terminado em `.pdf` forem ignorados conforme o critério 10, THEN THE Aplicacao_Web SHALL responder com status HTTP 422, fornecer uma Resposta_de_Erro indicando que nenhum PDF pôde ser lido e preservar zero alterações do Upload na Base_de_Conhecimento.
13. WHEN um arquivo regular com nome terminado em `.pdf` for descoberto, THE Servico_de_Ingestao SHALL produzir um Nome_Exibivel_do_Documento correspondente ao caminho relativo normalizado do arquivo dentro do Arquivo_ZIP.
14. WHEN um Documento_PDF possuir ao menos uma Pagina_Sem_Texto e ao menos uma Pagina que não seja Pagina_Sem_Texto, THE Servico_de_Ingestao SHALL indexar somente os Chunks das Paginas que não sejam Paginas_Sem_Texto.

### Requisito 8: Identidade, duplicação e idempotência

**História do Usuário:** Como Administrador_da_Base, quero evitar reprocessamento e duplicação do mesmo PDF, para que uploads repetidos não degradem a base nem removam conhecimento válido.

#### Critérios de Aceitação

1. WHEN o Servico_de_Ingestao ler um Documento_PDF candidato, THE Servico_de_Ingestao SHALL calcular a Identidade_do_Documento sobre os bytes originais antes de gerar Chunks.
2. WHEN dois PDFs tiverem bytes idênticos em caminhos ou Uploads distintos, THE Servico_de_Ingestao SHALL atribuir a mesma Identidade_do_Documento aos dois PDFs. *(Propriedade de correção: identidade estável.)*
3. IF um Documento_PDF com a mesma Identidade_do_Documento já tiver sido processado em uma Transacao_de_Ingestao concluída com sucesso no mesmo Espaco_Vetorial, com os mesmos valores de `CHUNK_SIZE` e `CHUNK_OVERLAP` e a mesma versão do esquema de chunking, THEN THE Servico_de_Ingestao SHALL pular a extração de texto, a geração de Chunks e Embeddings e a persistência de novos Chunks para essa ocorrência.
4. WHEN o Servico_de_Ingestao pular um Documento_PDF por duplicação, THE Servico_de_Ingestao SHALL adicionar exatamente um Aviso de duplicação para a ocorrência, identificando o Nome_Exibivel_do_Documento.
5. WHEN uma importação adicional do mesmo Documento_PDF ocorrer no mesmo Espaco_Vetorial, com os mesmos valores de `CHUNK_SIZE` e `CHUNK_OVERLAP` e a mesma versão do esquema de chunking, THE Armazenamento_Vetorial_Local SHALL manter inalterados o conjunto de Identidades_do_Chunk e a quantidade de Chunks confirmados associados à Identidade_do_Documento. *(Propriedade de correção: idempotência.)*
6. WHEN o Servico_de_Ingestao processar um Arquivo_ZIP com duas ou mais ocorrências da mesma Identidade_do_Documento, THE Servico_de_Ingestao SHALL indexar somente a primeira ocorrência que satisfizer a definição de Documento_PDF, segundo a ordem em que as respectivas Entradas_ZIP estiverem declaradas no Arquivo_ZIP.
7. WHEN um Documento_PDF reutilizar um Nome_Exibivel_do_Documento com bytes diferentes dos bytes do documento anterior, THE Servico_de_Ingestao SHALL tratá-lo como um Documento_PDF distinto conforme sua Identidade_do_Documento, sem substituir nem remover os Chunks confirmados associados ao documento anterior.
8. IF o processamento de um Documento_PDF falhar antes da conclusão bem-sucedida da Transacao_de_Ingestao, THEN THE Servico_de_Ingestao SHALL manter zero Chunks dessa tentativa como confirmados e não classificar uma ocorrência posterior da mesma Identidade_do_Documento como duplicata com base na tentativa que falhou.
9. WHEN uma Identidade_do_Chunk for gerada novamente a partir da mesma Identidade_do_Documento, do mesmo Numero_Humano_da_Pagina, da mesma posição inicial do Chunk e da mesma versão do esquema de chunking, THE Servico_de_Ingestao SHALL produzir o mesmo identificador. *(Propriedade de correção: determinismo.)*
10. WHEN uma Transacao_de_Ingestao de um Upload com duplicatas terminar com sucesso ou falha, THE Servico_de_Ingestao SHALL manter inalterados e consultáveis todos os Chunks confirmados antes do início dessa Transacao_de_Ingestao.

### Requisito 9: Chunking configurável e rastreável

**História do Usuário:** Como especialista de conhecimento, quero trechos sobrepostos com origem exata, para que a recuperação preserve contexto sem perder a página de referência.

#### Critérios de Aceitação

1. WHEN o Servico_de_Chunking receber o Texto_Extraido de uma Pagina que não seja Pagina_Sem_Texto, THE Servico_de_Chunking SHALL dividir o texto em uma sequência ordenada de Chunks não vazios, cada um com no máximo `CHUNK_SIZE` caracteres.
2. WHEN o Servico_de_Chunking receber o Texto_Extraido de uma Pagina que não seja Pagina_Sem_Texto com até `CHUNK_SIZE` caracteres, THE Servico_de_Chunking SHALL produzir exatamente um Chunk com posição inicial zero e texto idêntico ao Texto_Extraido completo.
3. WHEN o Texto_Extraido exceder `CHUNK_SIZE`, THE Servico_de_Chunking SHALL posicionar o primeiro Chunk no índice zero e cada Chunk subsequente exatamente `CHUNK_SIZE - CHUNK_OVERLAP` posições após o Chunk anterior.
4. WHEN dois Chunks adjacentes da mesma Pagina forem gerados, THE Servico_de_Chunking SHALL compartilhar exatamente as `CHUNK_OVERLAP` posições finais do Texto_Extraido cobertas pelo Chunk anterior como as primeiras posições cobertas pelo Chunk seguinte, preservando a mesma ordem e o mesmo conteúdo. *(Propriedade de correção: sobreposição.)*
5. IF `CHUNK_OVERLAP` for zero, THEN THE Servico_de_Chunking SHALL gerar Chunks adjacentes com intervalos de posições de origem disjuntos.
6. WHEN um Chunk for gerado, THE Servico_de_Chunking SHALL definir seu texto como a substring contígua do Texto_Extraido que começa na posição inicial associada e possui a mesma quantidade de caracteres que o texto do Chunk, sem inserção, remoção ou reordenação. *(Propriedade de correção: invariante de conteúdo.)*
7. WHEN uma Pagina que não seja Pagina_Sem_Texto for completamente dividida, THE Servico_de_Chunking SHALL cobrir cada caractere do Texto_Extraido em pelo menos um Chunk e encerrar a sequência no primeiro Chunk que contiver o último caractere do Texto_Extraido. *(Propriedade de correção: cobertura.)*
8. WHEN um Chunk for gerado, THE Servico_de_Chunking SHALL associar a Identidade_do_Documento e o Nome_Exibivel_do_Documento do Documento_PDF de origem, o Numero_Humano_da_Pagina da Pagina de origem, a posição inicial correspondente ao índice do primeiro caractere do Chunk no Texto_Extraido contado a partir de zero, o texto e a Identidade_do_Chunk.
9. THE Servico_de_Chunking SHALL impedir que um Chunk combine texto de duas Paginas diferentes.
10. WHEN o mesmo Texto_Extraido da mesma Pagina for processado novamente com os mesmos valores de `CHUNK_SIZE` e `CHUNK_OVERLAP`, THE Servico_de_Chunking SHALL produzir a mesma sequência ordenada de textos e posições iniciais. *(Propriedade de correção: determinismo.)*
11. WHEN a divisão das Paginas de um Upload terminar, THE Servico_de_Ingestao SHALL contabilizar para o Upload a soma das quantidades de Chunks de todas as sequências produzidas durante esse Upload.

### Requisito 10: Embeddings locais e espaço vetorial consistente

**História do Usuário:** Como Operador_de_Suporte, quero recuperação semântica adequada a português, para que perguntas e documentação com redações diferentes ainda possam ser relacionadas localmente.

#### Critérios de Aceitação

1. IF a variável `EMBEDDING_MODEL` não estiver definida, THEN THE Servico_Local_de_Embeddings SHALL usar o Modelo_de_Embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
2. WHEN o Servico_Local_de_Embeddings gerar embeddings de Chunks, THE Servico_Local_de_Embeddings SHALL usar o Modelo_de_Embedding configurado.
3. WHEN o Servico_Local_de_Embeddings gerar o embedding de uma pergunta, THE Servico_Local_de_Embeddings SHALL usar o mesmo Espaco_Vetorial usado pelos Chunks da coleção.
4. WHEN o Servico_Local_de_Embeddings processar texto, THE Servico_Local_de_Embeddings SHALL executar a inferência no Ambiente_Local sem transmitir o texto para serviço externo.
5. WHEN o Servico_Local_de_Embeddings carregar o Modelo_de_Embedding configurado, THE Servico_Local_de_Embeddings SHALL obter as dependências e os artefatos do modelo exclusivamente do Ambiente_Local.
6. IF o Modelo_de_Embedding configurado não estiver disponível no Ambiente_Local, THEN THE Servico_Local_de_Embeddings SHALL rejeitar a geração solicitada sem produzir ou persistir Embedding e retornar uma Resposta_de_Erro que identifique o modelo e oriente sua instalação local.
7. IF a geração de um Embedding não produzir um vetor com todos os componentes numéricos finitos e com quantidade de componentes igual à dimensão do Espaco_Vetorial da coleção, THEN THE Servico_Local_de_Embeddings SHALL abortar a ingestão ou consulta em andamento, retornar uma Resposta_de_Erro que indique a falha e não disponibilizar nem persistir o resultado da geração.
8. THE Armazenamento_Vetorial_Local SHALL registrar junto a cada coleção o Modelo_de_Embedding, a dimensão dos vetores, a normalização aplicada e a métrica de similaridade que compõem o Espaco_Vetorial.
9. THE Armazenamento_Vetorial_Local SHALL usar similaridade de cosseno para indexação e consulta.
10. WHEN um texto for codificado sob o Espaco_Vetorial de uma coleção, THE Servico_Local_de_Embeddings SHALL produzir um Embedding cuja quantidade de componentes seja exatamente igual à dimensão registrada junto à coleção. *(Propriedade de correção: invariante dimensional.)*

### Requisito 11: Persistência, concorrência e atomicidade do índice

**História do Usuário:** Como administrador local, quero um índice persistente e protegido contra alterações parciais, para que reinicializações e falhas não corrompam a base de conhecimento.

#### Critérios de Aceitação

1. THE Armazenamento_Vetorial_Local SHALL persistir a coleção ChromaDB no caminho configurado por `CHROMA_PATH`.
2. THE Armazenamento_Vetorial_Local SHALL operar sem serviço ChromaDB hospedado.
3. WHEN um Chunk for persistido, THE Armazenamento_Vetorial_Local SHALL armazenar Identidade_do_Chunk, texto, embedding, Identidade_do_Documento, Nome_Exibivel_do_Documento e Numero_Humano_da_Pagina.
4. WHEN a Aplicacao_Web reiniciar com os mesmos valores de `CHROMA_PATH` e `CHROMA_COLLECTION` e com Espaco_Vetorial igual ao registrado na coleção, THE Armazenamento_Vetorial_Local SHALL recuperar sem alteração e tornar novamente consultáveis todos e somente os Chunks de cada Transacao_de_Ingestao concluída com sucesso antes da reinicialização. *(Propriedade de correção: round trip de persistência.)*
5. WHEN uma Transacao_de_Ingestao concluir com sucesso, THE Armazenamento_Vetorial_Local SHALL tornar consultáveis todos e somente os Chunks produzidos para os Documentos_PDF desse Upload que não tenham sido ignorados nem pulados como duplicatas e manter sem alteração os Chunks confirmados antes do Upload que não sejam alvo de upsert.
6. IF a geração de qualquer Embedding ou qualquer leitura ou gravação necessária no ChromaDB falhar durante uma Transacao_de_Ingestao, THEN THE Armazenamento_Vetorial_Local SHALL remover todas as alterações não confirmadas dessa Transacao_de_Ingestao e restaurar qualquer registro preexistente alterado por ela.
7. IF uma Transacao_de_Ingestao falhar, THEN THE Armazenamento_Vetorial_Local SHALL manter a Base_de_Conhecimento exatamente no estado confirmado imediatamente anterior ao Upload, sem alteração persistida ou consultável proveniente desse Upload. *(Propriedade de correção: atomicidade sobre estado preexistente.)*
8. WHILE uma Transacao_de_Ingestao estiver modificando a Base_de_Conhecimento, THE Servico_de_Ingestao SHALL impedir que qualquer outra Transacao_de_Ingestao inicie uma modificação até a primeira concluir com sucesso ou falhar.
9. IF outro Upload solicitar modificação depois que uma Transacao_de_Ingestao iniciar sua primeira modificação da Base_de_Conhecimento e antes de ela concluir com sucesso ou falhar, THEN THE Aplicacao_Web SHALL responder com status HTTP 409 e uma Resposta_de_Erro.
10. IF uma pergunta não puder ler o ChromaDB por indisponibilidade ou um Upload não puder acessar o ChromaDB por indisponibilidade ou impossibilidade de gravação, THEN THE Aplicacao_Web SHALL responder à operação afetada com status HTTP 503 e uma Resposta_de_Erro que indique indisponibilidade ou impossibilidade de gravação da base vetorial local.
11. WHEN uma Identidade_do_Chunk já existir e a nova persistência apresentar os mesmos texto, Identidade_do_Documento, Nome_Exibivel_do_Documento e Numero_Humano_da_Pagina, com Embedding pertencente ao Espaco_Vetorial da coleção, THE Armazenamento_Vetorial_Local SHALL aplicar upsert ao registro existente e manter exatamente um registro com essa Identidade_do_Chunk. *(Propriedade de correção: idempotência de persistência.)*
12. WHILE uma Transacao_de_Ingestao tiver iniciado sua primeira modificação da Base_de_Conhecimento e ainda não tiver concluído com sucesso ou falha, THE Armazenamento_Vetorial_Local SHALL disponibilizar às consultas somente os Chunks confirmados antes do Upload, sem expor qualquer alteração dessa Transacao_de_Ingestao.
13. IF uma nova persistência reutilizar uma Identidade_do_Chunk existente e apresentar diferença em pelo menos um dos valores de texto, Identidade_do_Documento, Nome_Exibivel_do_Documento ou Numero_Humano_da_Pagina, THEN THE Aplicacao_Web SHALL rejeitar o Upload com uma Resposta_de_Erro que indique conflito de identidade e preservar sem alteração o registro existente.

### Requisito 12: Recuperação semântica e rejeição segura

**História do Usuário:** Como Operador_de_Suporte, quero que apenas trechos semanticamente relevantes sejam usados, para que o chatbot não pesquise todos os PDFs nem responda com contexto fraco.

#### Critérios de Aceitação

1. WHEN o Servico_de_Recuperacao receber uma pergunta validada, THE Servico_de_Recuperacao SHALL gerar um Embedding da pergunta no Espaco_Vetorial da coleção.
2. WHILE a coleção contiver pelo menos `TOP_K` Chunks, WHEN a geração do Embedding da pergunta for concluída, THE Servico_de_Recuperacao SHALL solicitar exatamente `TOP_K` Candidatos_Recuperados ao Armazenamento_Vetorial_Local.
3. WHILE a coleção contiver de 1 a `TOP_K - 1` Chunks, WHEN a geração do Embedding da pergunta for concluída, THE Servico_de_Recuperacao SHALL solicitar uma quantidade de Candidatos_Recuperados igual à quantidade de Chunks disponíveis.
4. WHEN Candidatos_Recuperados forem retornados, THE Servico_de_Recuperacao SHALL ordená-los por Pontuacao_de_Relevancia decrescente, preservando a ordem de retorno do Armazenamento_Vetorial_Local entre candidatos com pontuações iguais.
5. IF um Candidato_Recuperado tiver Pontuacao_de_Relevancia menor que `RELEVANCE_THRESHOLD`, THEN THE Servico_de_Recuperacao SHALL excluir o candidato do Contexto_Recuperado.
6. WHEN o Contexto_Recuperado for montado, THE Servico_de_Recuperacao SHALL incluir exatamente os Chunks originados dos Candidatos_Recuperados com Pontuacao_de_Relevancia maior ou igual a `RELEVANCE_THRESHOLD`, na ordem definida pelo critério 4. *(Propriedade de correção: confinamento da recuperação.)*
7. IF não houver Candidatos_Recuperados ou nenhum Candidato_Recuperado tiver Pontuacao_de_Relevancia maior ou igual a `RELEVANCE_THRESHOLD`, THEN THE Servico_RAG SHALL retornar a Resposta_de_Insuficiencia.
8. IF não houver Candidatos_Recuperados ou nenhum Candidato_Recuperado tiver Pontuacao_de_Relevancia maior ou igual a `RELEVANCE_THRESHOLD`, THEN THE Servico_RAG SHALL omitir a chamada ao Servico_Local_de_Geracao.
9. WHEN o Servico_RAG iniciar uma solicitação de geração com um Contexto_Recuperado não vazio, THE Servico_RAG SHALL enviar ao Modelo_de_Geracao_Configurado exatamente os Chunks do Contexto_Recuperado, na mesma ordem, sem anexar conteúdo dos PDFs que não pertença a esses Chunks.
10. THE Servico_de_Recuperacao SHALL limitar `TOP_K` ao intervalo configurável de 5 a 8, inclusive.
11. WHILE a Base_de_Conhecimento estiver vazia, WHEN o Servico_RAG receber uma pergunta validada, THE Servico_RAG SHALL retornar a Resposta_de_Insuficiencia com uma lista de Fontes vazia.

### Requisito 13: Geração local estritamente fundamentada

**História do Usuário:** Como Operador_de_Suporte, quero respostas curtas, didáticas e sem invenções, para que eu possa orientar usuários do ERP com base somente na documentação interna.

#### Critérios de Aceitação

1. WHEN o Servico_RAG solicitar geração para um Contexto_Recuperado não vazio, THE Servico_Local_de_Geracao SHALL enviar a solicitação ao endereço `OLLAMA_URL` com o Modelo_de_Geracao_Configurado.
2. IF o Servico_Local_de_Geracao não estabelecer conexão com o Ollama no endereço `OLLAMA_URL` em até `OLLAMA_TIMEOUT_SECONDS` contados do início da tentativa, THEN THE Servico_Local_de_Geracao SHALL retornar uma Resposta_de_Erro com o código `ollama_unavailable` e orientação para iniciar o Ollama, sem retornar resposta à pergunta.
3. IF o Modelo_de_Geracao_Configurado não estiver instalado no Ollama, THEN THE Servico_Local_de_Geracao SHALL retornar uma Resposta_de_Erro com o código `ollama_model_missing` e orientação para executar `ollama pull <modelo-configurado>`, substituindo `<modelo-configurado>` pelo nome do Modelo_de_Geracao_Configurado, sem retornar resposta à pergunta.
4. WHEN o Servico_RAG montar o prompt, THE Servico_RAG SHALL declarar em português do Brasil que o modelo atua como assistente de suporte ao ERP.
5. WHEN o Servico_RAG montar o prompt, THE Servico_RAG SHALL exigir que todas as afirmações factuais da resposta sejam sustentadas exclusivamente pelo Contexto_Recuperado.
6. WHEN o Servico_RAG montar o prompt, THE Servico_RAG SHALL exigir exclusivamente a Resposta_de_Insuficiencia sempre que não for possível produzir uma Resposta_Fundamentada para a pergunta a partir do Contexto_Recuperado.
7. WHEN o Servico_RAG montar o prompt, THE Servico_RAG SHALL exigir redação em português do Brasil, em frases completas, sem saudação, preâmbulo ou repetição de uma mesma afirmação factual.
8. WHEN o Servico_RAG montar o prompt para uma pergunta que solicite um procedimento cujos passos estejam presentes no Contexto_Recuperado, THE Servico_RAG SHALL exigir que cada passo seja apresentado em um item separado de uma lista numerada sequencialmente a partir de 1.
9. IF o Contexto_Recuperado contiver nomes de menus, campos ou telas, THEN THE Servico_RAG SHALL exigir no prompt que esses nomes sejam reproduzidos literalmente na resposta, sem tradução ou alteração.
10. WHEN o Servico_RAG montar o prompt, THE Servico_RAG SHALL proibir o preenchimento de lacunas por inferência e a apresentação como fato de qualquer informação não expressa no Contexto_Recuperado.
11. WHEN o Servico_RAG inserir a pergunta e o Contexto_Recuperado no prompt, THE Servico_RAG SHALL delimitá-los em blocos separados, identificados como dados não confiáveis e sem autoridade para alterar as regras do sistema.
12. WHEN uma Instrucao_Nao_Confiavel aparecer na pergunta ou no Contexto_Recuperado, THE Servico_Local_de_Geracao SHALL tratá-la somente como conteúdo informacional, sem executar comando nem alterar as regras de fundamentação.
13. WHEN o Servico_Local_de_Geracao solicitar uma resposta, THE Servico_Local_de_Geracao SHALL limitar a geração a no máximo `MAX_ANSWER_TOKENS`.
14. WHEN o Modelo_de_Geracao_Configurado devolver uma Resposta_Fundamentada ou a Resposta_de_Insuficiencia, THE Servico_RAG SHALL retornar o conteúdo como texto inerte, sem interpretar nem executar marcação, código ou comando contido na resposta.
15. IF, após a conexão com o Ollama ser estabelecida, a geração for interrompida, devolver conteúdo vazio ou somente espaços em branco, ou não concluir em até `OLLAMA_TIMEOUT_SECONDS` contados do início da geração, THEN THE Servico_Local_de_Geracao SHALL descartar qualquer conteúdo parcial e retornar uma Resposta_de_Erro com o código `generation_failed`, sem substituir a falha por conhecimento geral.
16. THE Servico_Local_de_Geracao SHALL manter o nome do modelo exclusivamente na configuração centralizada.
17. IF o Modelo_de_Geracao_Configurado devolver conteúdo não vazio que não seja uma Resposta_Fundamentada nem corresponda exatamente à Resposta_de_Insuficiencia, THEN THE Servico_RAG SHALL descartar o conteúdo gerado e retornar a Resposta_de_Insuficiencia.

### Requisito 14: Proveniência e composição de fontes

**História do Usuário:** Como Operador_de_Suporte, quero ver o PDF e a página reais de cada resposta, para que eu possa conferir a orientação na documentação original.

#### Critérios de Aceitação

1. WHEN o Servico_RAG compuser uma Resposta_Fundamentada, THE Servico_RAG SHALL construir diretamente dos metadados dos Chunks do Contexto_Recuperado efetivamente enviados ao Modelo_de_Geracao_Configurado uma Fonte para cada par distinto de Nome_Exibivel_do_Documento e Numero_Humano_da_Pagina.
2. WHEN o Servico_RAG retornar uma Resposta_Fundamentada, THE Servico_RAG SHALL retornar as Fontes como uma lista de 1 a `TOP_K` itens no campo JSON `sources`, separado do campo `answer`.
3. THE Servico_RAG SHALL impedir que texto produzido pelo Modelo_de_Geracao_Configurado crie ou altere Fontes.
4. IF dois ou mais Chunks do Contexto_Recuperado efetivamente enviados ao Modelo_de_Geracao_Configurado tiverem o mesmo Nome_Exibivel_do_Documento e Numero_Humano_da_Pagina, THEN THE Servico_RAG SHALL retornar exatamente uma Fonte para o par. *(Propriedade de correção: deduplicação.)*
5. WHEN o Servico_RAG retornar Fontes, THE Servico_RAG SHALL preservar a ordem da primeira ocorrência de cada par de Nome_Exibivel_do_Documento e Numero_Humano_da_Pagina no Contexto_Recuperado.
6. WHEN uma Fonte for retornada, THE Servico_RAG SHALL incluir exatamente os campos `document` e `page`.
7. WHEN uma Fonte for retornada, THE Servico_RAG SHALL definir `page` como o Numero_Humano_da_Pagina do Chunk de origem, representado por um inteiro entre 1 e a quantidade de Paginas do Documento_PDF de origem, inclusive.
8. WHEN o Servico_RAG retornar a Resposta_de_Insuficiencia, THE Servico_RAG SHALL retornar `sources` como uma lista vazia.
9. WHEN o Servico_RAG retornar Fontes, THE Servico_RAG SHALL retornar somente Fontes correspondentes a Chunks do Contexto_Recuperado efetivamente enviados ao Modelo_de_Geracao_Configurado. *(Propriedade de correção: proveniência.)*

### Requisito 15: API de perguntas

**História do Usuário:** Como desenvolvedor da Interface_Web, quero um contrato JSON previsível para perguntas, para que estados de sucesso, insuficiência e erro sejam tratados de modo consistente.

#### Critérios de Aceitação

1. WHEN a Aplicacao_Web receber `POST /chat`, THE Aplicacao_Web SHALL exigir que o tipo de mídia de `Content-Type` seja `application/json` e que o corpo seja um objeto JSON raiz contendo ao menos o campo `question`.
2. IF `Content-Type` estiver ausente ou seu tipo de mídia não for `application/json`, THEN THE Aplicacao_Web SHALL responder com status HTTP 400 e uma Resposta_de_Erro que indique a incompatibilidade do tipo de mídia.
3. IF o tipo de mídia for `application/json` e o corpo estiver ausente, contiver JSON malformado ou não representar um objeto JSON raiz, THEN THE Aplicacao_Web SHALL responder com status HTTP 400 e uma Resposta_de_Erro que indique a invalidade do corpo JSON.
4. IF o corpo for um objeto JSON raiz e `question` estiver ausente ou não for uma string, THEN THE Aplicacao_Web SHALL responder com status HTTP 400 e uma Resposta_de_Erro que indique a ausência ou o tipo inválido de `question`.
5. IF `question` for uma string com no máximo `MAX_QUESTION_CHARS` pontos de código Unicode antes da normalização e resultar vazia após a remoção, nas duas extremidades, dos caracteres classificados como espaços em branco pelo padrão Unicode, THEN THE Aplicacao_Web SHALL responder com status HTTP 400 e uma Resposta_de_Erro que indique que a pergunta está vazia.
6. IF `question` for uma string com mais de `MAX_QUESTION_CHARS` pontos de código Unicode antes da normalização, THEN THE Aplicacao_Web SHALL responder com status HTTP 413 e uma Resposta_de_Erro que indique que o limite da pergunta foi excedido.
7. IF uma requisição atender a qualquer condição de rejeição dos critérios 2 a 6, THEN THE Aplicacao_Web SHALL encerrar o processamento sem chamar o Servico_de_Recuperacao ou o Servico_Local_de_Geracao.
8. WHEN a Aplicacao_Web receber `POST /chat` com tipo de mídia `application/json`, corpo representado por um objeto JSON raiz e `question` representada por uma string que não atenda às condições dos critérios 5 e 6, THE Aplicacao_Web SHALL remover os espaços em branco Unicode das extremidades de `question` e encaminhar somente o texto resultante ao Servico_de_Recuperacao.
9. WHEN uma Resposta_Fundamentada for produzida, THE Aplicacao_Web SHALL responder com status HTTP 200 e um objeto JSON com exatamente os campos `answer` e `sources`, em que `answer` contenha a Resposta_Fundamentada e `sources` seja uma lista de uma a `TOP_K` Fontes.
10. WHEN a Resposta_de_Insuficiencia for produzida, THE Aplicacao_Web SHALL responder com status HTTP 200 e um objeto JSON com exatamente os campos `answer` e `sources`, em que `answer` contenha a Resposta_de_Insuficiencia e `sources` seja uma lista vazia.
11. WHEN Fontes forem retornadas, THE Aplicacao_Web SHALL representar cada Fonte como um objeto JSON com exatamente os campos `document` e `page`, em que `document` seja o Nome_Exibivel_do_Documento e `page` seja o Numero_Humano_da_Pagina.
12. IF o Servico_Local_de_Embeddings ou o ChromaDB estiver indisponível durante o processamento da pergunta, THEN THE Aplicacao_Web SHALL responder com status HTTP 503 e uma Resposta_de_Erro cuja mensagem indique qual dependência está indisponível.
13. IF o Ollama não aceitar conexão dentro de `OLLAMA_TIMEOUT_SECONDS`, THEN THE Aplicacao_Web SHALL responder com status HTTP 503 e a Resposta_de_Erro correspondente à indisponibilidade do Ollama.
14. IF o Ollama estiver disponível e o Modelo_de_Geracao_Configurado estiver ausente, THEN THE Aplicacao_Web SHALL responder com status HTTP 503 e a Resposta_de_Erro correspondente à ausência do modelo.
15. IF ocorrer uma falha interna não coberta por outro critério deste requisito, THEN THE Aplicacao_Web SHALL responder com status HTTP 500 e uma Resposta_de_Erro genérica.
16. WHEN a Aplicacao_Web retornar um erro de chat, THE Aplicacao_Web SHALL omitir stack trace, prompt, Contexto_Recuperado e detalhes internos da resposta HTTP.
17. THE Aplicacao_Web SHALL processar cada pergunta de forma independente, sem usar perguntas, respostas ou Fontes de requisições `POST /chat` anteriores e sem persistir esses dados como histórico de conversa.

### Requisito 16: Resultado e erros da API de upload

**História do Usuário:** Como Administrador_da_Base, quero contagens e avisos objetivos após a importação, para que eu saiba quanto conteúdo entrou na base e quais arquivos exigem atenção.

#### Critérios de Aceitação

1. WHEN um Upload terminar com a Transacao_de_Ingestao concluída com sucesso, THE Aplicacao_Web SHALL responder com status HTTP 200 e um objeto JSON contendo exatamente os campos `{success, documents, pages, chunks, warnings}`.
2. WHEN um Upload terminar com a Transacao_de_Ingestao concluída com sucesso, THE Aplicacao_Web SHALL definir `success` como o booleano `true`.
3. WHEN um Upload terminar com a Transacao_de_Ingestao concluída com sucesso, THE Aplicacao_Web SHALL definir `documents` como um inteiro não negativo igual à quantidade de ocorrências de Documento_PDF abertas e processadas com sucesso pelo Extrator_de_PDF no Upload, excluídas todas as ocorrências puladas por duplicidade.
4. WHEN um Upload terminar com a Transacao_de_Ingestao concluída com sucesso, THE Aplicacao_Web SHALL definir `pages` como um inteiro não negativo igual à soma das Paginas enumeradas nos Documentos_PDF contabilizados em `documents`, incluindo cada Pagina_Sem_Texto.
5. WHEN um Upload terminar com a Transacao_de_Ingestao concluída com sucesso, THE Aplicacao_Web SHALL definir `chunks` como um inteiro não negativo igual à quantidade de novos Chunks confirmados pela Transacao_de_Ingestao.
6. WHEN uma Transacao_de_Ingestao concluir com sucesso e um ou mais Avisos tiverem sido produzidos para o Upload, THE Aplicacao_Web SHALL definir `warnings` como uma lista de strings contendo, em qualquer ordem, cada Aviso produzido para o Upload exatamente uma vez.
7. WHEN uma Transacao_de_Ingestao concluir com sucesso e nenhum Aviso tiver sido produzido para o Upload, THE Aplicacao_Web SHALL definir `warnings` como uma lista vazia.
8. WHEN o Extrator_de_PDF concluir o processamento de um Documento_PDF com pelo menos uma Pagina e todas as suas Paginas forem Pagina_Sem_Texto, THE Aplicacao_Web SHALL contabilizar esse documento em `documents`, todas as suas Paginas em `pages`, nenhum Chunk desse documento em `chunks` e os Avisos de provável necessidade de OCR dessas Paginas em `warnings`.
9. IF o Validador_de_Arquivo rejeitar o Arquivo_ZIP, nenhum arquivo `.pdf` for descoberto ou todos os arquivos `.pdf` descobertos forem ignorados por estarem corrompidos, criptografados sem acesso ou ilegíveis, THEN THE Aplicacao_Web SHALL responder com status HTTP 422 e `success` igual ao booleano `false`.
10. IF o tamanho compactado do Arquivo_ZIP exceder o limite configurado por `MAX_UPLOAD_MB`, THEN THE Aplicacao_Web SHALL responder com status HTTP 413 e `success` igual ao booleano `false`.
11. IF uma Transacao_de_Ingestao concorrente impedir o Upload, THEN THE Aplicacao_Web SHALL responder com status HTTP 409 e `success` igual a `false`.
12. IF a geração de um Embedding ou uma operação do ChromaDB impedir a confirmação da Transacao_de_Ingestao, THEN THE Aplicacao_Web SHALL responder com status HTTP 503 e `success` igual ao booleano `false`.
13. WHEN a Aplicacao_Web retornar um erro de Upload, THE Aplicacao_Web SHALL responder com uma Resposta_de_Erro que contenha `success` igual ao booleano `false`, código e mensagem não vazios, o mesmo código em todas as ocorrências da mesma condição de erro e nenhum stack trace, segredo, caminho interno do sistema operacional ou detalhe interno.
14. IF a Aplicacao_Web responder a um Upload com qualquer erro, THEN THE Armazenamento_Vetorial_Local SHALL confirmar zero Chunks provenientes desse Upload e preservar todos os Chunks confirmados antes do início do Upload.

### Requisito 17: Comportamento da interface durante requisições

**História do Usuário:** Como usuário da aplicação, quero feedback visual e controles seguros durante operações, para que eu entenda o estado atual e evite envios duplicados.

#### Critérios de Aceitação

1. WHEN o Operador_de_Suporte iniciar o envio de uma pergunta válida por `POST /chat`, THE Interface_Web SHALL exibir o estado `Consultando...` na área de Resposta_Fundamentada desde o início da requisição até ela terminar com sucesso ou erro.
2. WHILE uma pergunta estiver em processamento, THE Interface_Web SHALL desabilitar o campo de pergunta e o botão `Perguntar`.
3. WHEN uma requisição `POST /chat` terminar com sucesso ou erro, THE Interface_Web SHALL reabilitar o campo de pergunta e o botão `Perguntar`.
4. WHEN uma resposta HTTP 200 de `POST /chat` for recebida, THE Interface_Web SHALL substituir o conteúdo da área de Resposta_Fundamentada pelo valor de `answer` exibido como texto.
5. WHEN uma resposta HTTP 200 de `POST /chat` contiver uma lista não vazia de Fontes, THE Interface_Web SHALL substituir os itens da área de Fontes por exatamente um item para cada Fonte, na ordem recebida e no formato `document — página N`, em que `document` é o valor do campo `document` e `N` é o valor do campo `page`.
6. WHEN uma resposta HTTP 200 de `POST /chat` contiver uma lista de Fontes vazia, THE Interface_Web SHALL remover os itens de Fonte anteriormente exibidos e manter a área de Fontes com zero itens.
7. WHEN o Administrador_da_Base iniciar um Upload por `POST /upload`, THE Interface_Web SHALL exibir o estado `Processando...` na área `Base de conhecimento` desde o início da requisição até ela terminar com sucesso ou erro.
8. WHILE um Upload estiver em processamento, THE Interface_Web SHALL desabilitar o seletor de arquivo e a ação de importação.
9. WHEN uma resposta HTTP 200 de `POST /upload` com `success` igual a `true` for recebida, THE Interface_Web SHALL exibir o estado `Concluído` e substituir os valores dos contadores pelos valores de `documents`, `pages` e `chunks` recebidos.
10. WHEN uma resposta HTTP 200 de `POST /upload` for recebida, THE Interface_Web SHALL substituir os Avisos anteriormente exibidos por exatamente um item textual para cada elemento de `warnings`, na ordem recebida, ou por zero itens quando `warnings` estiver vazio.
11. IF uma requisição `POST /chat` ou `POST /upload` retornar uma resposta HTTP de erro com uma Resposta_de_Erro, THEN THE Interface_Web SHALL exibir como texto a mensagem pública dessa Resposta_de_Erro na área de Resposta_Fundamentada para `POST /chat` ou na área `Base de conhecimento` para `POST /upload`, mantendo inalterado, respectivamente, o conteúdo do campo de pergunta ou o Arquivo_ZIP selecionado.
12. WHEN uma requisição `POST /upload` terminar com sucesso ou erro, THE Interface_Web SHALL reabilitar o seletor de arquivo e a ação de importação.
13. THE Interface_Web SHALL usar `fetch()` para `POST /chat` e `POST /upload`.
14. THE Interface_Web SHALL limitar o seletor de importação à extensão `.zip`.
15. WHEN a Interface_Web renderizar pergunta, resposta, Fonte, nome de arquivo, Aviso ou erro, THE Interface_Web SHALL inserir o valor como texto não interpretado, sem `innerHTML` com conteúdo não confiável. *(Propriedade de correção: preservação literal e prevenção de XSS.)*
16. WHEN o valor de `answer` contiver quebras de linha, THE Interface_Web SHALL exibir cada quebra de linha como uma quebra de linha visual sem interpretar qualquer trecho como marcação HTML.
17. WHILE uma requisição `POST /chat` estiver em processamento, THE Interface_Web SHALL impedir o início de outra requisição `POST /chat`.
18. WHILE uma requisição `POST /upload` estiver em processamento, THE Interface_Web SHALL impedir o início de outra requisição `POST /upload`.
19. IF uma requisição `POST /chat` ou `POST /upload` terminar sem receber uma resposta HTTP ou receber um corpo incompatível com o contrato aplicável, THEN THE Interface_Web SHALL exibir como texto uma indicação de que a operação não pôde ser concluída na área correspondente e manter inalterado, respectivamente, o conteúdo do campo de pergunta ou o Arquivo_ZIP selecionado.

### Requisito 18: Segurança e privacidade operacional

**História do Usuário:** Como responsável pelos documentos internos, quero processamento local e superfícies restritas, para que o MVP não exponha conteúdo nem execute entradas não confiáveis.

#### Critérios de Aceitação

1. THE Aplicacao_Web SHALL usar `127.0.0.1` como host padrão.
2. THE Aplicacao_Web SHALL iniciar com modo debug desabilitado por padrão.
3. THE ERP_AI_Support SHALL manter credenciais, tokens e segredos fora do código-fonte e do Exemplo_de_Ambiente.
4. WHILE o ERP_AI_Support estiver processando documentos, perguntas, Contexto_Recuperado ou respostas, THE ERP_AI_Support SHALL restringir qualquer transmissão em rede desses conteúdos aos hosts `localhost`, `127.0.0.1` e `::1`.
5. WHEN o Servico_de_Ingestao gravar no Ambiente_Local um arquivo regular proveniente de um Arquivo_ZIP ou Documento_PDF, THE Servico_de_Ingestao SHALL gravá-lo sem permissão de execução para proprietário, grupo ou demais usuários.
6. IF uma Instrucao_Nao_Confiavel estiver presente em uma pergunta, Documento_PDF ou Contexto_Recuperado, THEN THE ERP_AI_Support SHALL impedir que essa instrução cause a inicialização de processo do sistema ou a execução de comando shell.
7. WHEN a Aplicacao_Web registrar uma falha local, THE Aplicacao_Web SHALL manter stack trace somente no log local do servidor.
8. WHEN a Aplicacao_Web compuser uma resposta HTTP para um cliente, THE Aplicacao_Web SHALL omitir caminhos absolutos, valores de credenciais, tokens e segredos, versões e caminhos de instalação de bibliotecas e mensagens internas de exceção.
9. WHEN a Interface_Web exibir um Nome_Exibivel_do_Documento, THE Interface_Web SHALL exibir somente o nome relativo sanitizado fornecido pela API.
10. IF uma operação falhar, THEN THE ERP_AI_Support SHALL omitir da Resposta_de_Erro o Texto_Extraido, os Chunks e o Contexto_Recuperado associados à operação.
11. WHILE a Aplicacao_Web estiver vinculada a `localhost`, `127.0.0.1` ou `::1`, THE ERP_AI_Support SHALL disponibilizar o MVP sem exigir autenticação.

### Requisito 19: Mensagens de falha acionáveis

**História do Usuário:** Como usuário local, quero distinguir as causas comuns de falha, para que eu possa corrigir a instalação ou o conteúdo importado sem analisar stack traces.

#### Critérios de Aceitação

1. IF o Ollama configurado em `OLLAMA_URL` não aceitar uma conexão dentro de `OLLAMA_TIMEOUT_SECONDS`, THEN THE Aplicacao_Web SHALL retornar uma Resposta_de_Erro que identifica o Ollama local como indisponível e orienta iniciar o serviço.
2. IF o Modelo_de_Geracao_Configurado não estiver instalado no Ollama, THEN THE Aplicacao_Web SHALL retornar uma Resposta_de_Erro que identifica o modelo ausente e informa o comando `ollama pull <modelo-configurado>`, com `<modelo-configurado>` substituído pelo nome do Modelo_de_Geracao_Configurado.
3. IF o conteúdo do arquivo enviado não possuir estrutura ZIP válida, THEN THE Aplicacao_Web SHALL rejeitar o Upload sem alterar a Base_de_Conhecimento e retornar uma Resposta_de_Erro que informa que o arquivo não é um ZIP válido e orienta enviar um Arquivo_ZIP válido.
4. IF um arquivo candidato a Documento_PDF não puder ser aberto por estar corrompido, THEN THE Servico_de_Ingestao SHALL ignorar esse arquivo, adicionar um Aviso que identifica o Nome_Exibivel_do_Documento e orienta substituí-lo ou removê-lo, sem interromper o processamento dos demais Documentos_PDF válidos.
5. IF uma Pagina_Sem_Texto for encontrada, THEN THE Servico_de_Ingestao SHALL adicionar um Aviso que identifica o Nome_Exibivel_do_Documento e o Numero_Humano_da_Pagina, informa que nenhum caractere não branco foi extraído e orienta aplicar OCR antes de uma nova importação caso a página contenha texto em imagem.
6. IF o Upload exceder qualquer um dos Limites_de_Upload, THEN THE Aplicacao_Web SHALL rejeitar o Upload sem alterar a Base_de_Conhecimento e retornar uma Resposta_de_Erro que identifica a categoria excedida como bytes compactados, quantidade de entradas, bytes por entrada, bytes descompactados totais ou razão de compressão e orienta reduzir a categoria indicada antes de reenviar o arquivo.
7. IF o ChromaDB estiver indisponível ou não gravável, THEN THE Aplicacao_Web SHALL retornar uma Resposta_de_Erro que identifica a base vetorial local como indisponível e orienta verificar a disponibilidade e a permissão de escrita do armazenamento configurado.
8. IF `question` contiver zero caracteres ou somente espaços em branco, THEN THE Aplicacao_Web SHALL retornar uma Resposta_de_Erro que informa ser obrigatório fornecer pelo menos um caractere não branco antes de enviar a pergunta.
9. IF o Modelo_de_Embedding não estiver disponível localmente, THEN THE Aplicacao_Web SHALL retornar uma Resposta_de_Erro que identifica o Modelo_de_Embedding configurado e orienta disponibilizá-lo no Ambiente_Local antes de tentar novamente.
10. IF a geração pelo Modelo_de_Geracao_Configurado falhar depois de iniciada, THEN THE Aplicacao_Web SHALL retornar uma Resposta_de_Erro que informa que a resposta não pôde ser gerada, orienta tentar novamente após verificar o Ollama local e o modelo configurado e omite qualquer resposta substituta.
11. WHEN uma condição descrita neste requisito for apresentada ao usuário, THE Aplicacao_Web SHALL usar uma mensagem em português do Brasil sem stack trace, segredo, caminho absoluto ou detalhe interno.
12. IF a geração de um Embedding falhar, THEN THE Aplicacao_Web SHALL retornar uma Resposta_de_Erro que identifica o Modelo_de_Embedding configurado, informa que a geração local de embeddings falhou e orienta verificar a disponibilidade local do modelo antes de tentar novamente.

### Requisito 20: Documentação de instalação, uso e manutenção

**História do Usuário:** Como desenvolvedor ou administrador Ubuntu/Linux, quero instruções completas e reproduzíveis, para que eu possa instalar modelos, iniciar a aplicação e validar o primeiro fluxo local.

#### Critérios de Aceitação

1. THE Guia_de_Operacao SHALL declarar Python 3.11 ou versão posterior, Ollama, o Modelo_de_Geracao_Configurado indicado por `OLLAMA_MODEL` e o Modelo_de_Embedding indicado por `EMBEDDING_MODEL` como pré-requisitos.
2. THE Guia_de_Operacao SHALL documentar criação e ativação de ambiente virtual e instalação pelo Manifesto_de_Dependencias.
3. THE Guia_de_Operacao SHALL documentar, para Ubuntu/Linux, a instalação e a inicialização do Ollama, a verificação de disponibilidade no endereço `OLLAMA_URL` e o comando `ollama pull <modelo-configurado>` para o valor de `OLLAMA_MODEL`.
4. THE Guia_de_Operacao SHALL documentar como disponibilizar no Ambiente_Local o Modelo_de_Embedding indicado por `EMBEDDING_MODEL` e verificar, antes da operação offline, que o modelo pode ser carregado sem conexão com a internet.
5. THE Guia_de_Operacao SHALL justificar `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` pela capacidade de gerar Embeddings para textos em português e pela execução no Ambiente_Local sem API externa paga.
6. THE Guia_de_Operacao SHALL documentar a cópia de `.env.example` para `.env` e, para cada variável do Perfil de Configuração do MVP, a finalidade, o valor de exemplo ou padrão e a regra de validação.
7. THE Guia_de_Operacao SHALL documentar a execução por `python app.py` e o endereço padrão `http://127.0.0.1:5000`.
8. THE Guia_de_Operacao SHALL documentar a importação de um Arquivo_ZIP válido com exatamente duas Entradas_ZIP, ambas Documentos_PDF distintos e com texto extraível, seguida do envio de uma pergunta não vazia de até `MAX_QUESTION_CHARS` caracteres cuja resposta esteja presente nos documentos, e identificar como resultado esperado o estado `Concluído`, os contadores da importação, uma Resposta_Fundamentada e entre 1 e `TOP_K` Fontes.
9. THE Guia_de_Operacao SHALL apresentar a estrutura de arquivos e declarar que `app.py` cria a Aplicacao_Web e suas rotas, `ingest.py` coordena o fluxo do Arquivo_ZIP até a persistência dos Chunks e `rag.py` coordena recuperação, montagem do Contexto_Recuperado, geração e composição das Fontes.
10. THE Guia_de_Operacao SHALL declarar que documentos, embeddings, perguntas e respostas são processados somente no Ambiente_Local, não são transmitidos a serviços externos durante a operação e não são persistidos como histórico de conversa.
11. THE Guia_de_Operacao SHALL documentar como limitações do MVP a ausência de OCR, autenticação, histórico de conversa e integrações com o ERP, além da aceitação exclusiva de Arquivo_ZIP contendo Documento_PDF como formato de importação.
12. THE Guia_de_Operacao SHALL incluir, para Ollama desligado, Modelo_de_Geracao_Configurado ausente, Modelo_de_Embedding ausente, Arquivo_ZIP inválido, Pagina_Sem_Texto, Upload que exceda um Limite_de_Upload e ChromaDB indisponível, a forma de identificar a condição, a ação corretiva, o efeito sobre a operação afetada e o estado dos dados já confirmados na Base_de_Conhecimento.
13. THE Guia_de_Operacao SHALL terminar com um procedimento numerado e ordenado que contenha comandos exatos para criar e ativar o ambiente virtual, instalar o Manifesto_de_Dependencias, instalar e iniciar o Ollama, disponibilizar os modelos no Ambiente_Local e executar `python app.py`, seguidos do endereço a abrir no navegador, das ações para importar o primeiro Arquivo_ZIP e enviar a primeira pergunta e dos resultados observáveis definidos no critério 8.
14. THE Manifesto_de_Dependencias SHALL listar Flask, ChromaDB, PyMuPDF, sentence-transformers, python-dotenv e todas as dependências de terceiros importadas diretamente pelo código necessário à execução, cada uma com versão exata, de modo que a instalação em um ambiente virtual limpo com Python 3.11 permita executar `python app.py` sem erro de dependência ausente.
15. THE Guia_de_Operacao SHALL explicar que a conexão com a internet pode ser necessária para obter dependências e modelos na primeira instalação e que, depois de instalados no Ambiente_Local, os fluxos de ingestão, recuperação e geração operam sem conexão com a internet, com a geração restrita ao endereço local configurado em `OLLAMA_URL`.

### Requisito 21: Delimitação do MVP e evolução

**História do Usuário:** Como patrocinador do MVP, quero limites explícitos e separação simples de responsabilidades, para que a primeira entrega permaneça pequena sem bloquear evolução futura.

#### Critérios de Aceitação

1. THE ERP_AI_Support SHALL limitar as capacidades do MVP a ingestão de PDFs por ZIP, indexação local, consulta semântica, geração local fundamentada e exibição de Fontes.
2. THE Aplicacao_Web SHALL operar com uma única Base_de_Conhecimento no Ambiente_Local, sem oferecer cadastro de usuários, controle de permissões ou autenticação.
3. WHEN a Aplicacao_Web receber uma pergunta, THE Aplicacao_Web SHALL processá-la sem usar perguntas ou respostas anteriores e sem persistir a pergunta recebida ou a resposta produzida.
4. THE Aplicacao_Web SHALL operar sem funcionalidade de avaliação de respostas, dashboard ou sessão conversacional.
5. THE Servico_de_Ingestao SHALL aceitar como conteúdo documental somente Documento_PDF contido em Arquivo_ZIP.
6. WHEN uma Pagina_Sem_Texto for detectada, THE Extrator_de_PDF SHALL adicionar um Aviso de provável necessidade de OCR sem executar OCR.
7. THE Servico_de_Ingestao SHALL usar a Identidade_do_Documento somente para identificação técnica, deduplicação e idempotência de Documento_PDF, sem oferecer versionamento ou categorização de documentos ao usuário.
8. THE ERP_AI_Support SHALL atribuir à Aplicacao_Web o recebimento de Upload e perguntas e o retorno dos resultados, ao Servico_de_Ingestao o processamento do Arquivo_ZIP até a persistência dos Chunks e ao Servico_RAG a montagem do Contexto_Recuperado, a solicitação de geração local fundamentada e a composição das Fontes, mantendo os três componentes dentro de uma única aplicação modular.
9. WHILE executando o MVP, THE ERP_AI_Support SHALL operar sem troca de dados ou comandos com aplicações PHP, Java ou COBOL, com o banco de dados ou a API do ERP e com sistemas clientes externos ao ERP_AI_Support.

### Requisito 22: Critério de sucesso ponta a ponta

**História do Usuário:** Como patrocinador, quero um fluxo local demonstrável, para que o MVP prove ingestão, recuperação, fundamentação e recusa segura sem custo de API externa.

#### Critérios de Aceitação

1. WHILE todas as dependências declaradas no Manifesto_de_Dependencias e o Modelo_de_Embedding estiverem instalados no Ambiente_Local e o Modelo_de_Geracao_Configurado estiver instalado no Ollama, WHEN `python app.py` for executado no Ambiente_Local, THE ERP_AI_Support SHALL iniciar a Aplicacao_Web no endereço local configurado.
2. WHEN o navegador abrir o endereço local configurado após a inicialização, THE Aplicacao_Web SHALL apresentar a Interface_Web com o campo de pergunta, o botão `Perguntar`, o seletor de Arquivo_ZIP e a ação de importação habilitados.
3. WHEN um Arquivo_ZIP válido contendo de 2 a `MAX_ZIP_ENTRIES` Documentos_PDF com Identidades_do_Documento distintas e ainda ausentes da Base_de_Conhecimento, cada um com pelo menos uma Pagina que não seja Pagina_Sem_Texto, for importado, THE ERP_AI_Support SHALL confirmar a contagem de documentos como a quantidade de Documentos_PDF abertos e processados, a contagem de páginas como a quantidade total de Paginas examinadas nesses documentos e a contagem de Chunks como a quantidade de novos Chunks confirmados pela Transacao_de_Ingestao.
4. WHEN uma pergunta validada solicitar uma informação declarada literalmente em um Chunk da Base_de_Conhecimento e o Candidato_Recuperado correspondente tiver Pontuacao_de_Relevancia maior ou igual ao Limiar_de_Relevancia, THE ERP_AI_Support SHALL retornar uma Resposta_Fundamentada que responda à informação solicitada, limitada a `MAX_ANSWER_TOKENS`, com uma lista não vazia contendo exatamente as Fontes deduplicadas dos Chunks do Contexto_Recuperado enviados ao Modelo_de_Geracao_Configurado.
5. WHEN uma pergunta validada for enviada e nenhum Candidato_Recuperado tiver Pontuacao_de_Relevancia maior ou igual ao Limiar_de_Relevancia, THE ERP_AI_Support SHALL retornar a Resposta_de_Insuficiencia com uma lista de Fontes vazia.
6. WHILE o Ambiente_Local estiver sem acesso à internet, o Modelo_de_Embedding e o Modelo_de_Geracao_Configurado estiverem instalados, o Armazenamento_Vetorial_Local estiver disponível e o Ollama estiver aceitando conexões em `OLLAMA_URL`, WHEN uma pergunta validada for enviada, THE ERP_AI_Support SHALL retornar status HTTP 200 com uma Resposta_Fundamentada ou a Resposta_de_Insuficiencia sem realizar requisição a API externa ou paga.

## Propriedades de Correção Consolidadas

As propriedades abaixo são critérios testáveis derivados dos requisitos e orientam testes baseados em propriedades quando o comportamento varia significativamente com a entrada:

1. **Confinamento de caminho — Requisito 6.3:** para toda Entrada_ZIP aceita, o caminho de destino resolvido permanece descendente do diretório temporário autorizado.
2. **Identidade estável — Requisito 8.2:** PDFs com bytes iguais produzem a mesma Identidade_do_Documento independentemente do nome, subpasta ou Upload.
3. **Idempotência de ingestão — Requisitos 8.5 e 11.11:** repetir a ingestão do mesmo PDF sob configuração compatível não altera a quantidade nem o conteúdo dos Chunks ativos.
4. **Determinismo de chunking — Requisitos 8.9 e 9.10:** a mesma página e configuração produzem os mesmos identificadores, textos e posições.
5. **Invariantes de chunking — Requisitos 9.4, 9.6 e 9.7:** Chunks são substrings contíguas, respeitam a sobreposição e cobrem todo o texto original não vazio.
6. **Consistência vetorial — Requisitos 10.3 e 10.10:** documentos e perguntas compartilham o mesmo Espaco_Vetorial e dimensão.
7. **Round trip de persistência — Requisito 11.4:** Chunks confirmados antes de uma reinicialização permanecem recuperáveis com metadados equivalentes depois da reinicialização.
8. **Atomicidade — Requisitos 11.6 e 11.7:** uma falha fatal não deixa alterações parciais do Upload e não modifica conhecimento previamente confirmado.
9. **Confinamento da recuperação — Requisito 12.6:** todo Chunk enviado à geração pertence ao conjunto recuperado e satisfaz o Limiar_de_Relevancia.
10. **Proveniência de fontes — Requisitos 14.4 e 14.9:** cada Fonte é única por documento/página e corresponde a um Chunk efetivamente fornecido ao modelo.
11. **Renderização literal — Requisito 17.15:** conteúdo não confiável exibido pela interface permanece texto e não cria nós HTML executáveis.

Testes baseados em propriedades são apropriados para validação de caminhos ZIP, deduplicação, chunking, identidades, filtragem de recuperação e composição de Fontes, pois esses comportamentos variam com muitas combinações de entrada e podem ser exercitados em memória. Integrações reais com Ollama, ChromaDB, sistema de arquivos e navegador devem usar testes unitários com mocks e testes de integração com poucos exemplos representativos, evitando centenas de operações externas ou de alto custo.

## Fora do Escopo do MVP

Os itens a seguir não fazem parte desta funcionalidade: autenticação; cadastro de usuários; permissões; histórico de conversas; avaliações; dashboard; OCR; ingestão direta de DOCX ou TXT; versionamento ou categorização de documentos; integração com ERP, banco de dados, APIs, PHP, Java ou COBOL; execução de código; acesso externo para clientes; microsserviços; Kubernetes; backend Node.js; React; APIs externas ou pagas. A separação entre aplicação, ingestão e RAG deve permitir evolução futura sem antecipar essas capacidades no MVP.
