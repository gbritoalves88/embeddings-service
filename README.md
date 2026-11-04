# Embeddings Service

Microserviço REST para geração de **embeddings de texto**, desenvolvido em Python com FastAPI e Sentence Transformers.

O serviço transforma textos em vetores numéricos que representam características semânticas do conteúdo. Esses vetores podem ser utilizados por outras aplicações em soluções de busca semântica, recuperação de documentos, recomendação e processamento de linguagem natural.

## 🎯 Objetivo

O Embeddings Service foi desenvolvido para disponibilizar a geração de embeddings por meio de uma API independente.

A ideia é desacoplar o processamento de linguagem natural da aplicação principal, permitindo que diferentes sistemas consumam o mesmo serviço sem precisar carregar ou executar diretamente o modelo de Machine Learning.

O projeto foi pensado para processamento de textos relacionados à jurisprudência, podendo ser integrado a sistemas de pesquisa e recuperação de informações jurídicas.

## 🧠 Como funciona

O serviço utiliza um modelo pré-treinado da biblioteca Sentence Transformers para converter textos em representações vetoriais.

Fluxo de processamento:

```text
        Aplicação cliente
               |
               v
        Requisição HTTP
               |
               v
        FastAPI
               |
               v
        Validação do texto
               |
               v
       Sentence Transformers
               |
               v
       Modelo de Embeddings
               |
               v
        Vetor numérico
               |
               v
        Resposta JSON
               |
               v
        Aplicação cliente
```

Cada texto enviado é processado pelo modelo, que retorna um vetor de valores numéricos.

Esses vetores podem posteriormente ser armazenados em um banco vetorial e utilizados em operações de busca por similaridade semântica.

## ✨ Funcionalidades

- Geração de embedding para um único texto.
- Geração de embeddings para múltiplos textos em uma única requisição (*batch processing*).
- Validação dos textos recebidos.
- Validação do tamanho máximo do lote.
- Modelo de embeddings configurável.
- Dimensão vetorial configurável.
- Carregamento de modelo local ou diretamente do Hugging Face.
- Endpoint de health check que verifica o funcionamento do modelo.
- Documentação interativa da API via Swagger.
- Logging de inicialização, processamento e erros.

## 🧰 Tecnologias

| Tecnologia | Utilização |
|---|---|
| Python | Linguagem principal |
| FastAPI | Construção da API REST |
| Uvicorn | Servidor ASGI |
| Sentence Transformers | Geração de embeddings |
| Hugging Face | Disponibilização de modelos pré-treinados |
| PyTorch | Execução dos modelos de Machine Learning |
| Pydantic | Validação e estruturação dos dados |
| Pydantic Settings | Gerenciamento de configurações |
| python-dotenv | Carregamento de variáveis de ambiente |

## 🤖 Modelo de embeddings

O modelo utilizado por padrão é:

`sentence-transformers/all-mpnet-base-v2`

Características da configuração:

| Propriedade | Valor |
|---|---|
| Modelo | `sentence-transformers/all-mpnet-base-v2` |
| Dimensão esperada | 768 |
| Biblioteca | Sentence Transformers |
| Origem | Hugging Face |

O modelo é carregado na inicialização do serviço e mantido em memória para reutilização durante as requisições.

O carregamento pode ser feito a partir de um diretório local ou pelo identificador do modelo no Hugging Face.

A dimensão retornada é verificada durante a inicialização. Caso seja diferente da dimensão configurada, o serviço registra um aviso no log.

**Observação:** a escolha do modelo deve considerar o idioma e o domínio dos textos. O `all-mpnet-base-v2` é voltado principalmente ao inglês. Para aplicações jurídicas em português, é recomendável avaliar modelos multilíngues ou treinados especificamente para esse contexto.

## 📡 Endpoints

A API disponibiliza três endpoints principais.

### `GET /`

Retorna informações básicas sobre o serviço.

Exemplo de resposta:

```json
{
  "service": "Embeddings Service",
  "version": "1.0.0",
  "status": "running",
  "model": "sentence-transformers/all-mpnet-base-v2",
  "dims": 768
}
```

### `GET /health`

Verifica se o modelo está carregado e consegue gerar um embedding.

Exemplo de resposta:

```json
{
  "status": "ok",
  "model": "sentence-transformers/all-mpnet-base-v2",
  "dims": 768
}
```

Caso ocorra uma falha durante a geração do vetor de teste, o endpoint retorna HTTP 500.

### `POST /embed`

Gera o embedding de um único texto.

**Requisição:**

```json
{
  "text": "O princípio da dignidade da pessoa humana."
}
```

**Resposta:**

```json
{
  "vector": [0.0123, -0.0456, 0.0789],
  "dims": 768,
  "model": "sentence-transformers/all-mpnet-base-v2"
}
```

O campo `vector` contém os valores numéricos do embedding. O exemplo apresenta apenas alguns valores ilustrativos; a resposta real contém 768 dimensões na configuração padrão.

### `POST /embed/batch`

Permite gerar embeddings para vários textos em uma única requisição.

**Requisição:**

```json
{
  "texts": [
    "Princípio da dignidade da pessoa humana.",
    "Direito constitucional brasileiro.",
    "Responsabilidade civil e indenização."
  ]
}
```

**Resposta:**

```json
{
  "vectors": [
    [0.0123, -0.0456, 0.0789],
    [-0.0234, 0.0567, 0.0891],
    [0.0345, -0.0678, 0.0123]
  ],
  "dims": 768,
  "model": "sentence-transformers/all-mpnet-base-v2"
}
```

Cada texto gera um vetor independente, mantendo a ordem dos elementos enviados.

O tamanho máximo configurado para o lote é de 32 textos por requisição.

## ⚙️ Configurações

As configurações são centralizadas em `app/config.py`, utilizando Pydantic Settings e variáveis de ambiente.

| Variável | Padrão | Descrição |
|---|---|---|
| `EMBEDDINGS_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Modelo utilizado |
| `EXPECTED_DIMS` | `768` | Dimensão esperada dos vetores |
| `HOST` | `0.0.0.0` | Endereço de inicialização |
| `PORT` | `8000` | Porta da aplicação |
| `TIMEOUT_SECONDS` | `30` | Configuração de timeout |
| `MAX_BATCH_SIZE` | `32` | Limite de textos por lote |
| `PREFER_BATCH` | `true` | Preferência de processamento em lote |

As configurações podem ser definidas em um arquivo `.env` na raiz do projeto ou por variáveis de ambiente.

Exemplo:

```env
EMBEDDINGS_MODEL=sentence-transformers/all-mpnet-base-v2
EXPECTED_DIMS=768

HOST=0.0.0.0
PORT=8000

TIMEOUT_SECONDS=30

MAX_BATCH_SIZE=32
PREFER_BATCH=true
```

Ao alterar o modelo, é importante verificar sua dimensão de saída e ajustar `EXPECTED_DIMS` de acordo com o modelo escolhido.

## 📁 Estrutura do projeto

```text
embeddings-service/
│
└── app/
    ├── __init__.py
    ├── config.py
    ├── main.py
    ├── models.py
    ├── router.py
    └── schemas.py
```

### Responsabilidade dos módulos

**`main.py`**
- Inicialização da aplicação FastAPI.
- Configuração de CORS.
- Registro das rotas.
- Logging de inicialização e encerramento.
- Endpoint raiz.

**`config.py`**
- Definição das configurações.
- Carregamento de variáveis de ambiente.
- Parâmetros do modelo e do servidor.

**`models.py`**
- Carregamento do modelo Sentence Transformers.
- Geração de embedding individual.
- Geração de embeddings em lote.
- Gerenciamento da instância do modelo em memória.

**`schemas.py`**
- Modelos de requisição e resposta.
- Validação dos textos.
- Estruturação dos vetores retornados.
- Validação do tamanho máximo do lote.

**`router.py`**
- Definição dos endpoints.
- Recebimento das requisições.
- Integração com o modelo.
- Tratamento de erros.
- Health check.

## 🚀 Executando localmente

### Pré-requisitos

- Python 3.10 ou superior.
- pip.
- Git.
- Acesso à internet para baixar o modelo do Hugging Face na primeira execução.

### 1. Clonar o repositório

```bash
git clone https://github.com/gbritoalves88/embeddings-service.git

cd embeddings-service
```

### 2. Criar ambiente virtual

Windows:

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv

source venv/bin/activate
```

### 3. Instalar dependências

O repositório contém o código da aplicação no diretório `app`. Como não há um arquivo de dependências versionado, instale as bibliotecas utilizadas pelo código:

```bash
pip install fastapi uvicorn sentence-transformers pydantic pydantic-settings python-dotenv
```

### 4. Iniciar o serviço

Na raiz do repositório:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Na primeira inicialização, o Sentence Transformers poderá baixar o modelo do Hugging Face. O carregamento acontece durante a inicialização da aplicação.

Após iniciar, o serviço estará disponível em:

```text
http://localhost:8000
```

## 📖 Documentação interativa

O FastAPI disponibiliza documentação automática dos endpoints.

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

Essas interfaces permitem visualizar os contratos da API e executar requisições de teste.

## 🔗 Integração com outras aplicações

O serviço foi estruturado para funcionar como um componente independente dentro de uma arquitetura maior.

Uma aplicação cliente pode enviar textos para geração de embeddings e utilizar os vetores retornados em outras etapas de processamento.

Exemplo de arquitetura de busca semântica:

```text
       Aplicação principal
               |
               v
       Embeddings Service
               |
               v
        Vetores de texto
               |
               v
        Banco vetorial
               |
               v
       Busca por similaridade
               |
               v
        Resultados relevantes
```

O serviço é responsável pela geração dos vetores. O armazenamento, indexação, comparação de similaridade e recuperação dos documentos são responsabilidades de outros componentes da solução.

## 📌 Considerações técnicas

- O modelo é carregado em memória e compartilhado pela instância da aplicação.
- O processamento em lote reduz a necessidade de realizar chamadas individuais para cada texto.
- A dimensão dos vetores depende do modelo configurado.
- A troca de modelo exige atenção à compatibilidade dos vetores já armazenados em sistemas consumidores.
- O primeiro carregamento pode ser mais demorado devido ao download e à inicialização do modelo.
- O serviço não mantém banco de dados próprio.
- Os vetores são retornados diretamente na resposta HTTP e não são persistidos pelo serviço.

## 🔮 Possibilidades de evolução

- Suporte a modelos multilíngues especializados em português.
- Autenticação entre serviços.
- Limites configuráveis de tamanho dos textos.
- Processamento assíncrono para cargas maiores.
- Métricas de latência e volume de requisições.
- Cache de embeddings para textos repetidos.
- Suporte a diferentes modelos por configuração.
- Empacotamento com Docker.
- Testes automatizados de integração e desempenho.

---

Desenvolvido por [Gabriel Brito](https://github.com/gbritoalves88).
