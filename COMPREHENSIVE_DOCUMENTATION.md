# Sales Transcript Analysis System - Comprehensive Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Design Patterns](#architecture--design-patterns)
3. [Component Specifications](#component-specifications)
4. [API Documentation](#api-documentation)
5. [Data Flow & Workflows](#data-flow--workflows)
6. [Deployment Guide](#deployment-guide)
7. [Business Documentation](#business-documentation)

---

## 1. System Overview

### 1.1 Purpose
The Sales Transcript Analysis System is an AI-powered application that analyzes sales conversations to extract actionable insights including requirements, recommendations, summaries, and action items. It supports multiple input formats (text, audio, documents) and provides an interactive chat interface for querying stored transcripts.

### 1.2 Key Features
- **Multi-format Input Support**: Text, PDF, Word (DOCX), CSV, Excel (XLSX), Audio (MP3, WAV, M4A, OGG)
- **AI-Powered Analysis**: Uses Azure OpenAI GPT-4o for intelligent transcript analysis
- **Vector Database Storage**: Milvus vector database for semantic search and retrieval
- **Interactive Chat Agent**: RASA chat agent using LangChain ReAct framework
- **RESTful API**: FastAPI-based REST API with automatic documentation
- **Web Interface**: Single-page 3-panel UI for transcript analysis and chat

### 1.3 Technology Stack
- **Backend Framework**: FastAPI 0.104.1
- **AI/ML**: Azure OpenAI (GPT-4o, Whisper, text-embedding-3-small)
- **LLM Orchestration**: LiteLLM 1.17.9, LangChain 0.3.0
- **Vector Database**: Milvus (optional)
- **Document Processing**: PyPDF2, python-docx, pandas, openpyxl
- **Configuration**: YAML, python-dotenv
- **Logging**: colorlog
- **Python Version**: 3.12+

---

## 2. Architecture & Design Patterns

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Interface (HTML/JS)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Transcript  │  │   Analysis   │  │  RASA Chat   │      │
│  │    Input     │  │   Results    │  │   Interface  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  API Endpoints (/analyze/*, /chat, /search, /health) │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
│  Transcript  │  │  Audio Processor │  │  Document    │
│  Analyzer    │  │  (Whisper API)   │  │  Processor   │
└──────────────┘  └──────────────────┘  └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                  ┌──────────────────┐
                  │  Azure OpenAI    │
                  │  - GPT-4o        │
                  │  - Whisper       │
                  │  - Embeddings    │
                  └──────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  Milvus Vector   │
                  │  Database        │
                  └──────────────────┘
```

### 2.2 Design Patterns

#### 2.2.1 Singleton Pattern
- **ConfigLoader**: Global configuration instance (`get_config()`)
- **Component Initialization**: Single instances of analyzers and agents

#### 2.2.2 Strategy Pattern
- **Text Chunking**: Multiple chunking strategies (recursive, character, token-based)
- **Document Processing**: Different processors for each file format

#### 2.2.3 Factory Pattern
- **Document Processor**: Routes to appropriate processor based on file extension

#### 2.2.4 Agent Pattern (LangChain ReAct)
- **Chat Agent**: Uses tools and reasoning to answer queries
- **Sales Helper Agent**: Multi-step reasoning for recommendations

---

## 3. Component Specifications

### 3.1 Core Components

#### 3.1.1 TranscriptAnalyzer
**Location**: `src/agent/transcript_analyzer.py`

**Purpose**: Analyzes sales transcripts using Azure OpenAI GPT-4o

**Key Methods**:
- `analyze_transcript(transcript: str) -> Dict[str, Any]`
  - Input: Raw transcript text
  - Output: Structured analysis with requirements, recommendations, summary, action items
  - Uses LiteLLM for Azure OpenAI integration
  - Supports text chunking for long transcripts (>5000 chars)

**Configuration**:
- Model: GPT-4o (configurable via `azure_openai.deployment_name`)
- Temperature: 0.7
- Max Tokens: 2000

**Prompts Used**:
- `system_prompt`: Defines AI role as sales conversation analyst
- `analysis_prompt`: Structured JSON output format

#### 3.1.2 AudioProcessor
**Location**: `src/agent/audio_processor.py`

**Purpose**: Transcribes audio files using Azure OpenAI Whisper

**Key Methods**:
- `transcribe_audio(audio_file_path: str) -> Optional[str]`
  - Validates audio file (format, size)
  - Calls Azure OpenAI Whisper API
  - Returns transcribed text

**Supported Formats**: MP3, WAV, M4A, OGG
**Max File Size**: 25 MB (configurable)

**Validation**:
- File existence check
- Format validation
- Size limit enforcement

#### 3.1.3 DocumentProcessor
**Location**: `src/utils/document_processor.py`

**Purpose**: Extract text from various document formats

**Supported Formats**:
- **PDF**: PyPDF2 (page-by-page extraction)
- **Word (DOCX)**: python-docx (paragraphs + tables)
- **CSV**: pandas (tabular data to text)
- **Excel (XLSX)**: pandas (multi-sheet support)
- **TXT**: Direct UTF-8 decoding

**Key Method**:
- `process_file(filename: str, file_content: bytes) -> str`

#### 3.1.4 MilvusVectorStore
**Location**: `src/agent/vector_store.py`

**Purpose**: Store and retrieve transcripts using vector embeddings

**Key Features**:
- Semantic search using embeddings
- Automatic collection creation
- Text chunking support for large documents

**Key Methods**:
- `store_transcript(transcript_id, transcript_text, analysis_result, source_type) -> bool`
- `search_similar_transcripts(query_text, top_k=5) -> List[Dict]`
- `get_transcript_by_id(transcript_id) -> Optional[Dict]`

**Schema**:
```python
{
    "transcript_id": str,      # Unique identifier
    "embedding": List[float],  # 1536-dim vector
    "transcript_text": str,    # Full transcript
    "analysis_result": str,    # JSON-encoded analysis
    "source_type": str,        # TEXT/AUDIO/FILE
    "timestamp": int           # Unix timestamp
}
```

**Embedding Model**: text-embedding-3-small (1536 dimensions)

#### 3.1.5 ChatAgent (RASA)
**Location**: `src/agent/chat_agent.py`

**Purpose**: Agentic AI chat interface using LangChain ReAct framework

**Architecture**:
- **LLM**: Azure OpenAI GPT-4o via LiteLLM
- **Framework**: LangChain ReAct Agent
- **Memory**: ConversationBufferMemory
- **Tools**: Vector database search

**Key Features**:
- Autonomous tool selection
- Reasoning and action loop
- Conversation history tracking
- Crisp, concise responses (1-2 sentences)

**Tools Available**:
1. `search_database`: Search vector store for relevant transcripts

**Key Methods**:
- `chat(user_message: str, session_id: Optional[str]) -> Dict[str, Any]`
- `clear_memory()`: Reset conversation history
- `get_chat_history() -> List[Dict]`

#### 3.1.6 SalesHelperAgent
**Location**: `src/agent/sales_helper_agent.py`

**Purpose**: Assist salespeople with requirement extraction and recommendations

**Workflow**:
1. Extract requirements from salesperson input
2. Search database for similar cases
3. Generate recommendations based on requirements + past cases
4. Update conversation history

**Key Methods**:
- `help_salesperson(user_input: str) -> Dict[str, Any]`
- `_extract_requirements(user_input: str) -> List[Dict]`
- `_search_similar_cases(requirements: List[Dict]) -> List[Dict]`
- `_generate_recommendations(user_input, requirements, search_results) -> List[Dict]`

#### 3.1.7 TextChunker
**Location**: `src/utils/text_chunker.py`

**Purpose**: Split large texts into manageable chunks using LangChain

**Chunking Strategies**:
1. **Recursive Character Splitter** (Recommended)
   - Chunk Size: 2000 chars
   - Overlap: 200 chars
   - Separators: `\n\n`, `\n`, `. `, ` `, ``
   - Best for semantic coherence

2. **Character Splitter**
   - Simple splitting by character count
   - Separator: `\n`

3. **Token Splitter**
   - Chunk Size: 1500 tokens
   - Overlap: 150 tokens
   - Ensures chunks fit within model limits

**Key Methods**:
- `chunk_text_recursive(text: str) -> List[str]`
- `chunk_text_by_character(text: str) -> List[str]`
- `chunk_text_by_tokens(text: str) -> List[str]`
- `chunk_documents(text: str, metadata: Dict) -> List[Dict]`
- `get_chunk_stats(chunks: List[str]) -> Dict`

#### 3.1.8 ConfigLoader
**Location**: `src/utils/config_loader.py`

**Purpose**: Load and manage configuration from YAML and environment variables

**Configuration Sources** (Priority Order):
1. Environment variables (.env file)
2. YAML configuration files (config.yaml, prompts.yaml)

**Key Methods**:
- `get(key: str, default: Any) -> Any`: Get config value (supports dot notation)
- `get_prompt(prompt_name: str) -> str`: Get prompt template
- `get_all() -> Dict`: Get complete configuration

**Configuration Files**:
- `config/config.yaml`: Main configuration
- `config/prompts.yaml`: AI prompt templates
- `config/.env`: Sensitive credentials (not in version control)

---

## 4. API Documentation

### 4.1 API Overview

**Base URL**: `http://localhost:8000`
**API Documentation**: `http://localhost:8000/docs` (Swagger UI)
**Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)

### 4.2 Endpoints

#### 4.2.1 Health Check
```
GET /health
```

**Description**: Check API health status

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "milvus_enabled": true
}
```

**Status Codes**:
- `200 OK`: Service is healthy

---

#### 4.2.2 Analyze Text Transcript
```
POST /analyze/text
```

**Description**: Analyze a text transcript

**Request Body**:
```json
{
  "transcript": "Sales Rep: Hi! I'm calling about our CRM solution...",
  "transcript_id": "optional-custom-id",
  "store_in_db": true
}
```

**Parameters**:
- `transcript` (required, string): The conversation transcript
- `transcript_id` (optional, string): Custom identifier (auto-generated if not provided)
- `store_in_db` (optional, boolean): Whether to store in vector database (default: true)

**Response** (200 OK):
```json
{
  "success": true,
  "transcript_id": "uuid-string",
  "transcript": "original transcript text",
  "analysis": {
    "requirements": [
      {
        "requirement": "CRM system for 100 users",
        "priority": "High",
        "mentioned_by": "Client",
        "context": "Client mentioned need for 100 user licenses"
      }
    ],
    "recommendations": [
      {
        "recommendation": "Enterprise Plan",
        "rationale": "Supports 100+ users with mobile access",
        "product_fit": "Meets all stated requirements",
        "priority": "High"
      }
    ],
    "summary": {
      "overview": "Client interested in CRM for 100 users",
      "client_needs": "Mobile access, 100 users, $5000 budget",
      "pain_points": "Current system lacks mobile support",
      "opportunities": "Upsell mobile features",
      "next_steps": "Send proposal for Enterprise plan",
      "sentiment": "Positive",
      "engagement_level": "High"
    },
    "key_points": [
      "100 user requirement",
      "$5000 budget confirmed"
    ],
    "action_items": [
      {
        "action": "Send Enterprise plan proposal",
        "owner": "Sales Rep",
        "priority": "High"
      }
    ]
  },
  "source_type": "TEXT"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: Analysis failed

---

#### 4.2.3 Analyze Audio File
```
POST /analyze/audio
```

**Description**: Transcribe and analyze an audio file

**Request**: `multipart/form-data`
- `file` (required, file): Audio file (MP3, WAV, M4A, OGG)
- `transcript_id` (optional, string): Custom identifier
- `store_in_db` (optional, boolean): Store in database (default: true)

**Supported Formats**: MP3, WAV, M4A, OGG
**Max File Size**: 25 MB

**Response**: Same structure as `/analyze/text` with `source_type: "AUDIO"`

**Processing Steps**:
1. Validate audio file (format, size)
2. Transcribe using Azure OpenAI Whisper
3. Analyze transcript using GPT-4o
4. Store in vector database (if enabled)

**Error Responses**:
- `400 Bad Request`: Invalid file format or size
- `500 Internal Server Error`: Transcription or analysis failed

---

#### 4.2.4 Analyze Document File
```
POST /analyze/file
```

**Description**: Extract text from document and analyze

**Request**: `multipart/form-data`
- `file` (required, file): Document file (PDF, DOCX, CSV, XLSX, TXT)
- `transcript_id` (optional, string): Custom identifier
- `store_in_db` (optional, boolean): Store in database (default: true)

**Supported Formats**: PDF, DOCX, DOC, CSV, XLSX, XLS, TXT

**Response**: Same structure as `/analyze/text` with `source_type: "FILE"`

**Processing Steps**:
1. Validate file format
2. Extract text using appropriate processor
3. Analyze extracted text using GPT-4o
4. Store in vector database (if enabled)

**Error Responses**:
- `400 Bad Request`: Unsupported file format
- `500 Internal Server Error`: Extraction or analysis failed

---

#### 4.2.5 Chat with RASA Agent
```
POST /chat
```

**Description**: Chat with the AI agent about stored transcripts

**Request Body**:
```json
{
  "message": "What were the client's requirements?",
  "session_id": "optional-session-id"
}
```

**Parameters**:
- `message` (required, string): User's question or message
- `session_id` (optional, string): Session identifier for conversation tracking

**Response** (200 OK):
```json
{
  "success": true,
  "answer": "The client required a CRM system for 100 users with mobile access and a $5000 budget.",
  "relevant_documents": 3,
  "session_id": "session-uuid"
}
```

**Agent Behavior**:
- Uses LangChain ReAct framework
- Autonomously decides whether to search database
- Provides crisp, concise answers (1-2 sentences)
- Maintains conversation history within session

**Error Responses**:
- `500 Internal Server Error`: Agent execution failed

---

#### 4.2.6 Search Transcripts
```
POST /search
```

**Description**: Search for similar transcripts using semantic search

**Request Body**:
```json
{
  "query": "CRM requirements for enterprise clients",
  "top_k": 5
}
```

**Parameters**:
- `query` (required, string): Search query
- `top_k` (optional, integer): Number of results to return (default: 5)

**Response** (200 OK):
```json
{
  "success": true,
  "query": "CRM requirements for enterprise clients",
  "results": [
    {
      "transcript_id": "uuid-1",
      "transcript_text": "Sales conversation text...",
      "analysis_result": {
        "requirements": [...],
        "recommendations": [...],
        "summary": {...}
      },
      "source_type": "TEXT",
      "similarity_score": 0.92
    }
  ],
  "total_results": 3
}
```

**Requirements**:
- Milvus vector database must be enabled
- At least one transcript must be stored

**Error Responses**:
- `503 Service Unavailable`: Vector database not available
- `500 Internal Server Error`: Search failed

---

#### 4.2.7 Sales Helper
```
POST /sales-helper
```

**Description**: Get sales recommendations based on client description

**Request Body**:
```json
{
  "user_input": "Client needs CRM for 100 users, has $5000 budget, wants mobile access"
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "requirements": [
    {
      "requirement": "CRM system for 100 users",
      "category": "Technical",
      "priority": "High",
      "details": "Scalability requirement"
    }
  ],
  "recommendations": [
    {
      "product_service": "Enterprise CRM Plan",
      "rationale": "Supports 100+ users with mobile features",
      "addresses_requirements": ["Technical", "Budget"],
      "key_benefits": ["Mobile access", "Scalability"],
      "next_steps": "Schedule demo",
      "priority": "High",
      "confidence": "High"
    }
  ],
  "similar_cases": 2
}
```

**Error Responses**:
- `500 Internal Server Error`: Processing failed

---

#### 4.2.8 Web Interface
```
GET /
```

**Description**: Serve the web interface

**Response**: HTML page with 3-panel layout
- Left Panel: Transcript input and file upload
- Middle Panel: Analysis results with button navigation
- Right Panel: RASA chat interface

---

### 4.3 Request/Response Schemas

#### 4.3.1 Data Models

**Requirement**:
```python
{
  "requirement": str,      # Description of requirement
  "priority": str,         # "High" | "Medium" | "Low"
  "mentioned_by": str,     # "Client" | "Sales Rep"
  "context": str          # Context from conversation
}
```

**Recommendation**:
```python
{
  "recommendation": str,   # Specific recommendation
  "rationale": str,       # Why this is recommended
  "product_fit": str,     # How product addresses need
  "priority": str         # "High" | "Medium" | "Low"
}
```

**Summary**:
```python
{
  "overview": str,           # Brief overview
  "client_needs": str,       # Main client needs
  "pain_points": str,        # Identified pain points
  "opportunities": str,      # Sales opportunities
  "next_steps": str,         # Recommended actions
  "sentiment": str,          # "Positive" | "Neutral" | "Negative"
  "engagement_level": str    # "High" | "Medium" | "Low"
}
```

**ActionItem**:
```python
{
  "action": str,    # Specific action to take
  "owner": str,     # Who should do it
  "priority": str   # "High" | "Medium" | "Low"
}
```

### 4.4 Error Handling

**Standard Error Response**:
```json
{
  "detail": "Error message description"
}
```

**HTTP Status Codes**:
- `200 OK`: Successful request
- `400 Bad Request`: Invalid input parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error
- `503 Service Unavailable`: Required service (e.g., Milvus) unavailable

### 4.5 Rate Limiting

**Current Implementation**: No rate limiting

**Recommendations for Production**:
- Implement rate limiting per IP/API key
- Suggested limits:
  - `/analyze/*`: 10 requests/minute
  - `/chat`: 30 requests/minute
  - `/search`: 20 requests/minute

### 4.6 Authentication & Authorization

**Current Implementation**: No authentication

**Recommendations for Production**:
- Implement API key authentication
- Add OAuth2/JWT for user authentication
- Role-based access control (RBAC)

---

## 5. Data Flow & Workflows

### 5.1 Text Analysis Workflow

```
User Input (Text)
    │
    ▼
FastAPI Endpoint (/analyze/text)
    │
    ├─► Validate Input
    │
    ├─► Generate Transcript ID
    │
    ├─► TranscriptAnalyzer.analyze_transcript()
    │       │
    │       ├─► Check text length
    │       │
    │       ├─► Chunk if > 5000 chars (TextChunker)
    │       │
    │       ├─► Call Azure OpenAI GPT-4o (LiteLLM)
    │       │       │
    │       │       ├─► System Prompt
    │       │       └─► Analysis Prompt
    │       │
    │       └─► Parse JSON Response
    │
    ├─► Store in Milvus (if enabled)
    │       │
    │       ├─► Generate Embedding (text-embedding-3-small)
    │       │
    │       └─► Insert into Collection
    │
    └─► Return AnalysisResponse
```

### 5.2 Audio Analysis Workflow

```
User Upload (Audio File)
    │
    ▼
FastAPI Endpoint (/analyze/audio)
    │
    ├─► Save to Temp Directory
    │
    ├─► AudioProcessor.transcribe_audio()
    │       │
    │       ├─► Validate File (format, size)
    │       │
    │       ├─► Call Azure OpenAI Whisper API
    │       │
    │       └─► Return Transcript Text
    │
    ├─► TranscriptAnalyzer.analyze_transcript()
    │       (Same as Text Analysis)
    │
    ├─► Store in Milvus (if enabled)
    │
    ├─► Delete Temp File
    │
    └─► Return AnalysisResponse
```

### 5.3 Document Analysis Workflow

```
User Upload (Document File)
    │
    ▼
FastAPI Endpoint (/analyze/file)
    │
    ├─► Read File Content
    │
    ├─► DocumentProcessor.process_file()
    │       │
    │       ├─► Detect File Type (extension)
    │       │
    │       ├─► Route to Appropriate Processor
    │       │       │
    │       │       ├─► PDF → PyPDF2
    │       │       ├─► DOCX → python-docx
    │       │       ├─► CSV → pandas
    │       │       ├─► XLSX → pandas
    │       │       └─► TXT → decode UTF-8
    │       │
    │       └─► Return Extracted Text
    │
    ├─► TranscriptAnalyzer.analyze_transcript()
    │       (Same as Text Analysis)
    │
    ├─► Store in Milvus (if enabled)
    │
    └─► Return AnalysisResponse
```

### 5.4 Chat Workflow (RASA Agent)

```
User Message
    │
    ▼
FastAPI Endpoint (/chat)
    │
    ├─► ChatAgent.chat()
    │       │
    │       ├─► LangChain ReAct Agent Executor
    │       │       │
    │       │       ├─► Thought: Analyze question
    │       │       │
    │       │       ├─► Action: Decide to use tool or answer directly
    │       │       │       │
    │       │       │       └─► Tool: search_database()
    │       │       │               │
    │       │       │               ├─► MilvusVectorStore.search_similar_transcripts()
    │       │       │               │
    │       │       │               └─► Return Context
    │       │       │
    │       │       ├─► Observation: Process tool result
    │       │       │
    │       │       ├─► Thought: Formulate answer
    │       │       │
    │       │       └─► Final Answer: Generate response
    │       │
    │       └─► Update Conversation Memory
    │
    └─► Return ChatResponse
```

### 5.5 Sales Helper Workflow

```
Salesperson Input
    │
    ▼
FastAPI Endpoint (/sales-helper)
    │
    ├─► SalesHelperAgent.help_salesperson()
    │       │
    │       ├─► Step 1: Extract Requirements
    │       │       │
    │       │       ├─► Call GPT-4o with requirement_extraction_prompt
    │       │       │
    │       │       └─► Parse JSON (requirements list)
    │       │
    │       ├─► Step 2: Search Similar Cases
    │       │       │
    │       │       ├─► Build search query from requirements
    │       │       │
    │       │       └─► MilvusVectorStore.search_similar_transcripts()
    │       │
    │       ├─► Step 3: Generate Recommendations
    │       │       │
    │       │       ├─► Format context from search results
    │       │       │
    │       │       ├─► Call GPT-4o with sales_recommendation_prompt
    │       │       │
    │       │       └─► Parse JSON (recommendations list)
    │       │
    │       └─► Step 4: Update Conversation History
    │
    └─► Return SalesHelperResponse
```

---

## 6. Deployment Guide

### 6.1 Prerequisites

**System Requirements**:
- Python 3.12 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB disk space

**External Services**:
- Azure OpenAI account with:
  - GPT-4o deployment
  - Whisper deployment
  - text-embedding-3-small deployment
- Milvus Cloud account (optional, for vector storage)

### 6.2 Environment Setup

#### 6.2.1 Clone Repository
```bash
git clone <repository-url>
cd Capstone-
```

#### 6.2.2 Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

#### 6.2.3 Install Dependencies
```bash
pip install -r requirements.txt
```

#### 6.2.4 Configure Environment Variables

Create `config/.env` file:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Milvus Configuration (Optional)
MILVUS_HOST=your-milvus-host.cloud.zilliz.com
MILVUS_PORT=443
MILVUS_USER=your-username
MILVUS_PASSWORD=your-password
MILVUS_SECURE=true
MILVUS_COLLECTION_NAME=sales_transcripts
```

### 6.3 Running the Application

#### 6.3.1 Development Mode
```bash
python run_api.py
```

Server starts at: `http://localhost:8000`

#### 6.3.2 Production Mode
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 6.3.3 Docker Deployment (Recommended for Production)

**Dockerfile**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run**:
```bash
docker build -t sales-transcript-analyzer .
docker run -p 8000:8000 --env-file config/.env sales-transcript-analyzer
```

### 6.4 Configuration Management

#### 6.4.1 Configuration Files

**config/config.yaml**:
- Azure OpenAI settings
- Milvus database settings
- FastAPI configuration
- Audio processing settings
- Analysis configuration
- Logging configuration

**config/prompts.yaml**:
- System prompts
- Analysis prompts
- Sales helper prompts
- Chat agent prompts

#### 6.4.2 Logging Configuration

**Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

**Log File**: `logs/app.log`
- Max Size: 10 MB
- Backup Count: 5 (rotation)

**Console Logging**: Enabled with color coding

### 6.5 Database Setup

#### 6.5.1 Milvus Cloud Setup

1. Create account at https://cloud.zilliz.com/
2. Create new cluster
3. Create collection (auto-created by application)
4. Copy connection details to `.env`

#### 6.5.2 Collection Schema

**Collection Name**: Configurable (default: `sales_transcripts`)

**Fields**:
- `transcript_id`: VARCHAR (primary key)
- `embedding`: FLOAT_VECTOR (1536 dimensions)
- `transcript_text`: VARCHAR
- `analysis_result`: VARCHAR (JSON string)
- `source_type`: VARCHAR
- `timestamp`: INT64

**Index**: IVF_FLAT with L2 metric

### 6.6 Monitoring & Maintenance

#### 6.6.1 Health Checks
```bash
curl http://localhost:8000/health
```

#### 6.6.2 Log Monitoring
```bash
tail -f logs/app.log
```

#### 6.6.3 Performance Metrics

**Key Metrics to Monitor**:
- API response times
- Azure OpenAI API latency
- Milvus query performance
- Memory usage
- Error rates

**Recommended Tools**:
- Prometheus + Grafana
- Azure Application Insights
- ELK Stack (Elasticsearch, Logstash, Kibana)

### 6.7 Backup & Recovery

#### 6.7.1 Milvus Backup
- Use Milvus backup utilities
- Schedule regular backups
- Store backups in cloud storage (Azure Blob, AWS S3)

#### 6.7.2 Configuration Backup
- Version control for config files (excluding `.env`)
- Secure storage for `.env` file

---

## 7. Business Documentation

### 7.1 Use Cases

#### 7.1.1 Sales Transcript Analysis
**Actor**: Sales Manager
**Goal**: Analyze sales conversations to extract insights

**Preconditions**:
- Sales conversation transcript available (text, audio, or document)

**Main Flow**:
1. User accesses web interface
2. User selects input type (text/file)
3. User provides transcript or uploads file
4. User clicks "Analyze Transcript"
5. System processes and displays analysis
6. User views requirements, recommendations, summary, action items via button navigation

**Postconditions**:
- Analysis stored in database
- Insights available for future reference

**Success Criteria**:
- Analysis completed within 30 seconds
- All sections (requirements, recommendations, summary, action items) populated
- Accuracy > 85% (based on manual review)

#### 7.1.2 Interactive Chat with RASA
**Actor**: Sales Representative
**Goal**: Query stored transcripts for specific information

**Preconditions**:
- At least one transcript stored in database

**Main Flow**:
1. User types question in chat interface
2. User clicks "Send"
3. RASA agent processes question
4. Agent searches database if needed
5. Agent provides concise answer
6. User can ask follow-up questions

**Postconditions**:
- Conversation history maintained
- User receives actionable information

**Success Criteria**:
- Response time < 10 seconds
- Answer relevance > 90%
- Concise responses (1-2 sentences)

#### 7.1.3 Sales Assistance
**Actor**: Salesperson
**Goal**: Get product recommendations based on client needs

**Preconditions**:
- Salesperson has client information

**Main Flow**:
1. Salesperson describes client needs
2. System extracts requirements
3. System searches similar past cases
4. System generates recommendations
5. Salesperson reviews and acts on recommendations

**Postconditions**:
- Recommendations provided
- Similar cases identified

**Success Criteria**:
- Requirement extraction accuracy > 90%
- Recommendations aligned with requirements
- Similar cases relevant (similarity score > 0.7)

### 7.2 Business Logic & Rules

#### 7.2.1 Priority Assignment
**High Priority**:
- Budget mentioned and confirmed
- Timeline urgent (< 1 month)
- Decision maker engaged

**Medium Priority**:
- Budget range discussed
- Timeline moderate (1-3 months)
- Influencer engaged

**Low Priority**:
- Budget not discussed
- Timeline flexible (> 3 months)
- Initial contact only

#### 7.2.2 Sentiment Analysis
**Positive**:
- Client asks about next steps
- Budget confirmed
- Multiple requirements discussed

**Neutral**:
- Information gathering phase
- Questions about features
- No commitment signals

**Negative**:
- Price objections
- Competitor mentions
- Disengagement signals

#### 7.2.3 Engagement Level
**High**:
- Client asks detailed questions
- Requests demo/proposal
- Discusses implementation

**Medium**:
- Client responds to questions
- Shows interest in features
- Asks about pricing

**Low**:
- Short responses
- Vague interest
- No follow-up questions

### 7.3 Success Criteria & Acceptance Requirements

#### 7.3.1 Functional Requirements
✅ **FR-1**: System shall accept text, audio, and document inputs
✅ **FR-2**: System shall analyze transcripts and extract requirements
✅ **FR-3**: System shall generate recommendations
✅ **FR-4**: System shall provide conversation summaries
✅ **FR-5**: System shall identify action items
✅ **FR-6**: System shall store transcripts in vector database
✅ **FR-7**: System shall provide semantic search
✅ **FR-8**: System shall provide chat interface
✅ **FR-9**: System shall maintain conversation history
✅ **FR-10**: System shall support multiple file formats

#### 7.3.2 Non-Functional Requirements
✅ **NFR-1**: Response time < 30 seconds for analysis
✅ **NFR-2**: Chat response time < 10 seconds
✅ **NFR-3**: Support files up to 25 MB
✅ **NFR-4**: 99% uptime (production)
✅ **NFR-5**: Secure API communication (HTTPS in production)
✅ **NFR-6**: Scalable to 1000+ concurrent users
✅ **NFR-7**: Data encryption at rest and in transit

#### 7.3.3 Acceptance Criteria

**Analysis Accuracy**:
- Requirement extraction: > 85% accuracy
- Recommendation relevance: > 80% accuracy
- Summary completeness: > 90% accuracy
- Sentiment detection: > 75% accuracy

**Performance**:
- Text analysis: < 15 seconds
- Audio transcription: < 30 seconds
- Document processing: < 20 seconds
- Chat response: < 10 seconds
- Search query: < 5 seconds

**Usability**:
- Single-page interface (no redirects)
- Intuitive file upload
- Clear button navigation for results
- Responsive chat interface

### 7.4 Workflow Documentation

#### 7.4.1 Sales Analysis Workflow
```
1. Pre-Call Preparation
   └─► Review similar past cases (Search)

2. Sales Call
   └─► Conduct conversation

3. Post-Call Analysis
   ├─► Upload transcript/audio
   ├─► Review analysis results
   │   ├─► Requirements
   │   ├─► Recommendations
   │   ├─► Summary
   │   └─► Action Items
   └─► Execute action items

4. Follow-up
   └─► Query chat agent for specific details
```

#### 7.4.2 Team Collaboration Workflow
```
1. Sales Rep uploads transcript
   └─► System analyzes and stores

2. Sales Manager reviews analysis
   ├─► Checks requirements accuracy
   ├─► Validates recommendations
   └─► Approves action items

3. Team members query via chat
   ├─► "What were client's pain points?"
   ├─► "What budget was discussed?"
   └─► "What are next steps?"

4. Knowledge Base Growth
   └─► Each transcript improves future recommendations
```

---

## 8. Technical Specifications

### 8.1 Dependencies

**Core Framework**:
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- pydantic==2.5.0

**AI/ML**:
- litellm==1.17.9
- langchain==0.3.0
- langchain-community==0.3.0
- langchain-core==0.3.0
- langchain-text-splitters==0.3.0

**Database**:
- pymilvus==2.3.4 (optional)

**Document Processing**:
- pypdf2==3.0.1
- python-docx==1.2.0
- pandas==2.3.3
- openpyxl==3.1.5

**Utilities**:
- pyyaml==6.0.1
- python-dotenv==1.0.0
- colorlog==6.8.0
- requests==2.31.0

### 8.2 API Rate Limits (Azure OpenAI)

**GPT-4o**:
- Tokens per minute: 150,000 (varies by subscription)
- Requests per minute: 1,000

**Whisper**:
- Requests per minute: 50

**Embeddings**:
- Tokens per minute: 350,000
- Requests per minute: 3,000

### 8.3 Security Considerations

**Current Implementation**:
- Environment variable-based configuration
- CORS enabled (all origins - development only)
- No authentication/authorization

**Production Recommendations**:
- Implement API key authentication
- Restrict CORS to specific origins
- Use HTTPS only
- Implement rate limiting
- Add input validation and sanitization
- Encrypt sensitive data
- Regular security audits
- Implement logging and monitoring

---

## 9. Troubleshooting

### 9.1 Common Issues

**Issue**: "Milvus not available"
**Solution**: Check Milvus credentials in `.env`, ensure Milvus service is running

**Issue**: "Azure OpenAI API error"
**Solution**: Verify API key, endpoint, and deployment names in `.env`

**Issue**: "File upload fails"
**Solution**: Check file size (< 25 MB), verify file format is supported

**Issue**: "Chat agent not responding"
**Solution**: Check vector database connection, verify transcripts are stored

### 9.2 Debug Mode

Enable debug logging in `config/config.yaml`:
```yaml
logging:
  level: "DEBUG"
```

---

## 10. Version History

**Version 1.0.0** (Current)
- Initial release
- Text, audio, document analysis
- RASA chat agent
- Sales helper agent
- Vector database integration
- Web interface with 3-panel layout

---

## 11. Contact & Support

**Development Team**: [Your Team Name]
**Email**: [support@example.com]
**Documentation**: This file
**API Docs**: http://localhost:8000/docs

---

**Document Version**: 1.0
**Last Updated**: 2025-11-14
**Status**: Complete


