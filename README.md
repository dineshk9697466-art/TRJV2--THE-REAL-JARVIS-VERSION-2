# 🤖 TRJV2 — The Real Jarvis

**An AI-powered conversational assistant with Retrieval-Augmented Generation (RAG), PDF question answering, semantic search, and persistent chat history.**

TRJV2 is the second version of my personal AI assistant project, **The Real Jarvis**. Built with Python and Streamlit, it combines general-purpose AI conversation with document-based question answering and persistent conversation management.

## ✨ Features

- **AI-Powered Chat:** Interact with the assistant using the Google Gemini API.
- **PDF Question Answering:** Upload PDF documents and ask questions about their content.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document passages and use them as context for AI-generated answers.
- **Semantic Search:** Use ChromaDB and Sentence Transformers to search document chunks by meaning.
- **Persistent Chat History:** Save and revisit separate conversations using SQLite.
- **Conversation Context:** Support follow-up questions using previous messages in the active conversation.
- **Multiple Question Modes:**
  - **Auto Mode:** Automatically determines when document retrieval is appropriate.
  - **General Chat:** Answers general questions without retrieving PDF content.
  - **Ask My PDFs:** Searches indexed documents for relevant information.
- **Source References:** Display retrieved document filenames and page numbers when available.
- **Model Fallback Handling:** Attempt an alternative model when the primary model request fails.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and backend |
| Streamlit | Interactive web interface |
| Google Gemini API | AI-powered responses |
| SQLite | Persistent chat and application data |
| ChromaDB | Vector storage and similarity search |
| Sentence Transformers | Text embeddings |
| PyPDF | PDF text extraction |
| python-dotenv | Environment variable configuration |
| pytest | Testing |

## 🏗️ Project Structure

```text
TRJV2/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── database/
│   └── db.py
├── rag/
│   └── rag_engine.py
└── screenshots/
    ├── 01-main-interface.png
    ├── 02-pdf-indexing.png
    ├── 03-pdf-question-answering.png
    ├── 04-chat-history.png
    └── 05-conversation-context.png
```

*The structure above shows the main project files. Local databases, uploaded PDFs, vector-store data, and environment files are excluded from the public repository.*

## ⚙️ How It Works

### 1. General Conversation
The user sends a message through the Streamlit interface. The application sends the request to the configured Gemini model and displays the response.

### 2. PDF Ingestion
When a PDF is uploaded, the application extracts its text and divides it into smaller chunks. The chunks are converted into embeddings and stored in ChromaDB with document metadata.

### 3. Semantic Retrieval
When a document-related question is asked, the application searches the vector database for relevant passages. These passages are supplied to the AI model as context.

### 4. Context-Aware Answer Generation
The model generates an answer using the retrieved passages. Source filenames and page numbers are displayed when available. If the indexed content does not contain sufficient information, the assistant should communicate that limitation rather than invent document facts.

### 5. Persistent Conversations
SQLite stores chat history so that conversations can be revisited. Previous messages help the assistant understand follow-up questions within the active conversation.

## 🚀 Getting Started

### Prerequisites

- Python installed on your system
- A Google Gemini API key
- Git, if cloning the repository

### Step 1: Clone the Repository

```bash
git clone YOUR_TRJV2_REPOSITORY_URL
cd TRJV2
```

Replace `YOUR_TRJV2_REPOSITORY_URL` with the actual GitHub repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure the Environment

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key.

**Security:** Never commit `.env` or expose your API key in screenshots, source code, or public repositories.

### Step 5: Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal to access TRJV2.

## 🔐 Data and Security

- Keep API credentials in environment variables.
- Exclude `.env`, virtual environments, local databases, and private uploaded PDFs from Git.
- Store local document indexes and vector database files outside the public repository.
- Do not upload confidential documents or personal information to a public repository.

## 🧪 Testing

Before using the application, test the main workflows:

- General questions and follow-up questions
- PDF indexing and document-based question answering
- Retrieved source filenames and page numbers
- Chat creation, history, and conversation switching
- PDF isolation between separate chats
- API error and fallback handling

## 🔮 Future Improvements

Potential future improvements include voice interaction, more advanced memory, additional document formats, and further performance optimization.

## 👨‍💻 Author

**Dinesh Kumar Saravanan**

GitHub: [dineshk9697466-art](https://github.com/dineshk9697466-art)

## 📄 License

A license has not yet been specified. Add a `LICENSE` file if you decide to distribute the project under an open-source license.

---

**TRJV2 — The Real Jarvis**

*Building practical AI applications with Python, LLMs, and Retrieval-Augmented Generation.*# TRJV2--THE-REAL-JARVIS-VERSION-2
AI conversational assistant with PDF RAG, semantic search, and persistent chat history.
