"""
RAG Engine - Core logic for document processing, embedding, and retrieval
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


class RAGEngine:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.vectorstore = None
        self.embeddings = None

    # ----- STEP 1: Load Documents -----
    def load_documents(self, docs_path: str = "sample_docs"):
        """Load all .txt files from the given path"""
        path = Path(docs_path)
        self.documents = []

        for file_path in path.glob("*.txt"):
            content = file_path.read_text()
            doc = Document(
                page_content=content,
                metadata={"source": file_path.name}
            )
            self.documents.append(doc)

        return self.documents

    # ----- STEP 2: Chunk Documents -----
    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Split documents into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        self.chunks = text_splitter.split_documents(self.documents)
        return self.chunks

    # ----- STEP 3: Create Embeddings & Vector Store -----
    def create_vectorstore(self):
        """Create embeddings and store in ChromaDB"""
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        # Generate unique IDs for each chunk to prevent duplicates
        ids = [f"{chunk.metadata['source']}_{i}" for i, chunk in enumerate(self.chunks)]

        # Create fresh vectorstore (in-memory, resets each time)
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            collection_name="rag_demo",
            ids=ids
        )
        return self.vectorstore

    def get_embedding(self, text: str):
        """Get embedding vector for a text"""
        if self.embeddings is None:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return self.embeddings.embed_query(text)

    # ----- STEP 4: Query & Retrieve -----
    def retrieve(self, query: str, k: int = 3):
        """Retrieve relevant chunks for a query"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def generate_answer(self, query: str, context: str):
        """Generate answer using Gemini WITH RAG context"""
        prompt = ChatPromptTemplate.from_template(
"""You are a helpful assistant. Use the context below to answer the question.
If the context is helpful, use it. If not, you can still provide a helpful answer.

Context from company documents:
{context}

Question: {question}

Provide a clear and helpful answer:"""
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        chain = prompt | llm | StrOutputParser()

        return chain.invoke({"context": context, "question": query})

    def generate_without_rag(self, query: str):
        """Generate answer using Gemini WITHOUT any context (No RAG)"""
        prompt = ChatPromptTemplate.from_template(
"""You are a helpful assistant. Answer the question based on your general knowledge.

Question: {question}

Answer:"""
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        chain = prompt | llm | StrOutputParser()

        return chain.invoke({"question": query})

    # ----- Full RAG Pipeline -----
    def query(self, question: str, k: int = 3):
        """Run full RAG pipeline: retrieve + generate"""
        # Retrieve relevant chunks
        results = self.retrieve(question, k=k)

        # Build context
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

        # Generate answer
        answer = self.generate_answer(question, context)

        return {
            "answer": answer,
            "retrieved_chunks": results,
            "context": context
        }

    # ----- AGENTIC RAG -----
    def create_rag_agent(self):
        """Create an agent with RAG tool that dynamically handles multi-part queries"""
        vectorstore = self.vectorstore

        @tool
        def search_documents(query: str) -> str:
            """Search company documents. Use a focused, specific query for best results.

            For multi-part questions, call this tool MULTIPLE times with different queries."""
            docs = vectorstore.similarity_search(query, k=3)
            if docs:
                content = "\n\n".join(
                    f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
                    for doc in docs
                )
                return content
            return "No relevant information found for this query."

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

        agent = create_react_agent(
            model=llm,
            tools=[search_documents],
            prompt="""You are a helpful company assistant with access to search company documents.

WHEN TO USE THE SEARCH TOOL:
- Questions about company policies, HR, IT, expenses, leave, TechCorp → USE search_documents
- General knowledge questions (math, capitals, common facts) → Answer directly, NO search needed

FOR MULTI-PART QUESTIONS:
If the question asks about multiple different topics (e.g., "leave policy AND password requirements AND core values"),
make SEPARATE search calls for each topic:
- Search 1: "leave policy"
- Search 2: "password requirements"
- Search 3: "core values"

HOW TO ANSWER:
- Use information from search results to give helpful answers
- Cite which document the information came from
- Be clear and comprehensive"""
        )

        return agent, [search_documents]

    def run_agent(self, question: str):
        """Run the agentic RAG and capture ALL tool calls for complex multi-part questions"""
        from langchain_core.messages import AIMessage

        agent, tools = self.create_rag_agent()

        # Collect all steps and ALL tool calls across iterations
        steps = []
        final_answer = ""
        all_tool_calls = []
        all_retrieved_docs = []

        # Stream with higher recursion limit to allow multiple tool calls
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
            config={"recursion_limit": 50}
        ):
            last_msg = step["messages"][-1]
            steps.append(last_msg)

            # Collect ALL tool calls (append, don't overwrite)
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    all_tool_calls.append(tc)

            # Collect ALL retrieved docs from ToolMessages
            if hasattr(last_msg, 'artifact') and last_msg.artifact:
                all_retrieved_docs.extend(last_msg.artifact)

            # Get final answer - ONLY from AIMessage (not HumanMessage or ToolMessage)
            if isinstance(last_msg, AIMessage) and last_msg.content and isinstance(last_msg.content, str) and last_msg.content.strip():
                if not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls):
                    final_answer = last_msg.content

        return {
            "answer": final_answer,
            "steps": steps,
            "tool_calls": all_tool_calls,
            "retrieved_docs": all_retrieved_docs,
            "num_searches": len(all_tool_calls)
        }
