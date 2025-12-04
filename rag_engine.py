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
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            collection_name="rag_demo"
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
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering questions based on the provided context.

Use ONLY the information from the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that question."

Context:
{context}"""),
            ("human", "{question}")
        ])

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        chain = prompt | llm | StrOutputParser()

        return chain.invoke({"context": context, "question": query})

    def generate_without_rag(self, query: str):
        """Generate answer using Gemini WITHOUT any context (No RAG)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the question based on your general knowledge."),
            ("human", "{question}")
        ])

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
        """Create an agent with RAG tool"""
        vectorstore = self.vectorstore

        @tool(response_format="content_and_artifact")
        def search_company_docs(query: str):
            """Search company documents for HR policies, IT policies, and expense policies.
            Use this tool when you need information about company rules, leave policies,
            password requirements, expense limits, or any internal company information."""
            docs = vectorstore.similarity_search(query, k=3)
            content = "\n\n".join(
                f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
                for doc in docs
            )
            return content, docs

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

        agent = create_react_agent(
            model=llm,
            tools=[search_company_docs],
            prompt="""You are a helpful company assistant with access to internal documents.

When users ask about company policies (HR, IT, expenses, leave, TechCorp etc.),  or anything about techcorp use the search_company_docs tool to find accurate information.
if the query has more than one part use the tool more than once to get a good analysis so basically repeat the prcoess of analysis and all

For general questions not related to company policies or techcorp, answer from your knowledge.

Always be helpful and cite which document the information came from when using the tool."""
        )

        return agent, search_company_docs

    def run_agent(self, question: str):
        """Run the agentic RAG and capture ALL tool calls for complex multi-part questions"""
        agent, tool_func = self.create_rag_agent()

        # Collect all steps and ALL tool calls across iterations
        steps = []
        final_answer = ""
        all_tool_calls = []
        all_retrieved_docs = []

        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            last_msg = step["messages"][-1]
            steps.append(last_msg)

            # Collect ALL tool calls (append, don't overwrite)
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    all_tool_calls.append(tc)

            # Collect ALL retrieved docs
            if hasattr(last_msg, 'artifact') and last_msg.artifact:
                all_retrieved_docs.extend(last_msg.artifact)

            # Get final answer (last content message)
            if hasattr(last_msg, 'content') and isinstance(last_msg.content, str):
                final_answer = last_msg.content

        return {
            "answer": final_answer,
            "steps": steps,
            "tool_calls": all_tool_calls,
            "retrieved_docs": all_retrieved_docs,
            "num_searches": len(all_tool_calls)
        }
