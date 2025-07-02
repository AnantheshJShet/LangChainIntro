import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate


def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return pages
    except Exception as e:
        if "cryptography" in str(e) or "AES algorithm" in str(e):
            print("Error: This PDF appears to be encrypted and requires a password.")
            print("Please provide an unencrypted PDF or install cryptography library:")
            print("pip install cryptography>=3.1")
            raise
        elif "could not open" in str(e).lower():
            print(f"Error: Could not open the PDF file at '{file_path}'")
            print("Please check if the file path is correct and the file exists.")
            raise
        else:
            print(f"Error loading PDF: {e}")
            raise


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def create_embeddings(chunks, embedding_model="nomic-embed-text:latest"): 
    embeddings = OllamaEmbeddings(model=embedding_model)
    return embeddings


def store_in_faiss(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def load_faiss_index(index_path, embeddings):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Vector store index not found at '{index_path}'. Please process a PDF first (option 1).")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


def generate_mcqs_from_vectorstore(vectorstore, llm_model="llama3.2:1b", num_questions=5, query="important concepts"):
    llm = OllamaLLM(model=llm_model)
    prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template=(
            "Given the following context, generate {num_questions} multiple choice questions (with 4 options each) and their correct answers. "
            "Format the output as: Question, Options (A, B, C, D), and Answer.\nContext:\n{context}"
        )
    )
    chain = prompt | llm
    
    # Retrieve relevant chunks from vector store
    relevant_docs = vectorstore.similarity_search(query, k=3)
    context_text = "\n".join([doc.page_content for doc in relevant_docs])
    
    result = chain.invoke({"context": context_text, "num_questions": num_questions})
    return result

def generate_mcqs(text, llm_model="llama3.2:1b", num_questions=5):
    llm = OllamaLLM(model=llm_model)
    prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template=(
            "Given the following context, generate {num_questions} multiple choice questions (with 4 options each) and their correct answers. "
            "Format the output as: Question, Options (A, B, C, D), and Answer.\nContext:\n{context}"
        )
    )
    chain = prompt | llm
    result = chain.invoke({"context": text, "num_questions": num_questions})
    return result


def main():
    print("MCQ Creator with Vector Store")
    print("1. Process new PDF and create vector store")
    print("2. Use existing vector store")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        pdf_path = input("Enter the path to the PDF file: ")
        print("Loading PDF...")
        documents = load_pdf(pdf_path)
        print(f"Loaded {len(documents)} pages.")

        print("Splitting document into chunks...")
        chunks = split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")

        print("Creating embeddings and storing in FAISS...")
        embeddings = create_embeddings(chunks)
        vectorstore = store_in_faiss(chunks, embeddings)
        print("Embeddings stored in FAISS vector store.")
        
        # Save vector store for future use
        vectorstore.save_local("faiss_index")
        print("Vector store saved as 'faiss_index' for future use.")
        
    elif choice == "2":
        print("Loading existing vector store...")
        try:
            embeddings = create_embeddings([])  # Create embeddings object
            vectorstore = load_faiss_index("faiss_index", embeddings)
            print("Vector store loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run option 1 first to process a PDF and create the vector store.")
            return
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Interactive MCQ generation
    while True:
        print("\n" + "="*50)
        print("MCQ Generation Options:")
        print("1. Generate MCQs on general concepts")
        print("2. Generate MCQs on specific topic")
        print("3. Search and display relevant chunks")
        print("4. Exit")
        
        mcq_choice = input("Enter your choice (1-4): ")
        
        if mcq_choice == "1":
            print("Generating MCQs on general concepts...")
            mcqs = generate_mcqs_from_vectorstore(vectorstore, query="important concepts")
            print("\nGenerated MCQs:\n")
            print(mcqs)
            
        elif mcq_choice == "2":
            topic = input("Enter the specific topic for MCQs: ")
            print(f"Generating MCQs on '{topic}'...")
            mcqs = generate_mcqs_from_vectorstore(vectorstore, query=topic)
            print("\nGenerated MCQs:\n")
            print(mcqs)
            
        elif mcq_choice == "3":
            search_query = input("Enter search query: ")
            print(f"Searching for chunks related to '{search_query}'...")
            relevant_docs = vectorstore.similarity_search(search_query, k=3)
            print("\nRelevant chunks:\n")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"Chunk {i}:")
                print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                print("-" * 50)
                
        elif mcq_choice == "4":
            print("Exiting. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
