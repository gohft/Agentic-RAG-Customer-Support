from pathlib import Path
from pypdf import PdfReader
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from pdf and concatenate them into a single document string.

    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
 
    reader = PdfReader(pdf_path)
    page_texts = []
    for page in reader.pages:
        # guard against None
        text = (page.extract_text() or "").strip()
        if text:
            page_texts.append(text)
 
    if not page_texts:
        raise ValueError(f"No text found in {pdf_path}")
 
    return "\n\n".join(page_texts)


def chunk_pdf(
    pdf_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,  # 10% of chunk size
    embedding_model: str = "BAAI/bge-base-en-v1.5",
) -> list[str]:
    """
    Chunk a PDF and return the chunks as list of strings.
    Chunking is done when size/overlap measured in tokens.
    These chunks can be feed into vector database/RAGAS.
    
    """
    # use the tokenizer of embedding model to be used in vector database
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    # split by Paragraph, then newline 
    # splitter that uses tokenizer
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    full_text = extract_text_from_pdf(pdf_path)
    chunks = splitter.split_text(full_text)

    return chunks


if __name__ == "__main__":
    chunks = chunk_pdf(
        pdf_path="shoppee_policy.pdf",
    )
    print(f"Generated {len(chunks)} chunks")
    print(f"first chunk: {chunks[0]}")
    print(f"last chunk: {chunks[-1]}")

    # --- Feed into ChromaDB (embed + insert iteratively) ---
    # import uuid
    # import chromadb
    # from some_embedding_lib import embed  # your embedding model call
    #
    # client = chromadb.PersistentClient(path="./chroma_db")
    # collection = client.get_or_create_collection("my_collection")
    #
    # for chunk in chunks:
    #     collection.add(
    #         ids=[str(uuid.uuid4())],
    #         documents=[chunk],
    #         embeddings=[embed(chunk)],
    #     )

    # --- Feed into RAGAS ---s
    # from langchain_core.documents import Document
    # from ragas.testset import TestsetGenerator
    #
    # lc_documents = [Document(page_content=c) for c in chunks]
    # generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)
    # testset = generator.generate_with_langchain_docs(lc_documents, testset_size=10)
