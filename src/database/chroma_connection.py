import logging
from pathlib import Path

import torch
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder

from src.utils.my_logging import setup_logging
from src.utils.load_config import load_config
from src.utils.pdf_processing import chunk_pdf

setup_logging()
logger = logging.getLogger("db")

class ChromaConnection:
    """
    Class to automatically create default collection which stores documents to query.
    Offer flexibility to add new collections and query other collections.
    """
    def __init__(self, config_path: str = 'config/database/chroma_db.yaml'):
        # load config
        self.config_path = config_path
        self.config = load_config(self.config_path)
        logger.info(f"Configuration loaded from {self.config_path}")

        # set the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}.")

        # create chroma client linked to local storage
        chroma_cfg = self.config["chroma"]
        self.persist_directory = chroma_cfg["persist_directory"]
        self.distance_metric = chroma_cfg["distance_metric"]
        self.add_batch_size = chroma_cfg["batch_size"]

        # create storage dir if missing 
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # get default chunking 
        chunk_cfg = self.config["chunking"]
        self.chunk_size = chunk_cfg["chunk_size"]
        self.chunk_overlap = chunk_cfg["chunk_overlap"]

        # dictionary cache to store embedding functions to prevent reload which add latency
        # key is embedding function name
        self._embedding_fns = {}
        
        # create the default collection with the pdf and embedding model
        default_collection_cfg = self.config["default_collection"]
        self.default_collection_name = default_collection_cfg["collection_name"]
        self.default_embedding_model = default_collection_cfg["embedding_model"]
        self.pdf_path = default_collection_cfg["pdf_path"]
        
        # create collection 
        logger.info("Starting to create default collection.")
        self.create_collection(self.default_collection_name, self.pdf_path, self.default_embedding_model)

        # set reranking model (same across all collections)
        rerank_cfg = self.config["reranker"]
        self.cross_encoder_model = rerank_cfg["cross_encoder_model"]
        self.rerank_batch_size = rerank_cfg["batch_size"]
        self.rerank_max_length = rerank_cfg["max_length"]
        self._reranker =  None  # initialize first, will be loaded at first call

   
    def _collection_exists(self, name: str) -> bool:
        """
        Check if a collection exists in the vector database by name.

        Args:
            name: Collection name to check if exist in database.
        
        Return:
            Boolean whether collection exist in database.
        """
        return name in {c.name for c in self.client.list_collections()}

    def _build_embedding_fn(self, embedding_model_name: str) -> SentenceTransformerEmbeddingFunction:
        """
        Get the embedding function from cache using the model name if present,
        else load and return the embedding function, adding it to cache to 
        prevent reloading it next time. 

        Args:
            embedding_model_name: Name of embedding model. Should be open source model from sentence transformer.
        
        Returns:
            Embedding model loaded form sentence transformer library.
        """
        if embedding_model_name not in self._embedding_fns:
            # load model from sentence transformer to be compatible to chromaDB
            self._embedding_fns[embedding_model_name] = SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name,
                device=self.device,
                normalize_embeddings=True,  # normalize embeddings to that distance metric/diff models
            )
            logger.info(f"Added embedding model {embedding_model_name} to cache.")

        return self._embedding_fns[embedding_model_name]

    # access via self.reranker
    @property
    def reranker(self) -> CrossEncoder:
        """Load the cross-encoder model at first call for reuse and prevent loading at setup."""
        if self._reranker is None:
            self._reranker = CrossEncoder(
                self.cross_encoder_model,
                device=self.device,
                max_length=self.rerank_max_length,
            )

        return self._reranker

    def create_collection(
        self,
        collection_name: str,
        pdf_path: str,
        embedding_model: str,
    ) -> None:
        """
        Create a collection by chunking the text in the pdf file, embedding the
        chunks in vectors and ingesting them into the vector database.

        No action done if collection already exists.
        
        Args:
            collection_name: Name to assign to new collection created.
            pdf_path: Path of pdf to load into collection.
            embedding_model: Name of embedding model to embed the text into vectors to ingest to collection.
        """
        if self._collection_exists(collection_name):
            logger.info(f"Collection {collection_name} already exists, no new creation of collection.")
            return 

        # Create collection with cosine distance
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self._build_embedding_fn(embedding_model),
            metadata={
                "hnsw:space": self.distance_metric,   # sepcify distance metric in collection
                "embedding_model": embedding_model,   # record embedding model use
                "source_pdf": Path(pdf_path).name,    # record pdf stored
            },
        )

        # get document chunks as list of strings from pdf
        doc_chunks = chunk_pdf(
            pdf_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_model=embedding_model,
        )

        # set IDs of docs to be integers starting from 1 in string format (required)
        ids = [str(i) for i in range(1, len(doc_chunks) + 1)]

        # add by batches to collection
        # no metadata for each doc
        for start in range(0, len(doc_chunks), self.add_batch_size):
            collection.add(
                ids=ids[start:start + self.add_batch_size],
                documents=doc_chunks[start:start + self.add_batch_size],
                # metadatas
            )

        logger.info(f"Inserted pdf {pdf_path} into collection {collection_name} with embedding model {embedding_model}")
        return 


    def load_collection(self, collection_name: str) -> Collection | None:
        """
        Get a collection with its corresponding embedding model attached to it.

        Args: 
            collection_name: Name of collection to load.
        
        Returns:
            The chromaDB Collection object that matched the collection name or 
            None if collection does not exist.
        """
        if not self._collection_exists(collection_name):
            logger.info(f"Fail to get collection as collection {collection_name} does exist in in database.")
            return 

        # get metadata of collection and return collection with embedding model attached
        metadata = self.client.get_collection(collection_name).metadata
        embedding_model_used = metadata["embedding_model"]

        return self.client.get_collection(
            collection_name, embedding_function=self._build_embedding_fn(embedding_model_used)
        )

    def get_top_k(
        self, input_query: str, top_k: int, collection_name: str,
    ) -> list[dict]:
        """
        Use input_query to search through the collection to get the top k matches.

        Args:
            input_query: Query string to search collection.
            top_k: Number of documents to return.
            collection_name: Name of collection to search the query.
        
        Returns:
            List containing the top k matches. Each match is represented by a dictionary containing the
            corresponding ID, text, metadata and distance.

        Raises:
            ValueError if collection does not exist.
        """
        collection = self.load_collection(collection_name)
        if collection is None:
            raise ValueError(f"Fail to query collection as collection {collection_name} does not exist.")

        # query to get document and corresponding distance score and metadata
        # ids is returned by default
        # return maximum top_k
        result = collection.query(
            query_texts=[input_query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # convert each of the top k doc to a dictionary
        return [
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],  
            }
            for i in range(len(result["ids"][0]))
        ]


    def query_collection_with_rerank(
        self,
        input_query: str,
        top_k: int,
        num_return_docs: int,
        collection_name: str,
    ) -> list[dict]:
        """
        Retrieve top_k matches of the query using method `get_top_k`, 
        rerank the top k documents using a more powerful cross-encoder model
        to get select the more relevant documents.
        Return the top num_return_docs documents.

         Args:
            input_query: Query string to search collection.
            top_k: Number of documents to return from the general search.
            num_return_docs: Number of documents to return after reranking the top k documents.
            collection_name: Name of collection to search the query.
        
        Returns:
            List containing the num_return_docs matches. Each match is represented by a dictionary containing the
            corresponding ID, text, metadata, distance and rerank score. List is sorted in descending rerank score.

        Raises:
            ValueError if collection does not exist or when the num_return_docs exceed the top_k value.
        """

        if num_return_docs > top_k:
            raise ValueError(f"Number of document to return ({num_return_docs}) exceeds top k ({top_k})")

        # get top k 
        candidates = self.get_top_k(input_query, top_k, collection_name)
        if not candidates:
            return []

        # get the query + text pair for each top k to be used to get cross encoder score
        pairs = [(input_query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs, batch_size=self.rerank_batch_size)

        # add rerank score
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)
        # sort in decreasing rerank score since closer to 1 stronger relevance
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        return candidates[:num_return_docs]


if __name__ == "__main__":
    # test 
    db = ChromaConnection()

    input_query = """If a buyer hasn't received their parcel and the status on the Shopee app shows 
    it as delivered, what steps should they take to request a refund and what are the policies regarding parcel not delivered?"""

    num_k = 15
    num_return = 5
    def_name = "refund_return_policy"

    return_docs = db.query_collection_with_rerank(input_query, num_k, num_return,  def_name)

    rscore = [d["rerank_score"] for d in return_docs]
    cscore = [1-d["distance"] for d in return_docs]
    all_id = [d["id"] for d in return_docs]
    all_text = [d["text"] for d in return_docs]

    logger.info(f"ID selected: {all_id}")
    logger.info(f"rerank score: {rscore}")
    logger.info(f"cosine similairty: {cscore}")

    for t in all_text:
        logger.info(f"Text: {t}\n\n")

