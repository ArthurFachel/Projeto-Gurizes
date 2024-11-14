from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser, SemanticSplitterNodeParser, TokenTextSplitter, HierarchicalNodeParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import os
from vars import EMBEDDINGS
from argparse import ArgumentParser

def get_splitter(args):
    if args.splitter == "sentence":
        return SentenceSplitter()
    elif args.splitter == "sentence_window":
        return SentenceWindowNodeParser(window_size=3, window_metadata_key="window")
    elif args.splitter == "semantic":
        return SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)
    elif args.splitter == "token":
        return TokenTextSplitter(chunk_size=1024, chunk_overlap=20, separator=" ",)
    elif args.splitter == "hierarchical":
        return HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
    else:
        raise ValueError(f"Invalid splitter: {args.splitter}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--collection_name", type=str, default="book_collection")
    parser.add_argument("--splitter", type=str, default='sentence')
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("-e", "--embedding", type=str, default="MiniLMv2")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDINGS[args.embedding],
        cache_folder="../LLMs/cache",
        device="cuda:0",
        trust_remote_code=True
    )

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(args.collection_name)

    documents = SimpleDirectoryReader(args.data_path, recursive=args.recursive).load_data()

    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        transformations=[get_splitter(args)],
        embed_model=embed_model,
        show_progress=True
    )