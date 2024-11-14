from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import FlatReader, PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.deepinfra import DeepInfraLLM
from llama_index.core.llms import ChatMessage, MessageRole

from argparse import ArgumentParser
from pathlib import Path
import gradio as gr
from vars import *
import chromadb
import torch
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, default="mistral")
    parser.add_argument("--embedding", type=str, default="MiniLMv2")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--embedding_device", type=str, default="cuda:0")
    parser.add_argument("--collection_name", type=str, default="PDFs")
    parser.add_argument("-cm", "--chat_mode", type=str, default="context")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create LLM Model
    llm = DeepInfraLLM(
    model="NousResearch/Hermes-3-Llama-3.1-405B",
    api_key="cPGaZl8ehRjUMYQa05X3fDMvFSKrbSPl",
    temperature=0.5,
    max_tokens=2500,
    additional_kwargs={"top_p": 0.9},
)

    # Create Embedding Model
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDINGS[args.embedding],
        cache_folder="../LLMs/cache",
        device=args.embedding_device,
    )

    # Create Vector Store
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name=args.collection_name)

    vector_store = ChromaVectorStore(
        chroma_collection=collection, embed_model=embed_model
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    chat_engine = index.as_chat_engine(
        chat_mode=args.chat_mode,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        # memory=ChatMemoryBuffer(token_limit=8192), # explore effects
    )

    def upload_file(file):
        from tqdm import tqdm

        try:
            reader = FlatReader()
            file = reader.load_data(Path(file))
        except:
            reader = PDFReader()
            file = reader.load_data(Path(file))
        nodes = SentenceSplitter().get_nodes_from_documents(file)
        for node in tqdm(nodes):
            node.embedding = embed_model.get_text_embedding(node.text)
        vector_store.add(nodes)

    def predict(message, history, something):
        print(message, history, something)
        chat = []
        if history:
            for item in history:
                chat.append(ChatMessage(role=MessageRole.USER, content=item[0]))
                chat.append(ChatMessage(role=MessageRole.ASSISTANT, content=item[1]))

        partial_message = ""
        for new_token in chat_engine.stream_chat(
            message, chat_history=chat
        ).response_gen:
            if new_token != "assistant\n\n":
                partial_message += new_token
                yield partial_message
        torch.cuda.empty_cache()

    with gr.Blocks(css=CSS) as demo:
        addfile_btn = gr.UploadButton(
            "Upload a file", file_count="single", render=False
        )
        addfile_btn.upload(upload_file, addfile_btn, [addfile_btn])

        gr.ChatInterface(
            predict,
            additional_inputs=[addfile_btn],
            concurrency_limit=2,
        )  # .launch(share=True)

    demo.queue().launch(share=True)
