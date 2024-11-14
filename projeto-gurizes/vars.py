CSS = """
    .contain { display: flex; flex-direction: column; }
    .gradio-container { height: 100vh !important; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; overflow: auto;}
    """

OLD_SYSTEM_PROMPT = """
    - You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
    - Generate human readable output, avoid creating output with gibberish text.
    - Generate only the requested output, don't include any other language before or after the requested output.
    - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
    - Generate professional language typically used in business documents in North America.
    - Never generate offensive or foul language.
    - If you create text based on a provided context, NEVER QUOTE THE OWN CONTEXT, generate text based on its content.
    - If the user talking to you did not mentioned the context, never mention that you receive a context.
    """


SYSTEM_PROMPT = """
        Você é um assistente de IA especializado em um grupo de chat. Siga estas regras:
        - APENAS crie mensagens com os usuarios DO GRUPO.
        - Quando o usuário disser "generate", você deve gerar uma conversa de um dia normal com base no "texto.txt."
        - Crie texto com base no contexto fornecido, sem citá-lo diretamente.
        - Ao gerar mensagens, adicione o nome e o horário em que a mensagem foi enviada.
        - Use o vocabulário utilizado pela pessoa.
        - Use palavrões quando necessario
        - Estruture a mensagem no formato (hora:minutos: nome : (mensagem)).
        - Use apenas tópicos que já foram discutidos no chat.
        - Pode responder com ou sem o uso de pontuação, acentos ou outros sinais diacríticos, conforme necessário. Use esses recursos apenas quando forem importantes para a clareza ou compreensão da mensagem, mas entenda que eles não são obrigatórios em suas respostas.
        - APENAS crie mensagens com os usuarios DO GRUPO.       
        - O usuário pode escolher entre 3 tipos de conversas: Quando ele usa "generate 1," a conversa deve ser sobre ex-relacionamentos; "generate 2," sobre política; "generate 3," uma discussão entre dois ou mais membros do grupo.
"""
LLMS = {
    "llama2_7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2_13b_chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama2_70b_chat": "meta-llama/Llama-2-70b-chat-hf",
    "llama_3_8b_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama_3_8b": "meta-llama/Meta-Llama-3-8B",
    "llama_3.1_8b": "meta-llama/Meta-Llama-3.1-8B",
    "llama_3.1_8b_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama_3_70b_instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistral_8x7b_instruct_v1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral_7b_instruct_v1": "mistralai/Mixtral-7B-v0.1",
    "mistral_7b_instruct_v2": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral_7b_instruct_v3": "mistralai/Mixtral-7B-Instruct-v0.3",
    "gemma_7b_instruct": "google/gemma-7b-it",  # Aqui estamos com problema no chat template
    "gemma2_9b_it": "google/gemma-2-9b-it",  # Aqui estamos com problema no chat template
    "dolly_12b": "databricks/dolly-v2-12b",
    "gptj": "EleutherAI/gpt-j-6b",
    "mistral": "NousResearch/Hermes-3-Llama-3.1-405B"
}

EMBEDDINGS = {
    "MiniLMv2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "stella_400M": "dunzhang/stella_en_400M_v5",
}
