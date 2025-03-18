config = {
    'CA_PATH' : 'ca-bundle.crt',
    'CHUNK_LIMIT' : 3000,
    'MAX_OUTPUT_TOKENS' : 1024,
    'CONTEXT_WINDOW' : 4096,
    'SERVICE_HOST_EMB' : 'https://aml-llm-models.icp.infineon.com/v1',
    'SERVICE_HOST_LLM' : 'https://gpt4ifx.icp.infineon.com/',
    'VECTOR_DB_HOST' : 'http://localhost:6333', #Vector DB Host Endpoint
    'embed_model_batch_size':100,
    'embed_model_name':"text-embedding-ada-002",
    'llm_model_name': [
                        'llm70b', # default first
                       ],
    'local_index_storage_path' : 'index_storage/vector_index',
    'similarity_top_k': 2
}
