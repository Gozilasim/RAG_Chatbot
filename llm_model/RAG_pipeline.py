# Load Modules
import os
import base64
import httpx
from pathlib import Path
from llm_model.config import config
import openai


from llama_index.callbacks import CallbackManager, LlamaDebugHandler
#from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms import OpenAILike
from llama_index import (LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage)

from llm_model.Prompt_Template import fmea_template, query_wrapper_prompt






class RAG:
    def __init__(self, config=config):
        service_user = "CEGpt"
        service_password = "__gHTNQn9_wn!$z_"
        self.config = config
        self.embedEndpoint = self.config['SERVICE_HOST_EMB']
        self.llmEndpoint = self.config['SERVICE_HOST_LLM']
        self.storage_path = Path("", self.config['local_index_storage_path'])
        self.api_key = self.generate_base64_string(username=service_user, password=service_password)
        self.headers = self.generate_headers()
        self.embedModel = self._init_embedding_model(endpoint=self.embedEndpoint, model = self.config['embed_model_name'])
        self.llm = self._init_llm_model(fmea_template, query_wrapper_prompt, endpoint=self.llmEndpoint, model_name='llama3.1-70b')
        self.service_context = self.create_service_context(self.embedModel, self.llm)
        self.idx = self.load_index(self.storage_path, self.service_context)

    # initialize llm model
    def _init_llm_model(self, system_prompt, query_wrapper_prompt, endpoint,
                    model_name,  
                    **kwargs):
        """To initialize LLM model

        Args:
            endpoint (str): Centralized AML-LLM services hosted URL
            model_name (str): model_name

        Returns:
            object: return LLM model object
        """
        client = self._init_openai_client(endpoint)
        
        llm = OpenAILike(model=model_name, context_window=self.config['CONTEXT_WINDOW'], 
                         temperature=kwargs.get('temperature', 0.1), 
                         max_tokens=self.config['MAX_OUTPUT_TOKENS'],
                         system_prompt = system_prompt, 
                         query_wrapper_prompt = query_wrapper_prompt)
        
        llm._client = client
        return llm
    
    # initialize openai client
    def _init_openai_client(self, endpoint):
        """To initialize AML hosted services

        Args:
            endpoint (str): Centralized AML-LLM services hosted URL

        Returns:
            object: _description_
        """
        client = openai.OpenAI(
                            api_key=self.api_key, ## Generated usinf basic/bearer token
                            base_url=endpoint,
                            default_headers=self.headers,
                            http_client = httpx.Client(verify=self.config['CA_PATH']))
        return client  

    # initialize embedding model
    def _init_embedding_model(self, endpoint, model, batch_size=100,):
        """To initialize embedding model

        Args:
            endpoint (str): Centralized AML-LLM services hosted URL
            batch_size (int, optional): batch_size. Defaults to 100.
            model (str, optional): model name mapped to AML-LLM embedding model (embed-large). Defaults to "text-embedding-ada-002".

        Returns:
            object: return embedding model
        """
        client = self._init_openai_client(endpoint)
        embed_model = OpenAIEmbedding(model=model, ## Mapped to AML-LLM embedding model (embed-large)
                                    embed_batch_size=batch_size)
        embed_model._client = client
        return embed_model   

    def generate_base64_string(self, username, password):
        """
        To Encode username & password strings into base64
        """

        sample_string = username + ":" + password
        sample_string_bytes = sample_string.encode("ascii")

        base64_bytes = base64.b64encode(sample_string_bytes)
        base64_string = base64_bytes.decode("ascii")
        return base64_string
    
    def create_service_context(self, embed_model,custom_llm):
        """Create service context to configure index and query
        Args:
            embed_model (object): embedding model
            custom_llm (object): LLM model
        Returns:
            object: service context object
        """
        llm_predictor = LLMPredictor(llm=custom_llm)
        chunk_limit = self.config['CHUNK_LIMIT']
        max_input_size = custom_llm.context_window
        num_output = self.config['MAX_OUTPUT_TOKENS']
        max_chunk_overlap = 0.01
        prompt_helper = PromptHelper(max_input_size, 
                                    num_output,
                                    max_chunk_overlap)
        
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                                            prompt_helper=prompt_helper,
                                                            embed_model=embed_model, 
                                                            chunk_size_limit=chunk_limit,
                                                            callback_manager=callback_manager)
        return service_context
    
    def load_index(self, storage_path, service_context):
        """To load created index
        Args:
            index_path (str): Path where index is stored

        Returns:
            object: return index object
        """
        index_path = Path(storage_path)
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context=storage_context, service_context=service_context)

        return index

    def generate_headers(self):
        headers = {
            'accept': 'application/json',
            'Authorization': "Basic {}".format(self.api_key)
        }
        return headers

if __name__ == "__main__":
    rag = RAG()