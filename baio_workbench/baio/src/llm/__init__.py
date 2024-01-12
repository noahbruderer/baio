# This module is meant for initializing and getting the ChatOpenAI object.
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import streamlit as st
import os

# llm = None
# llm35 = None  # Declare llm35 at the global scope
# embedding = None

# def initialize_llm(model_name, openai_api_key):
#     global llm
#     global llm35  # Declare llm35 as global within the function
#     global embedding
#     llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)
#     llm35 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106", openai_api_key=openai_api_key)
#     embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
# def get_llm():
#     if llm is None:
#         raise Exception("LLM has not been initialized")
#     return llm

# /usr/src/app/baio/src/llm/__init__.py

from langchain.chat_models import ChatOpenAI
import streamlit as st

class LLM:
    _instance_35 = None
    _instance_selected = None
    _embedding = None
    @staticmethod
    def initialize(openai_api_key, selected_model):
        LLM._instance_35 = LLM._instance_35 or ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=openai_api_key)
        LLM._instance_selected = LLM._instance_selected or ChatOpenAI(model_name=selected_model, temperature=0, openai_api_key=openai_api_key)
        LLM._embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
    @staticmethod
    def get_instance(model_name=None):
        if model_name == "gpt-3.5-turbo":
            return LLM._instance_35
        elif model_name is None:
            return LLM._instance_selected
        else:
            raise ValueError("Invalid model name")
    @staticmethod
    def get_embedding():
        if LLM._embedding is None:
            raise Exception("Embedding has not been initialized")
        return LLM._embedding
    # @staticmethod
    # def get_selected_model_instance():
    #     if LLM._instance_selected is None:
    #         raise Exception(f"{LLM._selected_model_name} LLM has not been initialized")
    #     return LLM._instance_selected

        