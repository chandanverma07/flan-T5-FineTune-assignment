import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np 
import torch
import logging
import logging
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv ,find_dotenv
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
def load_model(model_name="google/flan-t5-small"):
    try:
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logger.info("Model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"An error occurred while loading the model: {e}")
        return None, None
class Verify_flan_model:
    def __init__(self, model_name="google/flan-t5-small"):
        self.logger = logger
        try:
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred while loading the model and tokenizer: {e}")

    def summarize(self, text_to_summarize):
        try:
            summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
            summary = summarizer(text_to_summarize)
            self.logger.info(f"Summarization: {summary}")
            return summary
        except Exception as e:
            self.logger.error(f"An error occurred in summarization: {e}")
            return None

    def answer_question(self, question, context):
        try:
            qa_input = f"question: {question} context: {context}"
            answer = self.model.generate(**self.tokenizer(qa_input, return_tensors="pt"))
            decoded_answer = self.tokenizer.decode(answer[0])
            self.logger.info(f"Q&A: {decoded_answer}")
            return decoded_answer
        except Exception as e:
            self.logger.error(f"An error occurred in Q&A: {e}")
            return None

    def translate_to_french(self, text_to_translate):
        try:
            translator = pipeline("translation_en_to_fr", model=self.model, tokenizer=self.tokenizer)
            translation = translator(text_to_translate)
            self.logger.info(f"Translation: {translation}")
            return translation
        except Exception as e:
            self.logger.error(f"An error occurred in translation: {e}")
            return None
class LanguageModelHandler:
    def __init__(self, model_repo_id):
        # Initialize HuggingFace Language Model
        self.llm = HuggingFaceHub(repo_id=model_repo_id)
    
    def get_prompt(self, task, input_text, context=None):
        if task == "summarization":
            return f"Summarize the following text: {input_text}"
        elif task == "q&a":
            return f"Question: {input_text}\nContext: {context}\nAnswer:"
        elif task == "translation":
            return f"Translate the following English text to French: {input_text}"
        else:
            raise ValueError("Unsupported task")
    
    def execute_task(self, task, input_text, context=None):
        try:
            prompt = self.get_prompt(task, input_text, context)
            result = self.llm(prompt)
            return result
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

def load_model_hf_tf(model_name="transformers"):
    try:
        if model_name == "huggingface":
            # Assuming HuggingFaceHub is a custom function or class you have defined
            # Replace 'HuggingFaceHub' with the actual function/class you're using to load the model
            model = HuggingFaceHub(repo_id="google/flan-t5-large")
        elif model_name == "transformers":
            tokenizer, model = load_model()
        else:
            raise ValueError("Unsupported model type specified")
        
        logger.info(f"Model loaded successfully: {model_name}")
        return model

    except Exception as e:
        logger.error(f"An error occurred while loading the model: {e}")
        return None
def get_model_details(model):
    try:
        # Initialize a list to store model details
        model_details_list = []

        # Iterate through model parameters and add to the list
        for name, param in model.named_parameters():
            model_details_list.append({'Layer Name': name, 'Dimension': str(param.size())})

        # Convert list to DataFrame
        model_details_df = pd.DataFrame(model_details_list)

        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total number of parameters: {total_params}")

        return model_details_df, total_params

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return pd.DataFrame(), 0

