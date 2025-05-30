import fitz  # PyMuPDF
import json
import os
import numpy as np
from typing import Tuple
from openai import OpenAI

# Initialize OpenAI client using environment variable
client = OpenAI()

class DocumentProcessingService:
    def __init__(self):
        pass  # No need for bedrock setup if using OpenAI only

    def extract_text_from_pdf(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()

    def generate_embedding(self, text: str) -> np.ndarray:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        return np.array(embedding)

    def classify_document(self, text: str) -> str:
        lowered = text.lower()
        if "recipe" in lowered or "ingredient" in lowered:
            if "dessert" in lowered:
                return "recipe > dessert"
            elif "breakfast" in lowered:
                return "recipe > breakfast"
            elif "dinner" in lowered:
                return "recipe > dinner"
            else:
                return "recipe > general"
        elif "marathon" in lowered:
            return "sports > endurance"
        elif "olympic" in lowered:
            return "sports > history"
        elif "recovery" in lowered:
            return "sports > recovery"
        elif "news" in lowered:
            return "sports > news"
        else:
            return "other"

    def process_document(self, file_path: str) -> Tuple[str, str, np.ndarray]:
        text = self.extract_text_from_pdf(file_path)
        category = self.classify_document(text)
        embedding = self.generate_embedding(text)
        return text, category, embedding
