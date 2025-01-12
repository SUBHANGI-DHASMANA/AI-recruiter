from typing import Dict, Any
import json
from openai import OpenAI

class BaseAgent:
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions
        self.ollama_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key = "ollama"
        )
    async def run(self, messages: list) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement run()")
    
    def _query_ollama(self, prompt: str) -> str:
        try:
            response = self.ollama_client.chat.completions.create(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return "I'm sorry, I'm having trouble right now. Please try again later."
        
    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}