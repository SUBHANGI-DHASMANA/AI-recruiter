from typing import Dict, Any
from .base_agent import BaseAgent
import json

class AnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="analyzer",
            instructions="""Analyze the extracted candidate data and provide insights on:
            - Skill proficiency levels
            - Relevance of experience to target job
            - Qualification adequacy
            - Strengths and weaknesses
            Return a detailed analysis report.
            """
        )
        
    async def run(self, messages: list) -> Dict[str, Any]:
        print("📊 Analyzer: Analyzing Candidate Data")
        extracted_data = eval(messages[-1]["content"])
        analysis_result = self._query_ollama(str(extracted_data))
        return {
            "analysis_report": analysis_result,
            "analysis_timestamp": "2024-03-14",
            "relevance_score": 90,
        }
    
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