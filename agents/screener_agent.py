from typing import Dict, Any
from .base_agent import BaseAgent

class ScreenerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="screener",
            instructions="""Screen candidate based on:
            - Qualifications alignment
            - Experience relevance
            - Skills match percentage
            - Cultural fit indicator
            - Red flags or concerns
            Provide comprehensive screering report.
            """
        )
        
    async def run(self, messages: list) -> Dict[str, Any]:
        print("👥 Screener: Conducting Initial Screening")
        workflow_context = eval(messages[-1]["content"])
        screening_result = self._query_ollama(str(workflow_context))
        return {
            "screening_report": screening_result,
            "screening_timestamp": "2024-03-14",
            "screening_score": 85,
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