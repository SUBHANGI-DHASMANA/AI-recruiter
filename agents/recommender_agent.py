from typing import Dict, Any
from .base_agent import BaseAgent

class RecommenderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="recommender",
            instructions="""Generate final recommendation considering:
            1. Extracted profile
            2. Skills analysis
            3. Job matches
            4. Screening result
            Provide clear next steps and specific recommendations.
            """
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        print("🎯 Recommender: Generating Final Recommendation")
        workflow_context = eval(messages[-1]["content"])
        recommendation = self._query_ollama(str(workflow_context))
        return {
            "final_recommendation": recommendation,
            "recommendation_timestamp": "2024-03-14",
            "confidence_level": "high"
        }