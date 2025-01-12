# orchestrator.py
from .extractor_agent import ExtractorAgent
from .analyzer_agent import AnalyzerAgent
from .matcher_agent import MatcherAgent
from .screener_agent import ScreenerAgent
from .recommender_agent import RecommenderAgent
from .base_agent import BaseAgent
from typing import Dict, Any

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="orchestrator",
            instructions="""Orchestrate the workflow by integrating all other agents.
            - Extract information from resume
            - Analyze skills and experience
            - Match with relevant jobs
            - Screen candidates
            - Enhance profile summary
            - Provide final recommendation
            """
        )
        self._setup_agents()
    
    def _setup_agents(self):
        self.extractor = ExtractorAgent()
        self.analyzer = AnalyzerAgent()
        self.matcher = MatcherAgent()
        self.screener = ScreenerAgent()
        self.recommender = RecommenderAgent()

    async def run(self, messages: list) -> Dict[str, Any]:
        prompt = messages[-1]["content"]
        response = self._query_ollama(prompt)
        return self._parse_json_safely(response)
    
    async def process_application(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        print("🤖 Orchestrator: Processing Job Application")
        workflow_context = {
            "resume_data": resume_data,
            "status": "initiated",
            "current_stage": "extraction"
        }

        try:
            extracted_data = await self.extractor.run(
                [{"role": "user", "content": str(resume_data)}]
            )
            workflow_context.update(
                {"extracted_data": extracted_data, "current_stage": "analysis"}
            )
            analysis_results = await self.analyzer.run(
                [{"role": "user", "content": str(extracted_data)}]
            )
            workflow_context.update(
                {"analysis_result": analysis_results, "current_stage": "matching"}
            )
            job_matches = await self.matcher.run(
                [{"role": "user", "content": str(analysis_results)}]
            )
            workflow_context.update(
                {"job_matches": job_matches, "current_stage": "screening"}
            )
            screening_results = await self.screener.run(
                [{"role": "user", "content": str(workflow_context)}]
            )
            workflow_context.update(
                {"screening_result": screening_results, "current_stage": "recommendation"}
            )
            final_recommendation = await self.recommender.run(
                [{"role": "user", "content": str(workflow_context)}]
            )
            workflow_context.update(
                {"final_recommendation": final_recommendation, "status": "completed"}
            )
            return workflow_context
        except Exception as e:
            workflow_context.update({"status": "failed", "error": str(e)})
            raise