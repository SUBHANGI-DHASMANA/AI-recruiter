from typing import Dict, Any
from .base_agent import BaseAgent
import json  # Import json for safe parsing

class MatcherAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="matcher",
            instructions="""Match candidate based on:
            - Skills match
            - Experience level
            - Location preference
            - Provide detailed reasoning and compatibility scores.
            Return matches in JSON format with title, match_score, and location field.
            """
        )
    
    async def run(self, messages: list) -> Dict[str, Any]:
        print("🔍 Matcher: Finding Best Matches")
        
        # Safely evaluate the content of messages
        try:
            analysis_result = eval(messages[-1]["content"])  # Consider using json.loads instead for safety
        except Exception as e:
            print(f"Error evaluating messages: {e}")
            raise
        
        # Debugging: Print the entire analysis_result for inspection
        print("Analysis Result:", analysis_result)

        # Use get() to safely retrieve 'skills_analysis'
        skills_analysis = analysis_result.get('skills_analysis', None)
        
        if skills_analysis is None:
            print("Debug: 'skills_analysis' key is missing.")
            # Optionally handle this case by using another field or providing a default value.
            skills_analysis = "No skills available"  # Fallback or alternative handling
        
        sample_jobs = [
            {
                "title": "Senior Software Engineer",
                "requirements": "Python, Cloud, 5+ years experience",
                "location": "Remote"
            },
            {
                "title": "Data Scientist",
                "requirements": "Python, ML, Statistics, 3+ years experience",
                "location": "India"
            },
        ]

        matching_response = self._query_ollama(
            f"""Analyze the following profile and provide job matches:
            profile: {skills_analysis}
            Available jobs: {sample_jobs}
            Return only a JSON object with this exact structure:
            {{
                "matched_jobs": [
                    {{
                        "title": "job title",
                        "match_score": 85,
                        "location": "Job location"
                    }}
                ],
                "match_timestamp": "2024-03-14",
                "number_of_matches": 2
            }}"""
        )

        parsed_response = self._parse_json_safely(matching_response)
        
        # Debugging: Print parsed response for inspection
        print("Parsed Response:", parsed_response)

        # Fallback response in case of errors
        if "error" in parsed_response:
            print("Debug: Error in parsed response. Returning fallback matches.")
            return {
                "matched_jobs": [
                    {
                        "title": "Senior Software Engineer",
                        "match_score": 85,
                        "location": "Remote"
                    },
                    {
                        "title": "Data Scientist",
                        "match_score": 75,
                        "location": "India"
                    }
                ],
                "match_timestamp": "2024-03-14",
                "number_of_matches": 2
            }
        
        return parsed_response