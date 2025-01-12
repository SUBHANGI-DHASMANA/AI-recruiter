from typing import Dict, Any
from pdfminer.high_level import extract_text
from .base_agent import BaseAgent

class ExtractorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="extractor",
            instructions="""Extract information from the resume:
            - Name
            - Email
            - Phone number
            - Address
            - Work experience
            - Education
            - Skills
            - Certifications
            - Projects
            - Publications
            - Awards
            - Languages
            - Hobbies
            """
        )
        
    async def run(self, messages: list) -> Dict[str, Any]:
        print("📄 Extractor: Parsing Resume")
        workflow_context = eval(messages[-1]["content"])
        resume_data = eval(messages[-1]["content"])
        if resume_data.get("file_path"):
            raw_text = extract_text(resume_data["file_path"])
        else:
            raw_text = resume_data.get("text", "")
        extracted_info = self._query_ollama(raw_text)
        return {
            "raw_text": raw_text,
            "structured_data": extracted_info,
            "extraction_status": "Completed",
        }
    
    def _extract_info(self, resume_text: str) -> Dict[str, Any]:
        info = {
            "name": self._extract_name(resume_text),
            "email": self._extract_email(resume_text),
            "phone_number": self._extract_phone_number(resume_text),
            "address": self._extract_address(resume_text),
            "work_experience": self._extract_work_experience(resume_text),
            "education": self._extract_education(resume_text),
            "skills": self._extract_skills(resume_text),
            "certifications": self._extract_certifications(resume_text),
            "projects": self._extract_projects(resume_text),
            "publications": self._extract_publications(resume_text),
            "awards": self._extract_awards(resume_text),
            "languages": self._extract_languages(resume_text),
            "hobbies": self._extract_hobbies(resume_text),
        }
        return info
    
    def _extract_name(self, resume_text: str) -> str:
        # Extract name from resume text
        # Placeholder implementation
        return "John Doe"
    
    def _extract_email(self, resume_text: str) -> str:
        # Extract email from resume text
        # Placeholder implementation
        return 