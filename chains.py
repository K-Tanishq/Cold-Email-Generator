import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# load_env()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
                        model="llama3-70b-8192",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2)
    
    
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template("""
                I will give you scraped text from the job posting. 
                Your job is to extract the job details & requirements in a JSON format containing the following keys: 'role', 'experience', 'skills', and 'description'. 
                Only return valid JSON. No preamble, please.
                Here is the scraped text: {page_data}
                """)    
        
        chain_extract = prompt_extract | self.llm
        response = chain_extract.invoke(input={"page_data" : cleaned_text})
        
        try:
            json_parser = JsonOutputParser()
            response = json_parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Content too big, unable to parse jobs.")
        
        return response if isinstance(response, list) else [response]


    def write_email(self, job_description, portfolio_urls):
        prompt_email = PromptTemplate.from_template(
                 """
        I will give you a role and a task that you have to perform in that specific role.        
        Your Role: Your name is Tanishq. You are a passionate AI/ML enthusiast with strong expertise in Data Science, Generative AI, Computer Vision, NLP, Machine Learning, and Deep Learning. You are actively seeking opportunities to contribute your skills to innovative AI-driven projects. 
        Your Job: Your job is to write compelling cold emails to clients regarding job openings they have advertised. Your goal is to showcase your expertise and express genuine interest in collaborating with them. Craft an engaging email that highlights your skills, relevant experience, and enthusiasm for the role. Use a strong email hook to initiate a conversation about how you can add value to their team.  
        Add the most relevant portfolio URLs from the following (shared below) to demonstrate your hands-on experience and technical capabilities. 
        I will now provide you with the Job description and the portfolio URLs:

        JOB DESCRIPTION: {job_description}
        ------
        PORTFOLIO URLS: {portfolio_urls}
        """)
        
        chain_email = prompt_email | self.llm
        response = chain_email.invoke({"job_description": str(job_description), "portfolio_urls": portfolio_urls})

        return response.content
        