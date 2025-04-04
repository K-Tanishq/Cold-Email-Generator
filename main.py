import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from chains import Chain
from database import Portfolio
from utils import clean_text

load_dotenv()

def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a URL:", value="https://jobs.micro1.ai/post/b5e66e27-6fe6-4fcc-b9bf-2ac2b5df5979?utm_source=Machine_Learning_&_Data_Science_-_Math_AI_trainer_unstop&utm_medium=listing&utm_campaign=unstop")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_email(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)

