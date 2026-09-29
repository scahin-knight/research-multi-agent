import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.tools.tool import web_search, scrape_url

load_dotenv()

# Model initialization using Google Gemini API
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",  # or "gemini-1.5-pro"
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ---------------------------------------------------------------------
# 1. Search Agent
# ---------------------------------------------------------------------
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="You are a research search agent. Find recent, reliable, and detailed sources on the topic."
    )

# ---------------------------------------------------------------------
# 2. Reader / Scraper Agent
# ---------------------------------------------------------------------
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="You are a web reading agent. Extract and summarize key content from relevant URLs."
    )

# ---------------------------------------------------------------------
# 3. Writer Chain (LCEL)
# ---------------------------------------------------------------------
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. You are a professional research writer. Write a DETAILED, LONG-FORM report "
     "(minimum 800-1200 words) based on the research provided. "
     "Structure it with: an introduction, multiple detailed body sections with headers, "
     "specific facts/data from the research, and a conclusion. "
     "Do not summarize briefly — expand on each point with context and explanation.."
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}
Research: {research}

Structure the report as:
1. Introduction
2. Key Findings (minimum 3 well-explained points)
3. Conclusion
4. Sources (list URLs found in the research)

Be detailed, factual, and professional."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------
# 4. Critic Chain (LCEL)
# ---------------------------------------------------------------------
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. Be honest and specific."
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this format:
- Score: X/10
- Strengths: (bullet points)
- Areas to Improve: (bullet points)
- One-line Verdict:"""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()