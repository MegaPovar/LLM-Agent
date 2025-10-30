import requests
OPENAI_API_KEY= "sk-S13ySvEOLGpHKOAGf44XZoLDEfeKkYFTWsiQWiEoDL3iPFoj"
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
from bs4 import BeautifulSoup
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Hello! Are you working?")

def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()
    return text_content_with_links

def fetch_web_page(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)

web_fetch_tool = Tool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetches the content of a web page"
)

prompt_template = "Check triviality for following, considering context of data: {content}"
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

summarize_tool = Tool.from_function(
    func=llm_chain.run,
    name="Summarizer",
    description="Summarizes a web page"
)

tools = [ddg_search, web_fetch_tool, summarize_tool]

agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)
con = input('Context')
v = input('correlations')
prompt = con + '. ' + v

print(agent.run(prompt))
