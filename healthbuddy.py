"""
HealthBuddy - AI Healthcare Assistant
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.retrievers import ArxivRetriever

# LangGraph imports for React agent
try:
    from langgraph.prebuilt import create_react_agent
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph React agent available!")
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph not available, using simplified approach")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Global variables (like a simple database)
doctors_database = [
    {"name": "Dr. Janet Dyne", "specialization": "Endocrinology (Diabetes Care)", "available_timings": "10:00 AM - 1:00 PM", "location": "City Health Clinic", "contact": "janet.dyne@healthclinic.com"},
    {"name": "Dr. Don Blake", "specialization": "Cardiology (Heart Specialist)", "available_timings": "2:00 PM - 5:00 PM", "location": "Metro Cardiac Center", "contact": "don.blake@metrocardiac.com"},
    {"name": "Dr. Susan D'Souza", "specialization": "Oncology (Cancer Care)", "available_timings": "11:00 AM - 2:00 PM", "location": "Hope Cancer Institute", "contact": "susan.dsouza@hopecancer.org"},
    {"name": "Dr. Matt Murdock", "specialization": "Psychiatry (Mental Health)", "available_timings": "4:00 PM - 7:00 PM", "location": "Mind Care Center", "contact": "matt.murdock@mindcare.com"},
    {"name": "Dr. Dinah Lance", "specialization": "General Physician", "available_timings": "9:00 AM - 12:00 PM", "location": "Downtown Medical Center", "contact": "dinah.lance@downtownmed.com"}
]

# Global variables for the AI
ai_model = None
web_search_tool = None
research_tool = None
health_agent = None


def ensure_healthbuddy_setup() -> bool:
    """Ensure HealthBuddy is initialized before use."""
    if ai_model and web_search_tool and research_tool and health_agent:
        return True
    return setup_healthbuddy()


def _needs_doctor_recommendation(question: str) -> bool:
    doctor_keywords = [
        "doctor",
        "specialist",
        "physician",
        "consult",
        "consultation",
        "appointment",
        "cardiologist",
        "neurologist",
        "oncologist",
        "dermatologist",
        "psychiatrist",
        "pediatrician",
        "endocrinologist",
        "gastroenterologist",
        "surgeon",
        "orthopedist",
        "dentist"
    ]
    text = question.lower()
    return any(keyword in text for keyword in doctor_keywords)


def _needs_research_lookup(question: str) -> bool:
    research_keywords = [
        "research",
        "study",
        "studies",
        "paper",
        "papers",
        "arxiv",
        "clinical trial",
        "evidence",
        "meta-analysis",
        "systematic review"
    ]
    text = question.lower()
    return any(keyword in text for keyword in research_keywords)


def setup_healthbuddy():
    """
    Setup HealthBuddy using Streamlit secrets
    This is like turning on the robot!
    """
    global ai_model, web_search_tool, research_tool, health_agent
    
    # Get API keys from Streamlit secrets
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        tavily_api_key = st.secrets["TAVILY_API_KEY"]
    except KeyError as e:
        st.error(f"❌ Missing API key in secrets: {e}")
        st.info("Please add your API keys to `.streamlit/secrets.toml`")
        return False
    
    if openai_api_key == "your_openai_api_key_here" or tavily_api_key == "your_tavily_api_key_here":
        st.error("❌ Please update your API keys in `.streamlit/secrets.toml`")
        return False
    
    print("🏥 Setting up HealthBuddy...")
    
    # Create the AI brain
    ai_model = ChatOpenAI(
        model_name="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.1
    )
    
    # Create the web search tool
    web_search_tool = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
    
    # Create the research tool
    research_tool = ArxivRetriever(
        top_k_results=3,
        get_full_documents=True,
        doc_content_chars_max=20000
    )
    
    # Create the tools that HealthBuddy can use - EXACT SAME as original
    tools = [create_web_search_tool(), create_research_tool(), create_doctor_tool()]
    
    # EXACT SAME agent prompt from original working file
    AGENT_PROMPT_TXT = r"""You are an agent designed to act as an expert in researching on medical symptoms
and also recommend relevant doctors for booking appointments.
Also remember the current year is 2025 and use the same for all search queries when no specific dates are mentioned.

Given an input user query call relevant tools and give the most appropriate response.
Follow some of these guidelines to help you make more informed decisions:
  - If the user's query specifies recommending a doctor only then recommend an appropriate doctor
  - If the user is researching on detailed and specific aspects around symptoms, treatments and other aspects related to healthcare
  use both search_web and search_arxiv tools to get comprehensive information and then give a well-structured response
  - If the user is just looking for general information around healthcare then web search is enough
  - Use search_arxiv tool only if the query is related to information which might be found in research papers
  - Response should include cited source links and \ or arXiv Article Title, Publication Dates if available
  - If recommending doctors then use the recommend_doctor and show detailed information in a nice structured way and recommend them to book an appointment via email
  - Politely decline answering any queries not related to medical or healthcare information
"""
    
    AGENT_SYS_PROMPT = SystemMessage(content=AGENT_PROMPT_TXT)
    
    # Try to create React agent with LangGraph - EXACT SAME as original
    if LANGGRAPH_AVAILABLE:
        try:
            print("🤖 Creating React agent with LangGraph...")
            health_agent = create_react_agent(model=ai_model, tools=tools, prompt=AGENT_SYS_PROMPT)
            print("✅ React agent created successfully!")
        except Exception as e:
            print(f"⚠️ React agent failed, using simplified approach: {e}")
            health_agent = ai_model
    else:
        print("⚠️ Using simplified approach (no React agent)")
        health_agent = ai_model
    
    print("✅ HealthBuddy is ready to help!")
    return True

def create_web_search_tool():
    """
    Create a tool that can search the web - EXACT SAME as original working file
    """
    @tool
    def search_web(query: str) -> list:
        """
        Search the web for general or up-to-date information on healthcare topics.

        Inputs:
        - query (str): The search query string. Should describe the healthcare topic or information you want to find.

        Outputs:
        - list: A list of up to 3 formatted strings, each containing:
            - Title of the search result
            - Content extracted from the page
            - Source URL
        """
        print(f"🔍 Searching the web for: {query}")
        
        try:
            results = web_search_tool.raw_results(
                query=query, 
                max_results=3, 
                search_depth='advanced',
                include_answer=False, 
                include_raw_content=True
            )
            
            docs = results['results']
            docs = [doc for doc in docs if doc.get("raw_content")]

            # Limit raw_content to first 500 characters (enough for context)
            docs = [
                "## Title\n" + doc['title'] +
                "\n\n## Content (truncated)\n" + doc['raw_content'][:500] + "..." +
                "\n\n## Source\n" + doc['url']
                for doc in docs
            ]

            print(f"✅ Found {len(docs)} web results")
            return docs
            
        except Exception as e:
            print(f"❌ Web search failed: {e}")
            return [f"Error searching the web: {str(e)}"]

    return search_web

def create_research_tool():
    """
    Create a tool that can search research papers - EXACT SAME as original working file
    """
    @tool
    def search_arxiv(query: str) -> list:
        """
        Search arXiv for relevant scientific research papers and articles.

        Inputs:
        - query (str): The research topic or keywords to search for on arXiv.

        Outputs:
        - list: A list of up to 3 formatted strings, each containing:
            - Title of the paper
            - Summary of the paper
            - Full content (truncated to maximum allowed characters)
          Returns ["No articles found for the given query."] if no matches are found.
        """
        print(f"📚 Searching arXiv for: {query}")
        
        try:
            results = research_tool.invoke(query)
            if results:
                articles = ['## Title\n'+doc.metadata['Title']+'\n\n'+'## Summary\n'+doc.metadata['Summary']+'\n\n'+'##Content\n'+doc.page_content for doc in results]
                print(f"✅ Found {len(articles)} research papers")
                return articles
            else:
                print("⚠️ No research papers found")
                return ["No articles found for the given query."]
        except Exception as e:
            print(f"❌ Research search failed: {e}")
            return [f"Error fetching arXiv articles: {str(e)}"]

    return search_arxiv

def create_doctor_tool():
    """
    Create a tool that can recommend doctors - EXACT SAME as original working file
    """
    @tool
    def recommend_doctor(query: str) -> dict:
        """
        Recommend the most suitable doctor based on the user's symptoms or health-related query.

        Inputs:
        - query (str): A description of the patient's symptoms or healthcare needs.

        Outputs:
        - dict: A dictionary containing:
            - "recommended_doctor": JSON-formatted details of the selected doctor from the `doctors_database`.
              If the most suitable match cannot be determined, defaults to recommending the General Physician.
        """
        print(f"👨‍⚕️ Finding a doctor for: {query}")
        
        try:
            doctors_list = str(doctors_database)
            prompt = f"""You are an assistant helping recommend a doctor based on patient's health issues.

                         Here is the list of available doctors:
                        {doctors_list}

                        Given the user's query: "{query}"

                        Choose the most suitable doctor from the list. Only pick one doctor.
                        Return only the selected doctor's information in JSON format (no markdown).
                        If not sure, recommend the General Physician.
                      """

            response = ai_model.invoke(prompt)
            print(f"✅ Doctor recommendation generated")
            return {"recommended_doctor": response.content}
            
        except Exception as e:
            print(f"❌ Doctor recommendation failed: {e}")
            return {"recommended_doctor": f"Error recommending doctor: {str(e)}"}

    return recommend_doctor

def ask_healthbuddy(question):
    """
    Ask HealthBuddy a question and get an answer - EXACT SAME as original working file
    """
    # Auto-setup if not already done
    if health_agent is None:
        print("🔧 Auto-setting up HealthBuddy...")
        if not setup_healthbuddy():
            return "❌ HealthBuddy setup failed. Please check your API keys in `.streamlit/secrets.toml`"
    
    print(f"🤖 HealthBuddy is thinking about: {question}")
    
    try:
        # Check if we have a React agent (EXACT SAME as original)
        if hasattr(health_agent, 'stream') and LANGGRAPH_AVAILABLE:
            print("🔄 Using React agent with streaming...")
            return call_agent_with_streaming(question)
        else:
            print("⚠️ Using simplified approach...")
            answer, _ = ask_with_simple_approach(question)
            return answer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Sorry, I encountered an error: {e}"

def call_agent_with_streaming(query):
    """
    Call the React agent with streaming - EXACT SAME as original working file
    """
    print("🚀 React Agent: Starting with streaming...")
    
    try:
        # Stream the agent's execution with the given query (EXACT SAME as original)
        final_event = None
        for event in health_agent.stream(
            {"messages": [HumanMessage(content=query)]}, # input prompt
            stream_mode='values'  # Stream output as intermediate values
        ):
            # Print each intermediate message step-by-step
            print(f"🔄 Agent step: {event['messages'][-1]}")
            final_event = event
        
        # Get the final response
        if not final_event:
            raise RuntimeError("React agent stream returned no events")

        final_response = final_event["messages"][-1].content
        print("✅ React Agent: Final answer generated!")
        return final_response
        
    except Exception as e:
        print(f"❌ React agent streaming error: {e}")
        answer, _ = ask_with_simple_approach(query)
        return answer

def ask_with_react_agent(question):
    """
    Use React agent with visual tool calling feedback
    """
    print("🚀 React Agent: Starting tool selection process...")
    
    try:
        # Use the React agent
        response = health_agent.invoke({"messages": [HumanMessage(content=question)]})
        
        # Extract the final answer
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            # Get the last AI message
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    print("✅ React Agent: Final answer generated!")
                    return msg.content
        
        # Fallback
        return str(response)
        
    except Exception as e:
        print(f"❌ React agent error: {e}")
        answer, _ = ask_with_simple_approach(question)
        return answer

def ask_with_simple_approach(question):
    """
    Simple React-like approach with tool calling and structured logging.
    Returns (answer, logs) so both console and UI can stay in sync.
    """
    print("🔧 Simple approach: Let AI decide...")

    if not ensure_healthbuddy_setup():
        error_message = "❌ HealthBuddy setup failed. Please check your API keys in `.streamlit/secrets.toml`"
        return error_message, {
            "reasoning": error_message,
            "tools_selected": [],
            "execution_log": []
        }

    if ai_model is None:
        error_message = "❌ AI model is not initialized"
        return error_message, {
            "reasoning": error_message,
            "tools_selected": [],
            "execution_log": []
        }

    # Create tools for manual calling with visual feedback
    tools = {
        "search_web": create_web_search_tool(),
        "search_arxiv": create_research_tool(),
        "recommend_doctor": create_doctor_tool()
    }

    # Enhanced system prompt
    system_prompt = """You are HealthBuddy, an AI healthcare assistant.

Rules:
- You MUST call at least one tool before answering a question. Never answer directly without tool use.
- If the query is about general health info (symptoms, causes, treatments, advice), use TOOL: search_web.
- If the query mentions research, studies, or papers, use TOOL: search_arxiv.
- If the query asks about a doctor, specialist, or consultation, you MUST use TOOL: recommend_doctor.
- If the query mixes research + doctor, you MUST call both tools step by step.
- NEVER invent or fabricate doctors. Only use doctors from the database provided by the `recommend_doctor` tool.

IMPORTANT: When you need to use tools, respond with:
- "TOOL: search_web" for web search
- "TOOL: search_arxiv" for research papers
- "TOOL: recommend_doctor" for doctor recommendations

There is NO "TOOL: none" option. Always select at least one tool.
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    response = ai_model.invoke(messages)
    ai_response = getattr(response, "content", response)

    # --- Logging structure ---
    logs = {
        "reasoning": ai_response,          # raw AI thought process
        "tools_selected": [],              # which tools AI decided
        "execution_log": [],               # step by step execution trace
    }

    print(f"🤖 AI Response: {ai_response}")

    # --- Determine tool plan (AI suggestion + rule-based enforcement) ---
    tool_plan = []
    if "TOOL: search_web" in ai_response:
        tool_plan.append("search_web")
    if "TOOL: search_arxiv" in ai_response:
        tool_plan.append("search_arxiv")
    if "TOOL: recommend_doctor" in ai_response:
        tool_plan.append("recommend_doctor")

    if _needs_research_lookup(question) and "search_arxiv" not in tool_plan:
        logs["execution_log"].append("ℹ️ Added search_arxiv because the question references research/studies.")
        tool_plan.append("search_arxiv")

    if _needs_doctor_recommendation(question) and "recommend_doctor" not in tool_plan:
        logs["execution_log"].append("ℹ️ Added recommend_doctor because the question requests a doctor.")
        tool_plan.append("recommend_doctor")

    if not tool_plan:
        logs["execution_log"].append("ℹ️ No tool selected by AI; defaulting to search_web.")
        tool_plan.append("search_web")

    # --- Run tools in plan order (deduplicated) ---
    tool_outputs = {}
    executed_tools = set()

    for tool_name in tool_plan:
        if tool_name in executed_tools:
            continue
        executed_tools.add(tool_name)
        logs["tools_selected"].append(tool_name)

        if tool_name == "search_web":
            print("🔍 Executing: search_web")
            web_results = tools["search_web"].invoke({"query": question})
            logs["execution_log"].append(
                f"search_web returned {len(web_results) if isinstance(web_results, list) else 'N/A'} results"
            )
            tool_outputs["web"] = web_results

        elif tool_name == "search_arxiv":
            print("📚 Executing: search_arxiv")
            research_results = tools["search_arxiv"].invoke({"query": question})
            logs["execution_log"].append(
                f"search_arxiv returned {len(research_results) if isinstance(research_results, list) else 'N/A'} papers"
            )
            tool_outputs["arxiv"] = research_results

        elif tool_name == "recommend_doctor":
            print("👨‍⚕️ Executing: recommend_doctor")
            doctor_results = tools["recommend_doctor"].invoke({"query": question})
            logs["execution_log"].append(f"recommend_doctor returned: {doctor_results}")
            tool_outputs["doctor"] = doctor_results

    # --- Final synthesis step ---
    if tool_outputs:
        final_messages = [
            SystemMessage(content="You are a helpful healthcare assistant. Combine all tool results into a single comprehensive answer."),
            HumanMessage(content=f"Question: {question}\n\nTool Results: {tool_outputs}\n\nWrite a clear, structured answer. Always include doctor info if available.")
        ]
        final_response = ai_model.invoke(final_messages)
        answer = getattr(final_response, "content", final_response)
    else:
        logs["execution_log"].append("⚠️ AI didn't specify tools clearly — falling back to direct response")
        answer = ai_response

    print("✅ HealthBuddy has an answer!")
    return answer, logs


def get_api_keys_status():
    """
    Check if API keys are properly configured
    """
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
        tavily_key = st.secrets.get("TAVILY_API_KEY", "")
        
        if not openai_key or openai_key == "your_openai_api_key_here":
            return False, "OpenAI API key not configured"
        
        if not tavily_key or tavily_key == "your_tavily_api_key_here":
            return False, "Tavily API key not configured"
        
        return True, "API keys configured"
        
    except Exception as e:
        return False, f"Error checking API keys: {e}"

def get_all_doctors():
    """
    Get a list of all available doctors
    """
    return doctors_database

def add_new_doctor(name, specialization, timings, location, contact):
    """
    Add a new doctor to the database
    """
    global doctors_database
    
    new_doctor = {
        "name": name,
        "specialization": specialization,
        "available_timings": timings,
        "location": location,
        "contact": contact
    }
    
    doctors_database.append(new_doctor)
    print(f"✅ Added new doctor: {name}")

def get_healthbuddy_status():
    """
    Check if HealthBuddy is ready
    """
    if health_agent is None:
        return "❌ HealthBuddy is not set up"
    else:
        return "✅ HealthBuddy is ready to help!"

def show_react_agent_workflow():
    """
    Show how the React agent works with LangGraph
    """
    workflow_diagram = """
    🤖 **React Agent Workflow with LangGraph**
    
    ```
    User Question → React Agent → Tool Selection → Tool Execution → Final Answer
         ↓              ↓              ↓              ↓              ↓
    "Chest pain?"   "I need to    "Let me use    "Searching     "Here's Dr. Don
                   recommend a    recommend_     doctor         Blake (Cardiologist)"
                   doctor"        doctor tool    database
    ```
    
    **Step-by-Step Process:**
    
    1. **🤔 Reasoning**: Agent thinks about what tools to use
    2. **🔧 Tool Selection**: Chooses appropriate tool (web search, research, doctor)
    3. **⚡ Tool Execution**: Runs the selected tool
    4. **🔄 Iteration**: May use multiple tools if needed
    5. **📝 Final Answer**: Combines results into comprehensive response
    
    **Available Tools:**
    - 🔍 `search_web`: Search current health information
    - 📚 `search_research_papers`: Find scientific studies
    - 👨‍⚕️ `recommend_doctor`: Get doctor recommendations
    
    **React Agent Benefits:**
    - ✅ **Automatic tool selection** - No manual intervention needed
    - ✅ **Multi-tool usage** - Can combine multiple tools for complex queries
    - ✅ **Intelligent reasoning** - Decides which tools are most relevant
    - ✅ **Visual feedback** - Shows exactly which tools are being called
    """
    return workflow_diagram

# Example usage function
def run_example():
    """
    Run a simple example to show how HealthBuddy works
    """
    print("🏥 HealthBuddy Example")
    print("=" * 30)
    
    # Check if HealthBuddy is set up
    if health_agent is None:
        print("❌ Please set up HealthBuddy first!")
        print("Run: setup_healthbuddy('your_openai_key', 'your_tavily_key')")
        return
    
    # Ask some example questions
    questions = [
        "What are the symptoms of diabetes?",
        "I have chest pain, can you recommend a doctor?",
        "What does research say about exercise and mental health?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        answer = ask_healthbuddy(question)
        print(f"🤖 Answer: {answer[:200]}...")
        print("-" * 50)
