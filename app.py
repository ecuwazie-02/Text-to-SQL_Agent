import os
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page configuration and title
st.set_page_config(page_title=" PostegreSQL Text-to-SQL Agent", layout="wide" , initial_sidebar_state="expanded")

st.title(" Autonomous Text-to-SQL & Analytics Agent")
st.caption("Powered by PostgreSQL • Groq (Llama 3.3 70B) • LangChain • Streamlit")

# Load environment variables and Initialization 
load_dotenv()

@st.cache_resource
def init_db_and_llm():
    """ Initializes the database connection and LLM model."""
    DB_USER = os.getenv("pg_user")
    DB_HOST = os.getenv("pg_host")
    DB_PORT = os.getenv("pg_port")
    DB_PASSWORD = os.getenv("pg_password")

    DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set in the environment variables. Please set it to continue.")
        st.stop()

    db = SQLDatabase.from_uri(DB_URL, schema="public" , include_tables=['customer_segments' , 'transactions'])
    llm = ChatGroq(model_name= 'llama-3.3-70b-versatile', temperature= 0, groq_api_key= os.getenv("GROQ_API_KEY"))
    return db, llm
db, llm = init_db_and_llm()
schema_info = db.get_table_info()

# Sidebar Diagnostic Information
with st.sidebar:
    st.header("⚙️ Agent Diagnostics")
    st.success("🟢 Connected to PostgreSQL")
    st.info("**Active Database:** `postgres`")
    
    st.subheader("📋 Connected Views")
    for table in db.get_usable_table_names():
        st.markdown(f"- `{table}`")
        
    st.subheader("🧠 Model Config")
    st.markdown("- **Engine:** Groq LPUs")
    st.markdown("- **LLM:** `llama-3.3-70b-versatile`")
    st.markdown("- **Guardrails:** Read-Only (`SELECT` / `WITH`)")

    with st.expander("🔍 View Raw Schema Info"):
        st.code(schema_info, language="sql")

# Agent Prompt and Chain 

system_prompt = f"""
You are an expert PostgreSQL Data Analyst. 
Given a user question, write a syntactically correct PostgreSQL query using ONLY the views available below.

### DATABASE SCHEMA:
{schema_info}

### INSTRUCTIONS:
1. Return ONLY the raw SQL query. Do not wrap it in markdown block quotes or extra conversational text.
2. Use standard PostgreSQL syntax.
3. For text filters, use ILIKE for case-insensitive matching.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}\n\nPrevious error (if any): {error}")
])
summary_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a senior data analytics consultant. Summarize the dataset query result in 1-2 clear, professional sentences answering the original user question."),
    ("human", "Question: {question}\n\nQuery Executed:\n{sql}\n\nData Sample:\n{data_sample}")
])
chain = prompt_template | llm | StrOutputParser()
summary_chain = summary_prompt_template | llm | StrOutputParser()

#  Core Agent Functionality

def run_query_tool(query: str) -> str:
    """
    Function to run a SQL query using the database connection.
    """
    cleaned_query = query.strip().replace("```sql", "").replace("```", "")

    if not cleaned_query.upper().startswith("SELECT") and not cleaned_query.upper().startswith("WITH"):
        return "ERROR: Only read-only SELECT queries are allowed."

    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]
    if any(keyword in cleaned_query.upper() for keyword in forbidden_keywords):
        return 'ERROR: FORBIDDEN KEYWORD DETECTED. Only read-only SELECT queries are allowed.'

    try: 
        df = pd.read_sql(cleaned_query, db._engine)
        return True, df
    except Exception as e:
        return False, f'ERROR: An error occurred while executing the query: {str(e)}'

def process_agent_query(user_question: str, max_retries: int = 3):
    """ReAct loop: Generates SQL, executes, handles self-correction retries."""
    error_context = "None"
    
    status_box = st.status("🧠 Agent is processing query...", expanded=True)
    
    for attempt in range(1, max_retries + 1):
        status_box.write(f"🔄 **Attempt {attempt}:** Generating SQL query via Llama 3.3...")
        
        generated_sql = chain.invoke({"question": user_question, "error": error_context})
        cleaned_sql = generated_sql.strip().replace("```sql", "").replace("```", "")
        
        status_box.code(cleaned_sql, language="sql")
        status_box.write("⚙️ Running query against PostgreSQL database...")
        
        success, result = run_query_tool(cleaned_sql)
        
        if success:
            status_box.write("✅ **Execution successful!** Summarizing results...")
            
            # Generate executive summary
            data_sample = result.head(5).to_string()
            summary_text = summary_chain.invoke({
                "question": user_question,
                "sql": cleaned_sql,
                "data_sample": data_sample
            })
        
            status_box.update(label="🎉 Execution complete!", state="complete", expanded=False)
            return cleaned_sql, result, summary_text
        else:
            status_box.write(f"⚠️ **Error encountered:** {result}")
            status_box.write("🔧 Triggering self-correction loop...")
            error_context = result

    status_box.update(label="❌ Max retries reached. Query failed.", state="error")
    return None, None, "Failed to resolve query after self-correction attempts."

# Chat Interface & Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display sample query buttons for quick testing
st.markdown("### 💡 Quick Prompt Suggestions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Average Recency & Monetary Spend"):
        st.session_state.current_prompt = "What is the average recency and total monetary value across customer_rfm?"

with col2:
    if st.button("🌍 Top 5 Countries by Total Amount"):
        st.session_state.current_prompt = "Show me the top 5 countries by total transaction amount from stg_transactions."

with col3:
    if st.button("🏆 Highest Monetary Customer"):
        st.session_state.current_prompt = "Find the customer_id with the highest monetary spend in customer_rfm."

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sql" in message:
            st.code(message["sql"], language="sql")
        if "dataframe" in message and message["dataframe"] is not None:
            st.dataframe(message["dataframe"], use_container_width=True)

# User Input Handling
prompt = st.chat_input("Ask a question about customer_rfm or stg_transactions...")

# Handle sample button clicks
if "current_prompt" in st.session_state and st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    st.session_state.current_prompt = None

if prompt:
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run Agent
    with st.chat_message("assistant"):
        sql, df, summary = process_agent_query(prompt)

        # Escape the dollar signs so Streamlit doesn't render them as LaTeX
        safe_summary = summary.replace("$", r"\$")
        
        st.markdown(f"**Executive Summary:**\n{safe_summary}")
        
        if sql:
            st.markdown("**Executed SQL:**")
            st.code(sql, language="sql")
            
        if df is not None and not df.empty:
            st.markdown("**Query Results:**")
            st.dataframe(df, use_container_width=True)
            
            # Auto Visualization for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                st.markdown("---")
                st.markdown("📈 **Auto-Generated Visualization**")
                fig = px.bar(
                    df.head(10), 
                    x=categorical_cols[0], 
                    y=numeric_cols[0], 
                    title=f"{numeric_cols[0].replace('_', ' ').title()} by {categorical_cols[0].replace('_', ' ').title()}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Save assistant message to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Executive Summary:**\n{safe_summary}",
            "sql": sql,
            "dataframe": df
        })