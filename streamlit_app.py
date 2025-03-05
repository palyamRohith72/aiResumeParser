import streamlit as st
from streamlit_option_menu import option_menu
from groq import Groq
import pdfplumber  # Lightweight and fast PDF parser

# Insights options
insights_string = [
    "Is Skill Set Matches the current role - {role}",
    "Skills Possessing By The User",
    "Skills Missing By The User for This - {role}",
    "Projects Done By The User",
    "Projects Level - Beginner, Intermediate, Advanced",
    "Are Projects Related to This - {role}",
    "How Would You Rate User For This Role having skills, missing skills, done projects, projects level",
    "Would you suggest to hire or not"
]

# Initialize session state variables (ensures everything runs only once)
if "run_query_once" not in st.session_state:
    st.session_state["run_query_once"] = None
if "selected_insight" not in st.session_state:
    st.session_state["selected_insight"] = None
if "insight_responses" not in st.session_state:
    st.session_state["insight_responses"] = {insight: None for insight in insights_string}  # Store responses

def llm(api_key, string, string_type, extracted_text, job_role=None, supported_string=None):
    """
    Calls the Groq LLM API with the given string and string_type.
    """
    role = "You are a very good resume parser and good information retrieval system" if string_type == "query_string" \
        else "You are a good analyzer in analyzing candidates' profiles for the current job roles."
    
    client = Groq(api_key=api_key)

    user_content = f"{string} Use This Information {extracted_text} and {supported_string}"
    if job_role:
        user_content = f"{string} for job role: {job_role}, use information to extract: {extracted_text} and {supported_string}"

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": user_content}
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content

def parse_pdf(file):
    """
    Parses the uploaded PDF and extracts text content.
    """
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Define query strings
query_string = """You Have to Extract The Following Information From The Given String:
Name, Phone Number, Email, LinkedIn URL, GitHub URL, Portfolio Link, Other URLs, References with their phone numbers and emails, Work Experience, Designation, Education, Education Achievements, Other Achievements, Address"""

supporting_query = "Extracted Information Should be in the form of a Python dictionary, so it would be good to pass into Python's eval function and then pass it into the Streamlit write function or markdown function."

supporting_insights = "You Have To give the response in the form of a paragraph, listing detailed information for each content."

# Streamlit App Structure
st.sidebar.title("Settings")

# API Key Input
api_key = st.sidebar.text_input("Enter API Key", type="password")

# Role Input
role = st.sidebar.text_input("Enter Role")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Option Menu
selected_option = option_menu(
    menu_title="Navigation",
    options=["How to use this app", "Primary Info", "Insights"],
    icons=["info", "list-task", "bar-chart"],
    menu_icon="cast",
    default_index=0
)

# Process uploaded file
extracted_text = parse_pdf(uploaded_file) if uploaded_file else ""

# Processing Based on Selection
if api_key:
    job_role = role  # Ensuring job_role is assigned before calling llm()
    
    if selected_option == "Primary Info":
        if st.session_state["run_query_once"] is None:
            response = llm(api_key, query_string, "query_string", extracted_text, job_role, supporting_query)
            st.write(response)
            st.session_state["run_query_once"] = True  # Ensures it runs only once

    elif selected_option == "Insights":
        # Create two columns with ratio 1:2
        col1, col2 = st.columns([1, 2])

        # Column 1: Radio button for insights selection
        with col1:
            selected_insight = st.radio("Select an insight to generate:", insights_string)

        # Process insight selection
        if st.session_state["selected_insight"] != selected_insight:
            st.session_state["selected_insight"] = selected_insight  # Store selection
            st.session_state["insight_responses"][selected_insight] = None  # Reset stored response for new selection

        # Column 2: Display insight result
        with col2:
            if selected_insight:
                if st.session_state["insight_responses"][selected_insight] is None:
                    response = llm(api_key, selected_insight, "insights_string", extracted_text, job_role, supporting_insights)
                    st.session_state["insight_responses"][selected_insight] = response  # Store response

                # Display stored response
                st.write(st.session_state["insight_responses"][selected_insight])

else:
    st.warning("Please enter an API key to proceed.")
