import streamlit as st
from streamlit_option_menu import option_menu
from groq import Groq
import pdfplumber  # Lightweight and fast PDF parser

# Streamlit Sidebar Settings
st.sidebar.title("Settings")

# API Key Input
api_key = st.sidebar.text_input("Enter API Key", type="password")

# Role Input
role = st.sidebar.text_input("Enter Role")

# Insights options
insights_string = [
    f"Does the skill set match the current role - {role}?",
    "Skills possessed by the user",
    f"Skills missing for this role - {role}",
    "Projects completed by the user",
    "Project levels - Beginner, Intermediate, Advanced",
    f"Are projects related to this role - {role}?",
    "Overall candidate rating based on skills, missing skills, completed projects, and project level",
    "Would you suggest hiring this candidate?"
]

# Initialize session state variables
if "run_query_once" not in st.session_state:
    st.session_state["run_query_once"] = False
if "selected_insight" not in st.session_state:
    st.session_state["selected_insight"] = None
for i in insights_string:
    if i not in st.session_state:
        st.session_state[i]=None

def llm(api_key, query, query_type, extracted_text, job_role=None, support_text=None):
    """
    Calls the Groq LLM API with the given query and type.
    """
    role_description = (
        "You are an advanced resume parser and an expert in extracting structured information." if query_type == "query_string" 
        else "You are an expert in analyzing candidates' profiles, summarizing, and reporting insights for HR decisions."
    )
    
    client = Groq(api_key=api_key)
    user_prompt = f"Just Tell Me {query}\nUsing this information: {extracted_text}\nAdditional context: {support_text}"
    if job_role:
        user_prompt += f"\nJob Role: {job_role}"
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": role_description},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b-versatile",
    )
    
    return response.choices[0].message.content


def parse_pdf(file):
    """
    Extracts text content from the uploaded PDF.
    """
    text = ""
    if file:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    return text


# Queries
query_string = (
    "Extract the following details in a structured format: "
    "Name, Phone Number, Email, LinkedIn, GitHub, Portfolio, Other URLs, "
    "References (Name, Phone, Email), Work Experience, Designation, Education, "
    "Educational Achievements, Other Achievements, Address."
)

supporting_query = "Ensure the extracted information is well-structured and complete. Output should be clear and formatted."

supporting_insights = (
    f"### Output Format Guidelines:\n"
    "**Section One:** Only essential information realetd to the main content(Query You Are Performing). No paragraphs.\n"
    "**Section Two:** Insights with structured headings and bullet points. where insights should be drawn according to the main content\n"
    "Ensure output is clean, properly structured,short,report-oriented and professional."
)


# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF Resume", type=["pdf"])
extracted_text = parse_pdf(uploaded_file) if uploaded_file else ""

# Sidebar Navigation
with st.sidebar:
    selected_option = option_menu(
        menu_title="Navigation",
        options=["How to use this app", "Primary Info", "Insights","Clear Responses"],
        icons=["info", "list-task", "bar-chart","eraser"],
        menu_icon="cast",
        default_index=0
    )

# Main Processing
if api_key and extracted_text:
    job_role = role  # Ensure job_role is set
    if selected_option=="How to use this app":
        st.video("https://youtu.be/kq2AANSBInE?si=-zv2jgYm_nhfzV3Q")
    elif selected_option == "Primary Info":
        if not st.session_state["run_query_once"]:
            st.session_state["response_primary"] = llm(api_key, query_string, "query_string", extracted_text, job_role, supporting_query)
            st.session_state["run_query_once"] = True  # Ensures it runs only once
        st.write(st.session_state["response_primary"])

    elif selected_option == "Insights":
        col1, col2 = st.columns([1, 2],border=True)

        with col1:
            selected_insight = st.radio("Select an insight to generate:", insights_string)
        
        with col2:
            if selected_insight:
                if st.session_state[selected_insight] is None:
                    response = llm(api_key, selected_insight, "insights_string", extracted_text, job_role, supporting_insights)
                    st.session_state[selected_insight] = response
                st.write(st.session_state[selected_insight])
    elif selected_option=="Clear Responses":
        col1,col2=st.columns([1,2],border=True)
        radio_options=col1.radio("Select the stored response to delete data",st.session_state.keys())
        try:
            col2.write(st.session_state[radio_options])
            if col2.button("Delete Data",use_container_width=True,type='primary'):
                del st.session_state[radio_options]
                st.rerun()
        except Exception as e:
            col2.error(e)
                
else:
    st.warning("Please enter an API key and upload a PDF to proceed.")
    
