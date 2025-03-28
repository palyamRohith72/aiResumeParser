import streamlit as st
from streamlit_option_menu import option_menu
from groq import Groq
import pdfplumber
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

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
if "interview_metrics" not in st.session_state:
    st.session_state["interview_metrics"] = {}
if "audio_responses" not in st.session_state:
    st.session_state["audio_responses"] = {}
if "text_responses" not in st.session_state:
    st.session_state["text_responses"] = {}
if "interview_questions" not in st.session_state:
    st.session_state["interview_questions"] = {}
    
for i in insights_string:
    if i not in st.session_state:
        st.session_state[i] = None

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

def generate_interview_questions(api_key, role, extracted_text, num_questions=10):
    """Generate a mix of interview questions based on role and resume"""
    prompt = f"""
    Generate {num_questions} interview questions for a {role} position based on the candidate's resume.
    Include a mix of:
    - 40% technical questions specific to {role}
    - 30% behavioral questions
    - 20% situational questions
    - 10% general HR questions
    
    The questions should be appropriate for the candidate's experience level.
    Here's the resume content:
    {extracted_text}
    
    Return the questions as a numbered list, nothing else.
    """
    
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a technical interviewer creating relevant questions."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )
    
    # Parse the response into individual questions
    questions = [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]
    return {f"Q{i+1}": q.split('. ')[1] if '. ' in q else q for i, q in enumerate(questions)}

def evaluate_answer(api_key, question, answer, role):
    """Evaluate the candidate's answer to an interview question"""
    prompt = f"""
    You are a technical interviewer evaluating a candidate's response for a {role} position.
    The question was: {question}
    The candidate's answer was: {answer}
    
    Evaluate the answer on the following criteria (0-100 scale):
    1. Technical accuracy (30 points)
    2. Clarity of explanation (20 points)
    3. Relevance to the role (20 points)
    4. Problem-solving approach (20 points)
    5. Communication skills (10 points)
    
    Provide:
    - A score out of 100
    - Justification for the score
    - What the ideal answer would include
    - Specific mistakes or areas for improvement
    
    Format your response as:
    Score: [score]/100
    Evaluation: [detailed evaluation]
    Ideal Answer: [what a strong answer would include]
    Improvements: [specific areas to improve]
    """
    
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a technical interviewer evaluating candidate responses."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )
    
    return response.choices[0].message.content

def analyze_resume_ats(api_key, extracted_text, role):
    """Analyze resume for ATS optimization"""
    prompt = f"""
    Analyze this resume for Applicant Tracking System (ATS) optimization for a {role} position.
    Provide specific recommendations to improve the resume's ATS score:
    
    1. Objective/Summary: What changes would make it more ATS-friendly?
    2. Skills Section: How to better align with {role} keywords?
    3. Work Experience: How to better quantify achievements and include relevant keywords?
    4. Education: Any improvements needed?
    5. Projects: How to make them more relevant to {role}?
    6. Formatting: ATS-friendly formatting suggestions
    7. Other sections: Any additions or deletions recommended?
    
    For each section, provide:
    - Current content (brief excerpt)
    - Issues identified
    - Specific recommended changes
    - Expected impact on ATS score
    
    Return the analysis in a structured, easy-to-follow format.
    """
    
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an ATS optimization expert analyzing resumes."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )
    
    return response.choices[0].message.content

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
        options=["How to use this app", "Primary Info", "Insights", "Take Interview", "Analyze Your Resume", "Clear Responses"],
        icons=["info", "list-task", "bar-chart", "mic", "file-earmark-text", "eraser"],
        menu_icon="cast",
        default_index=0
    )

# Show "How to use this app" by default when page loads
if selected_option == "How to use this app":
    st.video("https://youtu.be/kq2AANSBInE?si=-zv2jgYm_nhfzV3Q")

# Main Processing for other options (only if API key and PDF are provided)
if api_key and extracted_text:
    job_role = role  # Ensure job_role is set
        
    if selected_option == "Primary Info":
        if not st.session_state["run_query_once"]:
            st.session_state["response_primary"] = llm(api_key, query_string, "query_string", extracted_text, job_role, supporting_query)
            st.session_state["run_query_once"] = True  # Ensures it runs only once
        st.write(st.session_state["response_primary"])

    elif selected_option == "Insights":
        col1, col2 = st.columns([1, 2], border=True)

        with col1:
            selected_insight = st.radio("Select an insight to generate:", insights_string)
        
        with col2:
            if selected_insight:
                if st.session_state[selected_insight] is None:
                    response = llm(api_key, selected_insight, "insights_string", extracted_text, job_role, supporting_insights)
                    st.session_state[selected_insight] = response
                st.write(st.session_state[selected_insight])
    
    elif selected_option == "Take Interview":
        st.header("Mock Interview Session")
        
        # Button to regenerate questions and clear progress
        if st.button("Generate New Questions & Clear Progress"):
            st.session_state["interview_questions"] = {}
            st.session_state["interview_metrics"] = {}
            st.session_state["text_responses"] = {}
            st.rerun()
        
        # Initialize questions if not already present
        if not st.session_state["interview_questions"]:
            with st.spinner("Generating interview questions..."):
                st.session_state["interview_questions"] = generate_interview_questions(api_key, job_role, extracted_text)
        
        # Display questions in tabs
        tabs = st.tabs([f"Q{i+1}" for i in range(len(st.session_state["interview_questions"]))])
        
        for i, (q_key, question) in enumerate(st.session_state["interview_questions"].items()):
            with tabs[i]:
                st.subheader("Question:")
                st.write(question)
                
                # Text response area
                st.subheader("Your Answer:")
                text_response = st.text_area(
                    "Type your answer here:",
                    value=st.session_state["text_responses"].get(q_key, ""),
                    key=f"text_response_{q_key}"
                )
                
                # Save text response
                if text_response:
                    st.session_state["text_responses"][q_key] = text_response
                
                # Audio response (optional)
                st.subheader("Or record your answer:")
                audio_bytes = st.audio_input(
                    f"Record answer for {q_key}",
                    key=f"audio_{q_key}"
                )
                
                if audio_bytes:
                    st.session_state["audio_responses"][q_key] = audio_bytes
                    st.success("Audio response recorded!")
                
                # Evaluate button
                if st.button(f"Evaluate {q_key}", key=f"eval_{q_key}"):
                    if q_key in st.session_state["text_responses"] and st.session_state["text_responses"][q_key]:
                        evaluation = evaluate_answer(
                            api_key,
                            question,
                            st.session_state["text_responses"][q_key],
                            job_role
                        )
                        
                        # Extract score from evaluation
                        if "Score:" in evaluation:
                            score_line = evaluation.split("Score:")[1].split("\n")[0].strip()
                            score = int(score_line.split("/")[0])
                            st.session_state["interview_metrics"][q_key] = score
                        
                        st.subheader("Evaluation:")
                        st.write(evaluation)
                    else:
                        st.warning("Please provide a text answer before evaluating.")
        
        # Overall evaluation dashboard
        if st.button("Evaluate Total Performance"):
            if not st.session_state["interview_metrics"]:
                st.warning("No questions have been evaluated yet.")
            else:
                # Calculate metrics
                attempted = len(st.session_state["interview_metrics"])
                not_attempted = len(st.session_state["interview_questions"]) - attempted
                avg_score = sum(st.session_state["interview_metrics"].values()) / attempted if attempted else 0
                
                # Create dashboard
                st.header("Interview Performance Dashboard")
                
                # Pie chart for attempted vs not attempted
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Questions Attempted")
                    fig, ax = plt.subplots()
                    ax.pie([attempted, not_attempted], labels=["Attempted", "Not Attempted"], autopct='%1.1f%%')
                    st.pyplot(fig)
                
                # Bar chart for question scores
                with col2:
                    st.subheader("Question-wise Scores")
                    fig, ax = plt.subplots()
                    questions = list(st.session_state["interview_metrics"].keys())
                    scores = list(st.session_state["interview_metrics"].values())
                    ax.bar(questions, scores)
                    plt.xticks(rotation=45)
                    ax.set_ylim(0, 100)
                    ax.set_ylabel("Score")
                    st.pyplot(fig)
                
                # Overall metrics
                st.subheader("Overall Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Questions Attempted", f"{attempted}/{len(st.session_state['interview_questions'])}")
                with col2:
                    st.metric("Average Score", f"{avg_score:.1f}/100")
                with col3:
                    rating = avg_score / 10
                    st.metric("Overall Rating", f"{rating:.1f}/10")
                
                # Package estimation
                package_base = {
                    "Entry Level": "4-6 LPA",
                    "Mid Level": "8-12 LPA",
                    "Senior Level": "15-25 LPA"
                }
                level = "Entry Level" if avg_score < 60 else "Mid Level" if avg_score < 80 else "Senior Level"
                st.metric("Recommended Package", package_base[level])
                
                # Feedback
                st.subheader("Overall Feedback")
                if avg_score < 50:
                    st.error("Needs significant improvement. Focus on technical fundamentals and communication.")
                elif avg_score < 70:
                    st.warning("Good potential but needs improvement in some areas. Review technical concepts and practice more.")
                elif avg_score < 85:
                    st.success("Strong candidate. Minor improvements could make you exceptional.")
                else:
                    st.success("Outstanding performance! You're well-prepared for this role.")
    
    elif selected_option == "Analyze Your Resume":
        st.header("Resume ATS Optimization Analysis")
        
        if st.button("Analyze Resume for ATS Score Improvement"):
            with st.spinner("Analyzing your resume for ATS optimization..."):
                analysis = analyze_resume_ats(api_key, extracted_text, job_role)
                st.session_state["resume_analysis"] = analysis
            
        if "resume_analysis" in st.session_state:
            st.subheader("ATS Optimization Recommendations")
            st.write(st.session_state["resume_analysis"])
            
            st.subheader("Key Action Items")
            st.markdown("""
            - **Objective/Summary:** Tailor it specifically to the target role
            - **Skills Section:** Add relevant keywords from the job description
            - **Work Experience:** Quantify achievements with metrics
            - **Projects:** Highlight those most relevant to the target role
            - **Formatting:** Use standard headings and avoid graphics/tables
            """)
            
            st.info("Implementing these changes can significantly improve your resume's ATS score and visibility to recruiters.")
    
    elif selected_option == "Clear Responses":
        col1, col2 = st.columns([1, 2], border=True)
        radio_options = col1.radio("Select the stored response to delete data", st.session_state.keys())
        try:
            col2.write(st.session_state[radio_options])
            if col2.button("Delete Data", use_container_width=True, type='primary'):
                del st.session_state[radio_options]
                st.rerun()
        except Exception as e:
            col2.error(e)

# Warning message if API key or PDF is missing (but not for "How to use this app")
elif selected_option != "How to use this app":
    st.warning("Please enter an API key and upload a PDF to proceed.")
