import streamlit as st
import os
import io
import concurrent.futures
import json
import pandas as pd
import textwrap
from typing import Optional
from dotenv import load_dotenv

# File parsing
import fitz  # PyMuPDF
import docx

# Gemini SDK
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Intelligent Resume Shortlister",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
def local_css():
    st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Headings */
    h1, h2, h3, h4 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics / Cards */
    div[data-testid="stMetricValue"] {
        color: #2dd4bf;
    }
    
    /* Status Pills */
    .status-selected {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: bold;
        display: inline-block;
        font-size: 0.9em;
    }
    .status-rejected {
        background: linear-gradient(90deg, #ef4444, #dc2626);
        color: white;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: bold;
        display: inline-block;
        font-size: 0.9em;
    }
    
    /* Profile Box */
    .profile-box {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-top: 16px;
    }
    .profile-section-title {
        color: #cbd5e1;
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 12px;
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #020617;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #60a5fa, #3b82f6);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- Data Models (Structured Output) ---
class MandatorySkillEval(BaseModel):
    skill_name: str = Field(description="Name of the mandatory skill from the Minimum Requirements.")
    has_worked_on: bool = Field(description="True if the candidate has actually worked on this skill in a project.")
    years_of_experience: Optional[int] = Field(description="Estimated years of experience with this skill. Null if not specified.")

class OtherSkillEval(BaseModel):
    skill_name: str = Field(description="Name of another important skill from the Job Description.")
    proficiency: str = Field(description="Must be one of: 'Expert', 'Beginner', 'Just Mentioned', or 'Actual Project Experience'.")
    years_of_project_experience: Optional[int] = Field(description="Years of experience using this skill in actual projects. Null if not specified.")

class CandidateEvaluation(BaseModel):
    candidate_name: str = Field(description="The full name of the candidate found in the resume.")
    actual_work_summary: str = Field(description="A detailed 2-3 sentence summary of the specific tasks, accomplishments, and responsibilities the candidate actually worked on in their real projects or past roles.")
    mandatory_skills_evaluation: list[MandatorySkillEval] = Field(description="Evaluation of strictly the Minimum Requirements skills.")
    other_jd_skills_evaluation: list[OtherSkillEval] = Field(description="Evaluation of other important skills mentioned in the Job Description.")
    additional_good_points: list[str] = Field(description="List of any additional strong points or impressive achievements from the resume.")
    status: str = Field(description="Must be exactly 'Selected' or 'Rejected' based strictly on the Minimum Requirements.")
    reason: str = Field(description="A detailed rationale explaining exactly why the candidate was Selected or Rejected based on the Minimum Requirements.")
    match_percentage: int = Field(description="An estimated match percentage (0-100) based on how well their actual experience aligns with the Job Description and Requirements.")

# --- Helper Functions ---
def extract_text(file) -> str:
    """Extract text from uploaded PDF or DOCX file."""
    text = ""
    file_bytes = file.read()
    
    if file.name.lower().endswith('.pdf'):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text("text") + "\n"
        except Exception as e:
            text = f"Error reading PDF {file.name}: {e}"
            
    elif file.name.lower().endswith('.docx'):
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            text = f"Error reading DOCX {file.name}: {e}"
    else:
        # Fallback for txt
        try:
            text = file_bytes.decode('utf-8', errors='ignore')
        except:
             text = "Unsupported file format."
             
    # Reset pointer for Streamlit just in case
    file.seek(0)
    return text

def process_single_resume(client, resume_text: str, filename: str, jd_text: str, min_reqs: str) -> dict:
    """Calls Gemini to evaluate a single resume."""
    try:
        prompt = f"""
        You are an expert technical recruiter and hiring manager.
        Your task is to evaluate a candidate's resume against a Job Description and a set of Minimum Requirements.
        
        JOB DESCRIPTION:
        {jd_text}
        
        MINIMUM REQUIREMENTS (Candidate MUST meet these. If they fail even one, reject them):
        {min_reqs}
        
        CANDIDATE RESUME:
        {resume_text}
        
        Please deeply analyze the resume. Differentiate between skills just listed in a 'Skills' section versus skills actually used in 'Projects' or 'Work Experience'.
        1. For Mandatory Skills, verify if they actually worked on them and extract years of experience.
        2. For Other JD Skills, extract their proficiency (Expert, Beginner, Just Mentioned, Actual Project Experience) and years of project experience.
        3. Identify any additional good points from their resume.
        4. Make a strict Selection/Rejection based on Minimum Requirements.
        
        Fill out the required JSON schema accurately.
        """
        
        # Call Gemini 2.5 Flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CandidateEvaluation,
                temperature=0.1,
            ),
        )
        
        # Parse output
        result_dict = json.loads(response.text)
        result_dict['filename'] = filename
        return {"status": "success", "data": result_dict}
        
    except Exception as e:
        return {"status": "error", "filename": filename, "error": str(e)}

def process_all_resumes_concurrently(client, jd_text: str, min_reqs: str, resumes) -> list:
    """Process multiple resumes in parallel using ThreadPoolExecutor."""
    results = []
    
    # Pre-extract text synchronously to avoid passing BytesIO across threads
    resume_data = []
    for r in resumes:
        text = extract_text(r)
        resume_data.append((r.name, text))
        
    with st.spinner(f"Analyzing {len(resume_data)} resumes with Gemini 2.5 Flash..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_resume = {
                executor.submit(process_single_resume, client, text, name, jd_text, min_reqs): name
                for name, text in resume_data
            }
            
            # Progress bar setup
            progress_bar = st.progress(0)
            completed = 0
            
            for future in concurrent.futures.as_completed(future_to_resume):
                res = future.result()
                results.append(res)
                completed += 1
                progress_bar.progress(completed / len(resume_data))
                
    return results

# --- Main App ---
def main():
    st.title("⚡ AI Resume Shortlister")
    st.markdown("Automate candidate screening against your Job Description and Minimum Requirements using **Gemini 2.5 Flash**.")
    
    # Initialize Session State
    if 'evaluations' not in st.session_state:
        st.session_state['evaluations'] = []
    if 'selected_candidate' not in st.session_state:
        st.session_state['selected_candidate'] = None

    # --- Sidebar Settings ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key_input = st.text_input(
            "Google GenAI API Key", 
            value=os.getenv("GEMINI_API_KEY", ""), 
            type="password",
            help="Get it from Google AI Studio. Uses .env if available."
        )
        st.divider()
        
        st.header("📄 Inputs")
        
        jd_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=['pdf', 'docx', 'txt'])
        jd_text_manual = st.text_area("Or Paste Job Description Here")
        
        st.divider()
        min_reqs = st.text_area("Mandatory Minimum Requirements", height=150, 
            help="e.g., 'Must have 3+ years React experience. Must know AWS. Degree required.'")
            
        st.divider()
        resumes = st.file_uploader(
            "Upload Resumes (Max 20)", 
            type=['pdf', 'docx', 'txt'], 
            accept_multiple_files=True
        )

        process_btn = st.button("🚀 Process Resumes", use_container_width=True)

    # --- Actions ---
    if process_btn:
        if not api_key_input:
            st.error("Please provide a Gemini API Key.")
            return
        if not (jd_file or jd_text_manual):
            st.error("Please provide a Job Description.")
            return
        if not min_reqs:
            st.error("Please provide Minimum Requirements.")
            return
        if not resumes:
            st.error("Please upload at least one resume.")
            return
        if len(resumes) > 20:
            st.warning("You uploaded more than 20 resumes. Only processing the first 20.")
            resumes = resumes[:20]
            
        # Prepare inputs
        st.session_state['evaluations'] = []
        st.session_state['selected_candidate'] = None
        
        jd_text = extract_text(jd_file) if jd_file else jd_text_manual
        client = genai.Client(api_key=api_key_input)
        
        # Process
        results = process_all_resumes_concurrently(client, jd_text, min_reqs, resumes)
        
        # Filter successful results for state
        successful = [r['data'] for r in results if r['status'] == 'success']
        errors = [r for r in results if r['status'] == 'error']
        
        st.session_state['evaluations'] = sorted(
            successful, 
            key=lambda x: x.get('match_percentage', 0), 
            reverse=True
        )
        
        if errors:
            for e in errors:
                st.toast(f"Error processing {e['filename']}: {e['error']}", icon="❌")

    # --- UI Display ---
    evals = st.session_state.get('evaluations', [])
    
    if not evals:
        st.info("👈 Upload your files and click 'Process Resumes' to see results.")
        return
        
    tab1, tab2 = st.tabs(["📊 Summary Table", "👤 Detailed Profiles"])
    
    with tab1:
        st.subheader("All Candidates Results")
        
        df_data = []
        for ev in evals:
            mand_skills = "; ".join([f"{s.get('skill_name', '')} (Yrs: {s.get('years_of_experience') or 'N/A'}, Worked: {'Yes' if s.get('has_worked_on') else 'No'})" for s in ev.get('mandatory_skills_evaluation', [])])
            other_skills = "; ".join([f"{s.get('skill_name', '')} ({s.get('proficiency', '')}, Yrs: {s.get('years_of_project_experience') or 'N/A'})" for s in ev.get('other_jd_skills_evaluation', [])])
            good_points = "; ".join(ev.get('additional_good_points', []))
            
            df_data.append({
                "Candidate Name": ev.get('candidate_name', 'Unknown'),
                "Status": ev.get('status', 'Unknown'),
                "Match %": ev.get('match_percentage', 0),
                "Mandatory Skills Evaluation": mand_skills,
                "Other Skills Evaluation": other_skills,
                "Additional Good Points": good_points,
                "Reason": ev.get('reason', ''),
                "Filename": ev.get('filename', '')
            })
            
        df = pd.DataFrame(df_data)
        
        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV for HR",
            data=csv,
            file_name='resume_screening_results.csv',
            mime='text/csv',
        )
        
        # Table Display
        st.dataframe(df, use_container_width=True, height=600)
        
    with tab2:
        # Layout: Left column for summary list, Right column for detailed profile
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Candidates Summary")
            
            for idx, ev in enumerate(evals):
                # Create a card-like button for each candidate
                status_color = "#10b981" if ev['status'] == 'Selected' else "#ef4444"
                
                with st.container():
                    st.markdown(f"""
                    <div style="
                        border-left: 4px solid {status_color};
                        background-color: #1e293b;
                        padding: 12px;
                        border-radius: 0 8px 8px 0;
                        margin-bottom: 8px;
                        cursor: pointer;
                    ">
                        <h4 style="margin:0; font-size: 1.1em;">{ev['candidate_name']}</h4>
                        <p style="margin: 4px 0 0 0; font-size: 0.85em; color: #94a3b8;">Match: {ev['match_percentage']}% | {ev['filename']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Invisible native button overlaid to capture clicks
                    if st.button(f"View {ev['candidate_name']}", key=f"btn_{idx}", use_container_width=True):
                        st.session_state['selected_candidate'] = ev

        with col2:
            st.subheader("👤 Candidate Profile")
            selected = st.session_state.get('selected_candidate')
            
            if selected:
                # Render Profile
                status_class = "status-selected" if selected['status'] == 'Selected' else "status-rejected"
                
                profile_html = f"""<div class="profile-box">
<div style="display: flex; justify-content: space-between; align-items: start;">
<div>
<h2 style="margin: 0;">{selected['candidate_name']}</h2>
<span style="color: #64748b; font-size: 0.9em;">File: {selected['filename']}</span>
</div>
<div>
<span class="{status_class}">{selected['status']}</span>
<div style="margin-top: 8px; font-weight: bold; color: #38bdf8; text-align: right;">Match: {selected['match_percentage']}%</div>
</div>
</div>
<hr style="border-color: #334155; margin: 20px 0;">
<div class="profile-section-title">🎯 Decision Reasoning</div>
<p style="line-height: 1.6;">{selected['reason']}</p>
<div class="profile-section-title" style="margin-top: 24px;">💼 Actual Work & Projects Experience</div>
<p style="line-height: 1.6;">{selected['actual_work_summary']}</p>
<div class="profile-section-title" style="margin-top: 24px;">⭐ Additional Good Points</div>
<ul style="color: #cbd5e1; padding-left: 20px; line-height: 1.6;">
{"".join([f"<li>{p}</li>" for p in selected.get('additional_good_points', [])])}
</ul>
<div style="display: flex; gap: 24px; margin-top: 24px;">
<div style="flex: 1;">
<div class="profile-section-title">🚨 Mandatory Skills</div>
<ul style="color: #cbd5e1; padding-left: 20px; line-height: 1.6;">
{"".join([f"<li><b>{s.get('skill_name', '')}</b>: {'✅ Yes' if s.get('has_worked_on') else '❌ No'} (Exp: {s.get('years_of_experience') or 'N/A'} yrs)</li>" for s in selected.get('mandatory_skills_evaluation', [])])}
</ul>
</div>
<div style="flex: 1;">
<div class="profile-section-title">📊 Other JD Skills</div>
<ul style="color: #94a3b8; padding-left: 20px; line-height: 1.6;">
{"".join([f"<li><b>{s.get('skill_name', '')}</b>: {s.get('proficiency', '')} (Exp: {s.get('years_of_project_experience') or 'N/A'} yrs)</li>" for s in selected.get('other_jd_skills_evaluation', [])])}
</ul>
</div>
</div>
</div>"""
                st.markdown(profile_html, unsafe_allow_html=True)
            else:
                placeholder_html = """<div style="display: flex; height: 300px; align-items: center; justify-content: center; background-color: #1e293b; border-radius: 12px; border: 1px dashed #475569;">
<p style="color: #94a3b8;">Select a candidate from the summary list to view their detailed profile.</p>
</div>"""
                st.markdown(placeholder_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
