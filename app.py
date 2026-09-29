import streamlit as st
import os
import io
import concurrent.futures
import json
import pandas as pd
import textwrap
from typing import Optional
from dotenv import load_dotenv

import base64
import shutil

# File parsing & OCR
import fitz  # PyMuPDF
import docx
import pytesseract
from PIL import Image

# Gemini SDK
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# OpenAI / OpenRouter
from openai import OpenAI

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
# ATS metadata page signals — pages with these phrases are likely ATS/system pages, NOT resume content
ATS_PAGE_SIGNALS = [
    "Correspondence", "Jobs Applied", "Application Status Audit Trail",
    "Change History", "Hiring Manager Pending", "New Application",
    "Requisition Closed", "Not Suitable for Demand", "Interview 1 Reject",
    "Field Label", "Old Value", "New Value", "Changed By",
    "Initiator", "Type", "Subject", "Most Recent Message",
    "Req ID", "Job Title", "Recruiter", "AStatus", "OData", "API1 INT",
    "Applicant Profile", "Date/Time", "Source"
]

# Resume content signals — pages with these phrases are likely resume content
RESUME_PAGE_SIGNALS = [
    "PROFESSIONAL SUMMARY", "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE",
    "TECHNICAL SKILLS", "EDUCATION", "OBJECTIVE", "CERTIFICATIONS",
    "ACHIEVEMENTS", "PROJECTS", "KEY RESPONSIBILITIES", "ROLES & RESPONSIBILITIES",
    "EMPLOYMENT HISTORY", "ACADEMIC QUALIFICATION",
    "Data Engineer", "Data Warehousing", "ETL/ELT", "Apache Airflow",
    "Cloud Composer", "Dataflow", "BigQuery", "Dataproc",
]

def is_resume_page(page_text: str) -> bool:
    """
    Heuristic check: returns True if a page looks like resume content,
    False if it looks like ATS system metadata.

    Logic:
    - Count how many ATS signals appear on the page
    - Count how many resume signals appear on the page
    - If ATS signals dominate → skip the page
    - If resume signals dominate or tie → keep the page
    - A page with very little text (< 80 chars) is likely a divider/blank → skip
    """
    text_stripped = page_text.strip()
    if len(text_stripped) < 80:
        return False  # Blank/near-blank page

    text_upper = page_text.upper()

    ats_hits = sum(1 for signal in ATS_PAGE_SIGNALS if signal.upper() in text_upper)
    resume_hits = sum(1 for signal in RESUME_PAGE_SIGNALS if signal.upper() in text_upper)

    # Very high ATS signal count → definitely a system/metadata page
    if ats_hits >= 8 and resume_hits < 3:
        return False

    # Moderate ATS signals with weak resume signals → skip
    if ats_hits >= 3 and resume_hits < 2:
        return False

    # If both are weak (no clear signal) but the page has reasonable text → keep conservatively
    return True


def get_tesseract_cmd() -> Optional[str]:
    """Find tesseract executable on Windows or Unix PATH."""
    try:
        cmd = shutil.which("tesseract")
        if cmd:
            return cmd
    except Exception:
        pass

    if os.name == 'nt':
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p
    return None


def ocr_page(page, client=None, provider: str = None, model_name: str = None) -> str:
    """
    Renders a PyMuPDF page to an image and performs OCR using:
    1. PyTesseract (if tesseract binary is installed or on PATH)
    2. Multimodal LLM Vision fallback (Gemini or OpenAI/OpenRouter) if tesseract binary is unavailable
    """
    try:
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
    except Exception:
        return ""

    # Strategy 1: Fast Local OCR via PyTesseract
    try:
        t_cmd = get_tesseract_cmd()
        if t_cmd:
            pytesseract.pytesseract.tesseract_cmd = t_cmd
        img = Image.open(io.BytesIO(img_bytes))
        ocr_text = pytesseract.image_to_string(img)
        if ocr_text and len(ocr_text.strip()) > 30:
            return ocr_text.strip()
    except Exception:
        pass

    # Strategy 2: Multimodal LLM Vision Fallback (Zero external binary dependency)
    if client and provider:
        try:
            if provider == "Gemini":
                resp = client.models.generate_content(
                    model=model_name or "gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        "Extract and transcribe all text from this scanned resume page accurately and verbatim. Return only the extracted text without any commentary."
                    ],
                )
                if resp and resp.text and len(resp.text.strip()) > 20:
                    return resp.text.strip()
            elif provider in ["OpenAI", "OpenRouter"]:
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                resp = client.chat.completions.create(
                    model=model_name or "gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract and transcribe all text from this scanned resume page accurately and verbatim. Return only the extracted text without any commentary."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                        ]
                    }],
                    temperature=0.0
                )
                if resp and resp.choices and resp.choices[0].message.content:
                    return resp.choices[0].message.content.strip()
        except Exception:
            pass

    return ""


def extract_page_text_smart(page, client=None, provider: str = None, model_name: str = None) -> tuple[str, bool]:
    """
    Extracts text from a single PDF page.
    Only triggers OCR if the page has negligible native text (< 80 chars) AND has embedded images.
    Returns: (text, used_ocr)
    """
    native_text = page.get_text("text").strip()

    # Fast path: If page already has 80+ chars of selectable text, it's NOT a scanned image page!
    if len(native_text) >= 80:
        return native_text, False

    # Check if there are actual images on this page
    has_images = len(page.get_images()) > 0
    if not has_images:
        # Blank or separator page without images — no need to OCR
        return native_text, False

    # Scanned image page with < 80 characters of native text: run OCR
    ocr_text = ocr_page(page, client=client, provider=provider, model_name=model_name)
    if len(ocr_text) > len(native_text):
        return ocr_text, True

    return native_text, False


def extract_text(file_or_bytes, filename: str = "", client=None, provider: str = None, model_name: str = None) -> str:
    """Extract text from uploaded PDF or DOCX file (standard mode — with OCR support for scanned pages)."""
    text = ""
    if isinstance(file_or_bytes, bytes):
        file_bytes = file_or_bytes
        fname = filename or "document.pdf"
    else:
        file_bytes = file_or_bytes.read()
        fname = getattr(file_or_bytes, "name", filename or "document.pdf")
        file_or_bytes.seek(0)

    if fname.lower().endswith('.pdf'):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                p_text, _ = extract_page_text_smart(
                    page, client=client, provider=provider, model_name=model_name
                )
                text += p_text + "\n"
        except Exception as e:
            text = f"Error reading PDF {fname}: {e}"

    elif fname.lower().endswith('.docx'):
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            text = f"Error reading DOCX {fname}: {e}"
    else:
        # Fallback for txt
        try:
            text = file_bytes.decode('utf-8', errors='ignore')
        except:
            text = "Unsupported file format."

    return text


def extract_referral_resume_text(file_or_bytes, filename: str = "", client=None, provider: str = None, model_name: str = None) -> tuple[str, dict]:
    """
    Extract only the resume portion from a referral PDF.
    Returns: (resume_text, extraction_stats)
    
    Uses heuristic page filtering + smart OCR:
    - Reads each page individually
    - If a page is image-based / scanned, runs OCR to extract resume content
    - Evaluates whether the page is resume vs ATS metadata using is_resume_page()
    - Keeps only resume pages and skips ATS metadata pages
    """
    if isinstance(file_or_bytes, bytes):
        file_bytes = file_or_bytes
        fname = filename or "referral.pdf"
    else:
        file_bytes = file_or_bytes.read()
        fname = getattr(file_or_bytes, "name", filename or "referral.pdf")
        file_or_bytes.seek(0)

    stats = {
        "total_pages": 0,
        "resume_pages": 0,
        "skipped_pages": 0,
        "skipped_page_nums": [],
        "ocr_pages": 0,
    }

    if not fname.lower().endswith('.pdf'):
        # For non-PDF referrals (DOCX, TXT), fall back to standard extraction
        return extract_text(file_bytes, filename=fname, client=client, provider=provider, model_name=model_name), stats

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        stats["total_pages"] = len(doc)
        resume_text_parts = []

        for page_num, page in enumerate(doc, start=1):
            page_text, used_ocr = extract_page_text_smart(
                page, client=client, provider=provider, model_name=model_name
            )
            if used_ocr:
                stats["ocr_pages"] += 1

            if is_resume_page(page_text):
                resume_text_parts.append(page_text)
                stats["resume_pages"] += 1
            else:
                stats["skipped_pages"] += 1
                stats["skipped_page_nums"].append(page_num)

        resume_text = "\n".join(resume_text_parts)

        # Safety fallback: if we filtered out everything, return all text with OCR
        if not resume_text.strip():
            fallback_parts = []
            for page in doc:
                p_text, _ = extract_page_text_smart(page, client=client, provider=provider, model_name=model_name)
                fallback_parts.append(p_text)
            resume_text = "\n".join(fallback_parts)
            stats["skipped_pages"] = 0
            stats["resume_pages"] = stats["total_pages"]
            stats["skipped_page_nums"] = []

        return resume_text, stats

    except Exception as e:
        return f"Error reading referral PDF {fname}: {e}", stats




def clean_resume_with_llm(client, provider: str, model_name: str, raw_text: str) -> str:
    """
    LLM-based cleaning pass: asks the model to extract ONLY the resume content
    from a referral document, stripping all ATS metadata, job application history,
    correspondence logs, and change history.
    
    This is the second-pass cleanup after heuristic page filtering.
    """
    cleanup_prompt = f"""You are a document parser. The following text was extracted from a referral PDF document.
This document may contain a mix of:
1. The candidate's actual RESUME (professional summary, work experience, skills, education)
2. ATS/HR system metadata (job application history, correspondence, audit trails, change logs, email subjects, recruiter names, application statuses)

Your task: Extract and return ONLY the candidate's resume content.
- KEEP: Professional Summary, Work Experience, Technical Skills, Education, Certifications, Projects, Achievements
- REMOVE: Any job application statuses, recruitment correspondence, audit trails, change history tables, email threads, recruiter names/actions, ATS system fields (Field Label, Old Value, New Value, Changed By, Source, OData, etc.)
- REMOVE: Any prior rejection/acceptance statuses from OTHER job applications

Return ONLY the clean resume text. Do not add any commentary or explanation.

RAW DOCUMENT TEXT:
{raw_text}
"""
    try:
        if provider == "Gemini":
            response = client.models.generate_content(
                model=model_name,
                contents=cleanup_prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return response.text.strip()
        elif provider in ["OpenAI", "OpenRouter"]:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": cleanup_prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        # If LLM cleaning fails, return original text
        return raw_text

def process_single_resume(client, provider: str, model_name: str, resume_text: str, filename: str,
                           jd_text: str, min_reqs: str, source: str = "Standard") -> dict:
    """Calls the selected LLM provider to evaluate a single resume."""
    try:
        # Source-aware context injection for referral PDFs
        referral_context = ""
        if source == "Referral":
            referral_context = """
        ⚠️ IMPORTANT — REFERRAL DOCUMENT CONTEXT:
        This resume was extracted from a REFERRAL PDF document. The original document may have contained
        ATS/HR system pages such as job application history, correspondence logs, audit trails, and
        change history from an internal HR system. Those have been filtered out.
        
        CRITICAL RULES FOR REFERRAL RESUMES:
        - Base your evaluation SOLELY on the candidate's technical skills, projects, and work experience shown below.
        - Do NOT penalize the candidate for any prior rejections, application statuses, or screening outcomes
          visible in the document — those are from OTHER job applications and are IRRELEVANT.
        - Do NOT let phrases like "Not Suitable for Demand", "Interview 1 Reject", "Requisition Closed" 
          influence your assessment in any way.
        - Evaluate this as a FRESH, DIRECT application.
        """

        prompt = f"""
        You are an expert technical recruiter and senior hiring manager evaluating candidates for a technical role.
        Your task is to evaluate a candidate's resume against a Job Description and Minimum Requirements.
        {referral_context}
        JOB DESCRIPTION:
        {jd_text}

        MINIMUM REQUIREMENTS (Candidate MUST meet these. Reject only if they clearly fail):
        {min_reqs}

        CANDIDATE RESUME:
        {resume_text}

        EVALUATION INSTRUCTIONS:
        1. Deeply analyze the resume content. Differentiate between skills just listed in a 'Skills' section 
           versus skills ACTUALLY USED in 'Projects' or 'Work Experience' descriptions.
        2. For Mandatory Skills: verify if they actually worked on them and estimate years of experience 
           from work history dates and project descriptions.
        3. For Other JD Skills: extract proficiency level (Expert, Beginner, Just Mentioned, or Actual Project Experience) 
           and years of project experience.
        4. Identify additional strong points or achievements from the resume.
        5. Match Percentage: Estimate based ONLY on the candidate's actual skills and project experience 
           vs the JD requirements. Ignore any HR metadata or application history.
        6. Selection/Rejection: Make a fair decision based strictly on the Minimum Requirements.
           - Count a skill as "met" if the candidate has used it in actual work experience, even if years 
             are slightly below requirement (note the gap but don't auto-reject on 1 skill shortfall if 
             overall profile is strong).

        Fill out the required JSON schema accurately.
        """

        if provider == "Gemini":
            # Call Gemini
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CandidateEvaluation,
                    temperature=0.1,
                ),
            )
            result_dict = json.loads(response.text)
        elif provider in ["OpenAI", "OpenRouter"]:
            # Call OpenAI / OpenRouter
            schema_dict = CandidateEvaluation.model_json_schema()
            full_prompt = f"""
            {prompt}

            Return the output strictly as a JSON object matching this schema:
            {json.dumps(schema_dict, indent=2)}
            """

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            result_dict = json.loads(response.choices[0].message.content)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        result_dict['filename'] = filename
        result_dict['source'] = source
        return {"status": "success", "data": result_dict}

    except Exception as e:
        return {"status": "error", "filename": filename, "error": str(e)}


def process_single_candidate_pipeline(
    filename: str,
    file_bytes: bytes,
    client,
    provider: str,
    model_name: str,
    jd_text: str,
    min_reqs: str,
    source: str = "Standard",
    use_llm_cleaning: bool = False
) -> dict:
    """Processes a single candidate file end-to-end: extraction (with OCR), optional cleaning, and evaluation."""
    try:
        # Step 1: Extraction
        if source == "Referral":
            resume_text, stats = extract_referral_resume_text(
                file_bytes, filename=filename, client=client, provider=provider, model_name=model_name
            )
            if use_llm_cleaning and resume_text.strip():
                resume_text = clean_resume_with_llm(client, provider, model_name, resume_text)
        else:
            resume_text = extract_text(
                file_bytes, filename=filename, client=client, provider=provider, model_name=model_name
            )
            stats = {}

        # Step 2: Evaluation
        eval_result = process_single_resume(
            client, provider, model_name, resume_text, filename, jd_text, min_reqs, source=source
        )
        if eval_result.get("status") == "success" and stats:
            eval_result["data"]["_extraction_stats"] = stats

        return eval_result
    except Exception as e:
        return {"status": "error", "filename": filename, "error": str(e)}


def process_all_resumes_concurrently(client, provider: str, model_name: str, jd_text: str, min_reqs: str,
                                      resumes, source: str = "Standard",
                                      use_llm_cleaning: bool = False) -> list:
    """
    Process multiple resumes in parallel using ThreadPoolExecutor.
    Displays live progress bar and status updates from the very start.
    """
    total_count = len(resumes)
    results = []

    # Read uploaded file bytes in memory immediately (< 0.05s)
    file_payloads = []
    for r in resumes:
        b = r.read()
        r.seek(0)
        file_payloads.append((r.name, b))

    # Initialize live UI progress indicators IMMEDIATELY
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    progress_bar = progress_placeholder.progress(0.0)
    status_placeholder.info(f"⚡ Analyzing {total_count} candidate(s) in parallel with {provider} ({model_name})...")

    # Run extraction, OCR, and evaluation concurrently in parallel threads
    max_workers = min(total_count, 6)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(
                process_single_candidate_pipeline,
                name, b, client, provider, model_name, jd_text, min_reqs, source, use_llm_cleaning
            ): name
            for name, b in file_payloads
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_name):
            res = future.result()
            results.append(res)
            completed += 1
            progress_bar.progress(completed / total_count)

            # Live feedback as each candidate finishes
            if res.get("status") == "success":
                c_name = res["data"].get("candidate_name", future_to_name[future])
                m_pct = res["data"].get("match_percentage", 0)
                status_placeholder.markdown(
                    f"🔄 **[{completed}/{total_count}]** Completed: **{c_name}** (`{m_pct}% Match`)"
                )
            else:
                status_placeholder.markdown(
                    f"⚠️ **[{completed}/{total_count}]** Completed with error: `{future_to_name[future]}`"
                )

    progress_bar.progress(1.0)
    status_placeholder.success(f"✅ Successfully evaluated all {total_count} candidate(s)!")
    return results



def get_client(provider: str, api_key: str):
    """Initializes and returns the appropriate client based on the provider and API key."""
    if provider == "Gemini":
        # 1. If key is explicitly provided:
        if api_key:
            return genai.Client(api_key=api_key)
            
        # 2. Check for Service Account in Streamlit Secrets
        if "gcp_service_account" in st.secrets:
            import tempfile
            import json
            secret_data = st.secrets["gcp_service_account"]
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                if isinstance(secret_data, str):
                    temp_file.write(secret_data)
                else:
                    json.dump(dict(secret_data), temp_file)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name
            
            # Load project ID from secrets if available
            project_id = "akansha-academy-001"
            try:
                if isinstance(secret_data, str):
                    parsed = json.loads(secret_data)
                else:
                    parsed = dict(secret_data)
                project_id = parsed.get("project_id", project_id)
            except:
                pass
                
            return genai.Client(vertexai=True, project=project_id, location="us-central1")
            
        # 3. Check for local gcp-key.json file
        elif os.path.exists("gcp-key.json"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("gcp-key.json")
            project_id = "akansha-academy-001"
            try:
                with open("gcp-key.json", "r") as f:
                    parsed = json.load(f)
                    project_id = parsed.get("project_id", project_id)
            except:
                pass
            return genai.Client(vertexai=True, project=project_id, location="us-central1")
            
        # 4. Check for standard env variable or secrets
        api_key_env = os.getenv("GEMINI_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
        if api_key_env:
            return genai.Client(api_key=api_key_env)
            
    elif provider == "OpenAI":
        actual_key = api_key or os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
        if actual_key:
            return OpenAI(api_key=actual_key)
            
    elif provider == "OpenRouter":
        actual_key = api_key or os.getenv("OPENROUTER_API_KEY", "") or st.secrets.get("OPENROUTER_API_KEY", "")
        if actual_key:
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=actual_key,
                default_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Intelligent Resume Shortlister",
                }
            )
            
    return None


PROFILES_FILE = "profiles.json"

def load_profiles() -> dict:
    """Loads saved profiles from profiles.json."""
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            pass
    return {"default_profile": "None", "profiles": {}}

def save_profile(label: str, jd_text: str, min_reqs: str, is_default: bool):
    """Saves a profile and optionally sets it as default."""
    data = load_profiles()
    if "profiles" not in data:
        data["profiles"] = {}
    data["profiles"][label] = {
        "jd_text": jd_text,
        "min_reqs": min_reqs
    }
    if is_default:
        data["default_profile"] = label
    elif data.get("default_profile") == label and not is_default:
        data["default_profile"] = "None"
        
    with open(PROFILES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    st.session_state['profiles'] = data
    st.session_state['selected_profile'] = label

def delete_profile(label: str):
    """Deletes a saved profile."""
    data = load_profiles()
    if "profiles" in data and label in data["profiles"]:
        del data["profiles"][label]
    if data.get("default_profile") == label:
        data["default_profile"] = "None"
        
    with open(PROFILES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    st.session_state['profiles'] = data
    st.session_state['selected_profile'] = "None"

def on_profile_change():
    """Callback when selected profile changes to update input fields."""
    prof_name = st.session_state.get('selected_profile_name')
    profiles = st.session_state.get('profiles', {}).get('profiles', {})
    if prof_name in profiles:
        st.session_state['jd_text_manual'] = profiles[prof_name]['jd_text']
        st.session_state['min_reqs'] = profiles[prof_name]['min_reqs']
        st.session_state['selected_profile'] = prof_name
    else:
        st.session_state['jd_text_manual'] = ""
        st.session_state['min_reqs'] = ""
        st.session_state['selected_profile'] = "None"


# --- Main App ---
def main():
    st.title("⚡ AI Resume Shortlister")
    st.markdown("Automate candidate screening against your Job Description and Minimum Requirements using AI.")
    
    # Initialize Session State
    if 'evaluations' not in st.session_state:
        st.session_state['evaluations'] = []
    if 'selected_candidate' not in st.session_state:
        st.session_state['selected_candidate'] = None
    if 'profiles' not in st.session_state:
        st.session_state['profiles'] = load_profiles()
    if 'selected_profile' not in st.session_state:
        st.session_state['selected_profile'] = st.session_state['profiles'].get('default_profile', 'None')
        
    # Pre-populate fields on first load if a default profile is set
    profiles_data = st.session_state['profiles'].get('profiles', {})
    default_prof = st.session_state['profiles'].get('default_profile', 'None')
    if default_prof in profiles_data:
        if 'jd_text_manual' not in st.session_state:
            st.session_state['jd_text_manual'] = profiles_data[default_prof]['jd_text']
        if 'min_reqs' not in st.session_state:
            st.session_state['min_reqs'] = profiles_data[default_prof]['min_reqs']


    # --- Sidebar Settings ---
    with st.sidebar:
        st.header("⚙️ LLM Configuration")
        provider = st.selectbox(
            "LLM Provider",
            options=["Gemini", "OpenAI", "OpenRouter"],
            index=0,
            key="llm_provider"
        )
        
        if provider == "Gemini":
            default_model = "gemini-2.5-flash"
            env_key = os.getenv("GEMINI_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
            api_key = st.text_input("Gemini API Key", value=env_key, type="password", help="Leave blank to use environment variable.")
            model_name = st.text_input("Model Name", value=default_model)
        elif provider == "OpenAI":
            default_model = "gpt-4o-mini"
            env_key = os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
            api_key = st.text_input("OpenAI API Key", value=env_key, type="password", help="Leave blank to use environment variable.")
            model_name = st.text_input("Model Name", value=default_model)
        elif provider == "OpenRouter":
            default_model = "google/gemini-2.5-flash"
            env_key = os.getenv("OPENROUTER_API_KEY", "") or st.secrets.get("OPENROUTER_API_KEY", "")
            api_key = st.text_input("OpenRouter API Key", value=env_key, type="password", help="Leave blank to use environment variable.")
            model_name = st.text_input("Model Name", value=default_model)
            
        st.divider()
        
        # --- Job Profiles ---
        st.header("💼 Job Profiles")
        profiles_dict = st.session_state.get('profiles', {}).get('profiles', {})
        profile_names = ["None / Custom"] + list(profiles_dict.keys())
        
        selected_prof_idx = 0
        current_sel = st.session_state.get('selected_profile', 'None')
        if current_sel in profile_names:
            selected_prof_idx = profile_names.index(current_sel)
            
        selected_prof = st.selectbox(
            "Load Saved Profile",
            options=profile_names,
            index=selected_prof_idx,
            key="selected_profile_name",
            on_change=on_profile_change
        )
        
        # Profile Management Expander
        with st.expander("💾 Manage Profiles"):
            default_label_val = selected_prof if selected_prof != "None / Custom" else ""
            new_label = st.text_input("Profile Label", value=default_label_val, placeholder="e.g. React Developer")
            
            default_profile_name = st.session_state.get('profiles', {}).get('default_profile')
            is_default = st.checkbox(
                "Set as Default Profile", 
                value=(default_profile_name == selected_prof) if selected_prof != "None / Custom" else False
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Profile", use_container_width=True):
                    jd_content = st.session_state.get('jd_text_manual', '').strip()
                    reqs_content = st.session_state.get('min_reqs', '').strip()
                    
                    if not new_label.strip():
                        st.error("Please enter a profile label.")
                    elif not jd_content:
                        st.error("Please enter a Job Description.")
                    elif not reqs_content:
                        st.error("Please enter Minimum Requirements.")
                    else:
                        save_profile(new_label.strip(), jd_content, reqs_content, is_default)
                        st.success(f"Profile '{new_label.strip()}' saved!")
                        st.rerun()
            with col2:
                if selected_prof != "None / Custom":
                    if st.button("Delete Profile", use_container_width=True):
                        delete_profile(selected_prof)
                        st.warning(f"Profile '{selected_prof}' deleted.")
                        st.rerun()

        st.divider()
        st.header("📄 Inputs")

        jd_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=['pdf', 'docx', 'txt'])
        if jd_file:
            extracted_jd_text = extract_text(jd_file)
            st.session_state['jd_text_manual'] = extracted_jd_text

        jd_text_manual = st.text_area("Or Paste Job Description Here", key="jd_text_manual")

        st.divider()
        min_reqs = st.text_area(
            "Mandatory Minimum Requirements",
            height=150,
            help="e.g., 'Must have 3+ years React experience. Must know AWS. Degree required.'",
            key="min_reqs"
        )

        st.divider()

        # --- Upload Source Selector ---
        st.subheader("📂 Upload Source")
        upload_source = st.radio(
            "Select upload source type",
            options=["Standard Resumes", "Referral PDFs"],
            index=0,
            key="upload_source",
            help=(
                "Standard Resumes: normal PDF/DOCX resume files.\n\n"
                "Referral PDFs: multi-page documents from your ATS that contain the resume "
                "alongside correspondence logs, job application history, and audit trail pages."
            )
        )

        is_referral = upload_source == "Referral PDFs"

        if is_referral:
            st.info(
                "🔗 **Referral Mode Active**\n\n"
                "Each uploaded PDF will be scanned page-by-page. ATS metadata pages "
                "(Correspondence, Jobs Applied, Audit Trail, Change History) will be automatically "
                "filtered out before evaluation.",
                icon="ℹ️"
            )
            use_llm_cleaning = st.checkbox(
                "🧹 Also use LLM-based cleaning (more accurate, uses 1 extra API call per file)",
                value=False,
                help="After heuristic page filtering, a second LLM pass further strips any remaining ATS noise from the extracted text."
            )
        else:
            use_llm_cleaning = False

        resumes = st.file_uploader(
            f"Upload {'Referral PDFs' if is_referral else 'Resumes'} (Max 20)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )

        process_btn = st.button("🚀 Process Resumes", use_container_width=True)


    # --- Actions ---
    if process_btn:
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

        # Initialize client using our multi-provider get_client
        client = get_client(provider, api_key)
        if not client:
            st.error(f"No authentication credentials found for {provider}. Please provide an API Key in the settings or environment variables.")
            return

        # Prepare inputs
        st.session_state['evaluations'] = []
        st.session_state['selected_candidate'] = None

        jd_text = extract_text(jd_file, client=client, provider=provider, model_name=model_name) if jd_file else jd_text_manual

        # Determine source string
        source = "Referral" if is_referral else "Standard"

        # Show referral mode banner
        if is_referral:
            st.info(f"🔗 **Referral Mode**: Processing {len(resumes)} referral PDF(s). Filtering ATS metadata pages and OCR-extracting scanned resume pages automatically.")

        # Process
        results = process_all_resumes_concurrently(
            client, provider, model_name, jd_text, min_reqs, resumes,
            source=source, use_llm_cleaning=use_llm_cleaning
        )

        # Filter successful results for state
        successful = [r['data'] for r in results if r['status'] == 'success']
        errors = [r for r in results if r['status'] == 'error']

        st.session_state['evaluations'] = sorted(
            successful,
            key=lambda x: x.get('match_percentage', 0),
            reverse=True
        )

        # Show referral extraction stats
        if is_referral:
            for r in results:
                if r['status'] == 'success':
                    stats = r['data'].get('_extraction_stats', {})
                    if stats.get('total_pages', 0) > 0:
                        skipped = stats.get('skipped_pages', 0)
                        total = stats.get('total_pages', 0)
                        kept = stats.get('resume_pages', 0)
                        ocr_count = stats.get('ocr_pages', 0)
                        skipped_nums = stats.get('skipped_page_nums', [])
                        fname = r['data'].get('filename', '')
                        ocr_msg = f" ({ocr_count} scanned pages OCR'd)" if ocr_count > 0 else ""
                        st.toast(
                            f"📄 {fname}: Kept {kept}/{total} pages{ocr_msg}. Filtered ATS pages: {skipped_nums}",
                            icon="🔗"
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
            source_label = "🔗 Referral" if ev.get('source') == 'Referral' else "📄 Standard"

            df_data.append({
                "Candidate Name": ev.get('candidate_name', 'Unknown'),
                "Source": source_label,
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
                source_badge = "🔗 Referral" if ev.get('source') == 'Referral' else "📄 Standard"

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
                        <p style="margin: 4px 0 0 0; font-size: 0.85em; color: #94a3b8;">Match: {ev['match_percentage']}% | {source_badge}</p>
                        <p style="margin: 2px 0 0 0; font-size: 0.75em; color: #64748b;">{ev['filename']}</p>
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
                is_sel_referral = selected.get('source') == 'Referral'

                # Source badge
                if is_sel_referral:
                    source_html = '<span style="background:#0ea5e9;color:white;padding:3px 10px;border-radius:999px;font-size:0.8em;margin-left:8px;">🔗 Referral</span>'
                else:
                    source_html = '<span style="background:#475569;color:white;padding:3px 10px;border-radius:999px;font-size:0.8em;margin-left:8px;">📄 Standard</span>'

                # Referral extraction stats banner
                stats = selected.get('_extraction_stats', {})
                stats_html = ""
                if is_sel_referral and stats.get('total_pages', 0) > 0:
                    kept = stats.get('resume_pages', 0)
                    total = stats.get('total_pages', 0)
                    ocr_count = stats.get('ocr_pages', 0)
                    skipped_nums = stats.get('skipped_page_nums', [])
                    ocr_note = f" (<b>{ocr_count}</b> scanned page(s) OCR'd)" if ocr_count > 0 else ""
                    stats_html = (
                        f'<div style="background:#1e3a4c;border:1px solid #0ea5e9;border-radius:8px;'
                        f'padding:10px 14px;margin-bottom:16px;font-size:0.85em;color:#7dd3fc;">'
                        f'🔍 Referral extraction: <b>{kept}/{total} pages</b> kept as resume content{ocr_note}. '
                        f'Skipped ATS pages: {skipped_nums if skipped_nums else "none"}.</div>'
                    )


                mand_skills_html = "".join([
                    f"<li><b>{s.get('skill_name', '')}</b>: "
                    f"{'✅ Yes' if s.get('has_worked_on') else '❌ No'} "
                    f"(Exp: {s.get('years_of_experience') or 'N/A'} yrs)</li>"
                    for s in selected.get('mandatory_skills_evaluation', [])
                ])
                other_skills_html = "".join([
                    f"<li><b>{s.get('skill_name', '')}</b>: {s.get('proficiency', '')} "
                    f"(Exp: {s.get('years_of_project_experience') or 'N/A'} yrs)</li>"
                    for s in selected.get('other_jd_skills_evaluation', [])
                ])
                good_points_html = "".join([f"<li>{p}</li>" for p in selected.get('additional_good_points', [])])

                profile_html = f"""<div class="profile-box">
<div style="display: flex; justify-content: space-between; align-items: start;">
<div>
<h2 style="margin: 0;">{selected['candidate_name']}{source_html}</h2>
<span style="color: #64748b; font-size: 0.9em;">File: {selected['filename']}</span>
</div>
<div>
<span class="{status_class}">{selected['status']}</span>
<div style="margin-top: 8px; font-weight: bold; color: #38bdf8; text-align: right;">Match: {selected['match_percentage']}%</div>
</div>
</div>
<hr style="border-color: #334155; margin: 20px 0;">
{stats_html}
<div class="profile-section-title">🎯 Decision Reasoning</div>
<p style="line-height: 1.6;">{selected['reason']}</p>
<div class="profile-section-title" style="margin-top: 24px;">💼 Actual Work &amp; Projects Experience</div>
<p style="line-height: 1.6;">{selected['actual_work_summary']}</p>
<div class="profile-section-title" style="margin-top: 24px;">⭐ Additional Good Points</div>
<ul style="color: #cbd5e1; padding-left: 20px; line-height: 1.6;">{good_points_html}</ul>
<div style="display: flex; gap: 24px; margin-top: 24px;">
<div style="flex: 1;">
<div class="profile-section-title">🚨 Mandatory Skills</div>
<ul style="color: #cbd5e1; padding-left: 20px; line-height: 1.6;">{mand_skills_html}</ul>
</div>
<div style="flex: 1;">
<div class="profile-section-title">📊 Other JD Skills</div>
<ul style="color: #94a3b8; padding-left: 20px; line-height: 1.6;">{other_skills_html}</ul>
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
