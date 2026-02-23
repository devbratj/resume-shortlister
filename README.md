# AI Resume Shortlister

An intelligent Streamlit application that evaluates up to 20 candidate resumes against a Job Description and Minimum Requirements using Google's Gemini 2.5 Flash API.

## Features
- **Concurrent Processing**: Evaluates up to 20 resumes (PDF, DOCX, TXT) simultaneously for speed.
- **Detailed Skills Extraction**: 
  - Checks if candidates meet Mandatory Minimum Requirements (with years of experience).
  - Evaluates proficiency in other Job Description skills (Expert, Beginner, etc.).
  - Extracts additional positive points from the resume.
- **Summary Dashboard**: View all candidates side-by-side with a calculated match percentage.
- **CSV Export**: Instantly download HR-ready evaluation data.
- **Candidate Profiles**: Click on any candidate to read detailed reasoning on why they were Selected or Rejected.

---

## Quick Start (Local Setup)

Follow these instructions to move and run this project on any other local system.

### Prerequisites
1. **Python 3.10+** installed on the system.
2. A **Google Gemini API Key**. (Get one from [Google AI Studio](https://aistudio.google.com/)).

### Step-by-Step Installation

1. **Copy the Project Folder**: Move the entire `resume-shortlister` folder to the new system (via Git, USB, or ZIP file).

2. **Open a Terminal / Command Prompt** inside the `resume-shortlister` directory.

3. **(Optional but Recommended) Create a Virtual Environment**:
   It's best practice to keep Python dependencies isolated.
   ```bash
   python -m venv venv
   
   # Activate on Windows:
   venv\Scripts\activate
   
   # Activate on Mac/Linux:
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up your API Key**:
   - Rename `.env.example` to `.env` (or create a new `.env` file).
   - Add your API key inside:
     ```env
     GEMINI_API_KEY=your_api_key_here
     ```
   *(Alternatively, you can just paste the API key directly into the Streamlit UI sidebar when the app runs).*

6. **Run the Application**:
   ```bash
   python -m streamlit run app.py
   ```

7. **Open in Browser**: The terminal will display a local URL (usually `http://localhost:8501` or `8503`). Open that link in your browser to start using the app!

---

## Run Locally with Docker

If you prefer to run the application securely within a Docker container (and avoid installing Python dependencies directly on the host machine):

1. **Ensure Docker is Installed**: Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine) is installed and running on your system.

2. **Build the Docker Image**:
   Open a terminal in the `resume-shortlister` folder and run:
   ```bash
   docker build -t resume-shortlister .
   ```

3. **Run the Container**:
   Start the container and map port 8080 to your local machine:
   ```bash
   docker run -p 8080:8080 resume-shortlister
   ```

4. **Open in Browser**: Navigate to `http://localhost:8080` to access the Streamlit app.

---

## Deployment (Google Cloud Run)

If you wish to deploy this app to the cloud so anyone can access it:

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).
2. Authenticate locally:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
3. Build and Deploy using the included Dockerfile:
   ```bash
   # Build the container image
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/resume-shortlister
   
   # Deploy to Cloud Run
   gcloud run deploy resume-shortlister --image gcr.io/YOUR_PROJECT_ID/resume-shortlister --platform managed --allow-unauthenticated
   ```
