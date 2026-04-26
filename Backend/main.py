# =============================================================================
# main.py — FASTAPI SERVER
# =============================================================================
# This is the FastAPI server — it defines all the API endpoints
# that the React frontend communicates with
# When React sends a user profile, this file receives it,
# calls predict.py to run the model, and sends the result back
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from predict import predict

# =============================================================================
# SECTION 1 — CREATE THE FASTAPI APP
# =============================================================================
# FastAPI() creates the application instance
# title and description show up in the auto-generated API docs at /docs

app = FastAPI(
    title="Student Admission Calculator API",
    description="Predicts graduate school admission probability using XGBoost models trained on 185,000+ applicant records",
    version="1.0.0"
)

# =============================================================================
# SECTION 2 — CORS MIDDLEWARE
# =============================================================================
# CORS (Cross-Origin Resource Sharing) is a browser security feature
# By default browsers block requests from one origin (React on localhost:3000)
# to a different origin (FastAPI on localhost:8000)
# We need to explicitly allow this so React can talk to our API
# In production you would replace "*" with your actual frontend URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow requests from any origin (React dev server)
    allow_credentials=True,
    allow_methods=["*"],        # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],        # allow all headers
)

# =============================================================================
# SECTION 3 — REQUEST BODY SCHEMA
# =============================================================================
# Pydantic BaseModel defines the exact shape of data we expect from React
# FastAPI uses this to automatically validate incoming requests
# If React sends wrong data types or missing required fields, FastAPI
# automatically returns a clear error message without us writing any validation code
#
# Field(...) means required — no default value
# Field(0) means optional with a default of 0
# ge = greater than or equal to, le = less than or equal to (validation constraints)

class ApplicantProfile(BaseModel):

    # Degree type — determines which model to use
    degree_type: str = Field(..., description="masters or phd")

    # University being applied to
    applied_university: str = Field(..., description="Name of the university")
    university_region: str = Field(..., description="e.g. USA, Canada, Europe, Australia")
    university_country: str = Field(..., description="e.g. United States, Canada, United Kingdom")
    university_qs_tier: str = Field(..., description="1, 2, or 3 based on QS ranking tier")

    # Program details
    program_name: str = Field(..., description="e.g. MS Computer Science")
    program_field: str = Field(..., description="STEM, BIZ, or ARTS")

    # Academic scores
    undergrad_gpa: float = Field(..., ge=2.0, le=4.0, description="GPA on a 4.0 scale")
    gre_total: float = Field(..., ge=260, le=340, description="Total GRE score")
    gre_verbal: float = Field(..., ge=130, le=170, description="GRE Verbal score")
    gre_quantitative: float = Field(..., ge=130, le=170, description="GRE Quantitative score")
    gre_analytical_writing: float = Field(..., ge=0, le=6, description="GRE Analytical Writing score")

    # Optional test scores — default to 0 if not submitted
    gmat_total: Optional[float] = Field(0, ge=0, le=800, description="GMAT total score (optional)")
    gmat_verbal: Optional[float] = Field(0, description="GMAT verbal score (optional)")
    gmat_quant: Optional[float] = Field(0, description="GMAT quant score (optional)")
    toefl_score: Optional[float] = Field(0, ge=0, le=120, description="TOEFL score (optional)")
    ielts_score: Optional[float] = Field(0, ge=0, le=9, description="IELTS score (optional)")

    # Application strength
    sop_strength: float = Field(..., ge=1, le=5, description="Statement of Purpose strength (1-5)")
    sop_word_count: Optional[float] = Field(500, description="SOP word count")
    lor_count: float = Field(..., ge=0, le=10, description="Number of recommendation letters")
    lor_avg_strength: float = Field(..., ge=1, le=5, description="Average LOR strength (1-5)")
    lor_from_professor: Optional[float] = Field(0, description="Number of LORs from professors")
    lor_from_industry: Optional[float] = Field(0, description="Number of LORs from industry")

    # Experience
    research_experience_years: float = Field(..., ge=0, description="Years of research experience")
    publications_count: Optional[float] = Field(0, ge=0, description="Number of publications")
    conference_papers: Optional[float] = Field(0, ge=0, description="Number of conference papers")
    thesis_completed: Optional[int] = Field(0, description="1 if thesis completed, 0 if not")
    work_experience_years: float = Field(..., ge=0, description="Years of work experience")
    internships_count: Optional[float] = Field(0, ge=0, description="Number of internships")
    work_industry_relevance: Optional[float] = Field(3, ge=1, le=5, description="How relevant work experience is (1-5)")

    # Other
    is_international: int = Field(..., description="1 if international student, 0 if domestic")
    funding_type: str = Field(..., description="Fellowship, TA, RA, Partial, or Self-funded")
    undergrad_university_tier: str = Field(..., description="1, 2, or 3 based on undergrad university tier")


# =============================================================================
# SECTION 4 — ENDPOINTS
# =============================================================================

# Root endpoint — just a health check to confirm the server is running
# When you visit http://localhost:8000 in your browser you see this response
@app.get("/")
def root():
    return {
        "message": "Student Admission Calculator API is running",
        "docs": "Visit /docs for interactive API documentation"
    }


# Health check endpoint — React can ping this to confirm the backend is alive
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# =============================================================================
# MAIN PREDICTION ENDPOINT
# =============================================================================
# This is the main endpoint React calls when the user clicks "Predict"
# POST means React is sending data (the applicant profile) to this URL
# @app.post("/predict") tells FastAPI: when a POST request comes to /predict, run this function

@app.post("/predict")
def get_prediction(profile: ApplicantProfile):
    """
    Receives a student's profile from React and returns admission prediction.

    Expects: ApplicantProfile JSON body
    Returns: prediction results including probability, verdict, and confidence
    """

    try:
        # Convert the Pydantic model to a plain dictionary
        user_input = profile.dict()

        # Extract degree_type and clean it up
        # .lower() converts to lowercase so 'Masters' and 'MASTERS' both work
        # .strip() removes any accidental whitespace
        degree_type = str(user_input.pop('degree_type')).lower().strip()

        # Validate degree_type explicitly with a clear error message
        if degree_type not in ['masters', 'phd']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid degree_type '{degree_type}'. Must be 'masters' or 'phd'."
            )

        # Call the predict function from predict.py
        result = predict(user_input, degree_type)

        # Return the prediction result to React
        return result

    except HTTPException:
        # Re-raise HTTP exceptions as-is so FastAPI handles them properly
        raise

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# =============================================================================
# SECTION 5 — RUN THE SERVER
# =============================================================================
# uvicorn is the server that actually runs FastAPI
# host="0.0.0.0" makes it accessible on your network
# port=8000 is the port React will send requests to
# reload=True automatically restarts the server when you save changes (dev mode)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)