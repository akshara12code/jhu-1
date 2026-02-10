"""Pydantic models for API."""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class PatientInput(BaseModel):
    """Patient symptom input."""
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(male|female|other)$")
    symptoms_text: str = Field(..., min_length=10, max_length=2000, 
                               description="Describe your symptoms in natural language")
    symptom_duration_days: int = Field(..., ge=0, le=365)
    severity: Severity


class EnhancedPatientInput(BaseModel):
    """Patient input with optional document data."""
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(male|female|other)$")
    symptoms_text: str = Field(..., min_length=10, max_length=2000)
    symptom_duration_days: int = Field(..., ge=0, le=365)
    severity: Severity
    
    # Optional: Data from uploaded documents
    document_text: Optional[str] = Field(None, description="Text extracted from medical documents")
    previous_diagnoses: Optional[List[str]] = Field(default_factory=list)
    current_medications: Optional[List[str]] = Field(default_factory=list)


class MedicalEntity(BaseModel):
    """Extracted medical entity."""
    text: str
    entity_type: str
    confidence: float


class PredictedCondition(BaseModel):
    """Predicted medical condition."""
    condition_category: str
    confidence_score: float
    reasoning: str


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    patient_id: str
    
    # NER Results
    extracted_entities: List[MedicalEntity]
    
    # Classification Results
    predicted_conditions: List[PredictedCondition]
    
    # Risk Assessment
    risk_level: str
    risk_score: float
    
    # Recommendations
    recommendations: List[str]
    
    # Overall confidence
    overall_confidence: float
    
    disclaimer: str = "⚠️ AI-generated analysis for educational purposes only. Consult a healthcare professional."


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    success: bool
    message: str
    extracted_text: str
    text_preview: str
    extracted_entities: List[MedicalEntity]
    document_id: str
    file_info: dict