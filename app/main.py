"""
FastAPI application - Pure Hugging Face models with document upload.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid

from app.models import (
    PatientInput,
    EnhancedPatientInput,
    AnalysisResult,
    MedicalEntity,
    PredictedCondition,
    DocumentUploadResponse
)
from app.ml_service import get_ml_service
from app.document_service import get_document_service

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MedAssist Pro - Pure ML CDSS with Document Upload",
    version="2.0.0",
    description="Clinical Decision Support using Hugging Face models + Document Processing"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("=" * 60)
    logger.info("MedAssist Pro - Document Upload Version")
    logger.info("=" * 60)
    
    try:
        # Initialize ML service
        get_ml_service()
        logger.info("✓ ML models loaded")
        
        # Initialize document service
        get_document_service()
        logger.info("✓ Document processing ready")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MedAssist Pro - Clinical Decision Support System",
        "version": "2.0.0",
        "features": [
            "Symptom Analysis with AI",
            "Medical Document Upload (PDF/Images)",
            "Risk Stratification",
            "Evidence-based Recommendations"
        ],
        "models": [
            "Medical NER (d4data/biomedical-ner-all)",
            "Zero-shot Classification (facebook/bart-large-mnli)",
            "BioBERT QA (dmis-lab/biobert-v1.1)"
        ],
        "disclaimer": "⚠️ For educational purposes only",
        "endpoints": {
            "analyze": "/api/analyze",
            "upload_document": "/api/upload-document",
            "analyze_with_documents": "/api/analyze-with-documents",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "ml_models_loaded": True,
        "document_processing_ready": True
    }


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_symptoms(patient_input: PatientInput):
    """
    Analyze patient symptoms using pure ML models.
    
    Steps:
    1. Extract medical entities (NER)
    2. Classify into disease categories (Zero-shot)
    3. Calculate risk score (Rule-based + ML)
    4. Generate recommendations
    """
    try:
        logger.info("=" * 60)
        logger.info(f"New analysis request")
        logger.info(f"Symptoms: {patient_input.symptoms_text[:100]}...")
        
        ml_service = get_ml_service()
        patient_id = f"PAT-{uuid.uuid4().hex[:8].upper()}"
        
        # Extract entities
        logger.info("Step 1: Extracting medical entities...")
        entities = ml_service.extract_medical_entities(patient_input.symptoms_text)
        medical_entities = [MedicalEntity(**entity) for entity in entities]
        
        # Classify symptoms
        logger.info("Step 2: Classifying symptoms...")
        predictions = ml_service.classify_symptoms(patient_input.symptoms_text, top_k=5)
        predicted_conditions = [
            PredictedCondition(
                condition_category=pred['category'],
                confidence_score=pred['confidence'],
                reasoning=pred['reasoning']
            )
            for pred in predictions
        ]
        
        # Risk assessment
        logger.info("Step 3: Assessing risk...")
        risk_level, risk_score = ml_service.assess_risk(
            age=patient_input.age,
            severity=patient_input.severity.value,
            symptom_duration_days=patient_input.symptom_duration_days,
            entities=entities,
            predictions=predictions
        )
        
        # Recommendations
        logger.info("Step 4: Generating recommendations...")
        recommendations = ml_service.generate_recommendations(
            predictions=predictions,
            risk_score=risk_score,
            severity=patient_input.severity.value
        )
        
        overall_confidence = predictions[0]['confidence'] if predictions else 0.0
        
        result = AnalysisResult(
            patient_id=patient_id,
            extracted_entities=medical_entities,
            predicted_conditions=predicted_conditions,
            risk_level=risk_level,
            risk_score=risk_score,
            recommendations=recommendations,
            overall_confidence=overall_confidence
        )
        
        logger.info(f"✓ Analysis complete for {patient_id}")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-document", response_model=DocumentUploadResponse)
async def upload_medical_document(file: UploadFile = File(...)):
    """
    Upload medical document (PDF or image) and extract medical information.
    
    Supports:
    - PDF medical reports
    - Image scans of prescriptions/lab results
    - Medical history documents
    
    Max file size: 10 MB
    Allowed formats: PDF, JPG, JPEG, PNG
    """
    try:
        logger.info(f"📄 Document upload: {file.filename}")
        
        # Validate file type
        file_extension = file.filename.split(".")[-1].lower()
        allowed_types = ["pdf", "jpg", "jpeg", "png"]
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file_extension}'. Allowed: {', '.join(allowed_types)}"
            )
        
        # Read file
        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({file_size_mb:.1f} MB). Maximum size: 10 MB"
            )
        
        # Process document
        logger.info("Processing document...")
        doc_service = get_document_service()
        doc_data = doc_service.process_medical_document(
            file_bytes, 
            file_extension,
            file.filename
        )
        
        # Extract medical entities from document text
        logger.info("Extracting medical entities from document...")
        ml_service = get_ml_service()
        entities = ml_service.extract_medical_entities(doc_data["extracted_text"])
        
        medical_entities = [
            MedicalEntity(**entity) for entity in entities
        ]
        
        # Generate document ID
        document_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
        
        # Create preview (first 500 chars)
        text_preview = doc_data["extracted_text"][:500]
        if len(doc_data["extracted_text"]) > 500:
            text_preview += "..."
        
        logger.info(f"✓ Document processed: {document_id}")
        logger.info(f"  - Extracted {doc_data['text_length']} characters")
        logger.info(f"  - Found {len(medical_entities)} medical entities")
        
        return DocumentUploadResponse(
            success=True,
            message=f"Document processed successfully. Extracted {len(medical_entities)} medical entities.",
            extracted_text=doc_data["extracted_text"],
            text_preview=text_preview,
            extracted_entities=medical_entities,
            document_id=document_id,
            file_info={
                "filename": file.filename,
                "file_type": file_extension,
                "size_mb": round(file_size_mb, 2),
                "text_length": doc_data["text_length"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-with-documents", response_model=AnalysisResult)
async def analyze_with_documents(patient_input: EnhancedPatientInput):
    """
    Analyze patient with both symptoms AND uploaded document data.
    
    This combines:
    - Manual symptom input from patient
    - Extracted medical history from uploaded documents
    
    For more comprehensive and accurate analysis.
    """
    try:
        logger.info("=" * 60)
        logger.info("Enhanced analysis with documents requested")
        
        ml_service = get_ml_service()
        patient_id = f"PAT-{uuid.uuid4().hex[:8].upper()}"
        
        # Combine symptoms text with document text
        combined_text = patient_input.symptoms_text
        
        if patient_input.document_text:
            logger.info("Including document text in analysis")
            combined_text += "\n\nMedical History from Documents:\n" + patient_input.document_text[:1000]
        
        # Extract entities from combined text
        logger.info("Extracting entities from combined data...")
        entities = ml_service.extract_medical_entities(combined_text)
        medical_entities = [MedicalEntity(**entity) for entity in entities]
        
        # Classify symptoms
        logger.info("Classifying combined symptoms...")
        predictions = ml_service.classify_symptoms(combined_text, top_k=5)
        predicted_conditions = [
            PredictedCondition(
                condition_category=pred['category'],
                confidence_score=pred['confidence'],
                reasoning=pred['reasoning']
            )
            for pred in predictions
        ]
        
        # Risk assessment
        logger.info("Assessing risk with document context...")
        risk_level, risk_score = ml_service.assess_risk(
            age=patient_input.age,
            severity=patient_input.severity.value,
            symptom_duration_days=patient_input.symptom_duration_days,
            entities=entities,
            predictions=predictions
        )
        
        # Generate recommendations
        recommendations = ml_service.generate_recommendations(
            predictions=predictions,
            risk_score=risk_score,
            severity=patient_input.severity.value
        )
        
        # Add note about document usage
        if patient_input.document_text:
            recommendations.insert(0, "✓ Analysis enhanced with uploaded medical history documents")
        
        overall_confidence = predictions[0]['confidence'] if predictions else 0.0
        
        result = AnalysisResult(
            patient_id=patient_id,
            extracted_entities=medical_entities,
            predicted_conditions=predicted_conditions,
            risk_level=risk_level,
            risk_score=risk_score,
            recommendations=recommendations,
            overall_confidence=overall_confidence
        )
        
        logger.info(f"✓ Enhanced analysis complete for {patient_id}")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)