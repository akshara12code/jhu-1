"""
Pure Hugging Face Medical ML Service.
Uses only pre-trained models, no manual disease database.
"""

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification
)
import torch
import logging
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MedicalMLService:
    """Medical ML service using only Hugging Face models."""
    
    def __init__(self):
        """Initialize all ML models."""
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'CUDA' if self.device == 0 else 'CPU'}")
        
        self._init_models()
    
    def _init_models(self):
        """Initialize Hugging Face models."""
        try:
            # 1. Medical NER - Extract medical entities
            logger.info("Loading Medical NER model...")
            self.ner_pipeline = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                device=self.device,
                aggregation_strategy="simple"
            )
            
            # 2. Zero-shot Classification - Classify into disease categories
            logger.info("Loading Zero-shot Classification model...")
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
            
            # 3. Medical Question Answering (BioBERT)
            logger.info("Loading BioBERT QA model...")
            self.qa_pipeline = pipeline(
                "question-answering",
                model="dmis-lab/biobert-v1.1",
                device=self.device
            )
            
            # Define disease categories for zero-shot classification
            self.disease_categories = [
                "Respiratory infection (cold, flu, COVID-19, pneumonia)",
                "Cardiovascular disease (hypertension, heart disease)",
                "Gastrointestinal disorder (gastritis, IBS, food poisoning)",
                "Neurological condition (migraine, headache, dizziness)",
                "Musculoskeletal problem (arthritis, muscle pain, injury)",
                "Allergic reaction (hay fever, food allergy, skin allergy)",
                "Mental health condition (anxiety, depression, stress)",
                "Metabolic disorder (diabetes, thyroid issues)",
                "Infectious disease (bacterial or viral infection)",
                "Dermatological condition (skin rash, eczema, acne)"
            ]
            
            logger.info("✓ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def extract_medical_entities(self, text: str) -> List[Dict]:
        """
        Extract medical entities using NER.
        
        Args:
            text: Patient symptom description
            
        Returns:
            List of extracted entities with types and confidence
        """
        try:
            entities = self.ner_pipeline(text)
            
            # Process and deduplicate
            processed_entities = []
            seen_texts = set()
            
            for entity in entities:
                entity_text = entity['word'].strip()
                
                # Skip duplicates and very short entities
                if entity_text in seen_texts or len(entity_text) < 3:
                    continue
                
                seen_texts.add(entity_text)
                
                processed_entities.append({
                    'text': entity_text,
                    'entity_type': entity['entity_group'],
                    'confidence': round(float(entity['score']), 3)
                })
            
            logger.info(f"Extracted {len(processed_entities)} entities")
            return processed_entities
            
        except Exception as e:
            logger.error(f"NER error: {e}")
            return []
    
    def classify_symptoms(self, text: str, top_k: int = 5) -> List[Dict]:
        """
        Classify symptoms into disease categories using zero-shot classification.
        
        Args:
            text: Symptom description
            top_k: Number of top predictions
            
        Returns:
            List of predicted disease categories with scores
        """
        try:
            # Use zero-shot classification
            result = self.zero_shot_classifier(
                text,
                candidate_labels=self.disease_categories,
                multi_label=True
            )
            
            # Get top K predictions
            predictions = []
            for i in range(min(top_k, len(result['labels']))):
                category = result['labels'][i]
                score = result['scores'][i]
                
                # Only include if confidence is reasonable
                if score > 0.1:
                    predictions.append({
                        'category': category,
                        'confidence': round(float(score), 3),
                        'reasoning': self._generate_reasoning(category, score)
                    })
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return []
    
    def _generate_reasoning(self, category: str, confidence: float) -> str:
        """Generate human-readable reasoning for prediction."""
        confidence_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        
        return f"Based on symptom analysis, there is {confidence_level} confidence ({confidence:.1%}) " \
               f"that symptoms align with {category.split('(')[0].strip()}."
    
    def assess_risk(
        self,
        age: int,
        severity: str,
        symptom_duration_days: int,
        entities: List[Dict],
        predictions: List[Dict]
    ) -> Tuple[str, float]:
        """
        Calculate risk score using rule-based system + ML predictions.
        
        Returns:
            (risk_level_description, risk_score)
        """
        risk_score = 0.0
        
        # Age-based risk (0-20 points)
        if age > 70:
            risk_score += 20
        elif age > 60:
            risk_score += 15
        elif age > 50:
            risk_score += 10
        elif age < 5:
            risk_score += 15
        elif age < 12:
            risk_score += 10
        
        # Severity-based risk (0-30 points)
        severity_scores = {
            "mild": 10,
            "moderate": 20,
            "severe": 30
        }
        risk_score += severity_scores.get(severity.lower(), 10)
        
        # Duration-based risk (0-15 points)
        if symptom_duration_days > 14:
            risk_score += 15
        elif symptom_duration_days > 7:
            risk_score += 10
        elif symptom_duration_days > 3:
            risk_score += 5
        
        # ML prediction-based risk (0-35 points)
        if predictions:
            top_prediction = predictions[0]
            confidence = top_prediction['confidence']
            
            # Higher confidence in serious conditions increases risk
            category_lower = top_prediction['category'].lower()
            
            high_risk_keywords = ['cardiovascular', 'neurological', 'infection', 'metabolic']
            is_high_risk_category = any(keyword in category_lower for keyword in high_risk_keywords)
            
            if is_high_risk_category:
                risk_score += confidence * 35
            else:
                risk_score += confidence * 20
        
        # Entity-based risk (0-10 points)
        # More entities might indicate complex presentation
        if len(entities) > 5:
            risk_score += 10
        elif len(entities) > 3:
            risk_score += 5
        
        # Normalize to 0-100
        risk_score = min(risk_score, 100.0)
        
        # Determine risk level
        if risk_score < 30:
            risk_level = "LOW RISK - Self-monitoring recommended. Symptoms may resolve with rest and basic care."
        elif risk_score < 60:
            risk_level = "MODERATE RISK - Consider scheduling appointment with healthcare provider within 1-3 days."
        elif risk_score < 80:
            risk_level = "HIGH RISK - Recommend consulting healthcare provider within 24 hours."
        else:
            risk_level = "VERY HIGH RISK - Consider urgent medical evaluation or emergency services if worsening."
        
        logger.info(f"Risk assessment: {risk_level} (Score: {risk_score:.1f})")
        return risk_level, round(risk_score, 2)
    
    def generate_recommendations(
        self,
        predictions: List[Dict],
        risk_score: float,
        severity: str
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # General recommendations
        recommendations.append("Monitor your symptoms and track any changes")
        recommendations.append("Stay well hydrated and get adequate rest")
        
        # Risk-based recommendations
        if risk_score > 70:
            recommendations.append("⚠️ URGENT: Contact healthcare provider or consider emergency services")
            recommendations.append("Do not delay seeking medical attention")
        elif risk_score > 50:
            recommendations.append("Schedule appointment with healthcare provider soon")
            recommendations.append("Keep record of symptom progression")
        else:
            recommendations.append("Continue self-monitoring for 24-48 hours")
            recommendations.append("Over-the-counter remedies may help with symptom relief")
        
        # Severity-based
        if severity == "severe":
            recommendations.append("If symptoms worsen rapidly, seek immediate medical care")
        
        # Category-specific (from top prediction)
        if predictions:
            top_category = predictions[0]['category'].lower()
            
            if 'respiratory' in top_category:
                recommendations.append("Avoid smoke and air pollutants")
                recommendations.append("Use humidifier if air is dry")
            elif 'cardiovascular' in top_category:
                recommendations.append("Monitor blood pressure if equipment available")
                recommendations.append("Avoid strenuous activity until evaluated")
            elif 'gastrointestinal' in top_category:
                recommendations.append("Follow bland diet (BRAT: bananas, rice, applesauce, toast)")
                recommendations.append("Avoid dairy and fatty foods temporarily")
            elif 'mental health' in top_category:
                recommendations.append("Practice stress-reduction techniques")
                recommendations.append("Consider speaking with mental health professional")
        
        # Always include
        recommendations.append("This is AI-generated advice - always consult qualified healthcare provider")
        
        return recommendations


# Singleton instance
_ml_service = None

def get_ml_service() -> MedicalMLService:
    """Get or create ML service singleton."""
    global _ml_service
    if _ml_service is None:
        _ml_service = MedicalMLService()
    return _ml_service