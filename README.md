# MedAssist Pro 🏥

**AI-Powered Clinical Decision Support System with Document Processing**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/)


<img width="943" height="966" alt="Screenshot 2026-02-10 232913" src="https://github.com/user-attachments/assets/f8d9db4f-e692-4b87-87d4-a4bc2a3fbaae" />

<img width="941" height="963" alt="Screenshot 2026-02-10 232920" src="https://github.com/user-attachments/assets/f38b264f-1d84-40a4-9349-51aa9dbf4123" />

<img width="903" height="695" alt="Screenshot 2026-02-10 232929" src="https://github.com/user-attachments/assets/245eaaf5-77e2-4256-92aa-27ef36910046" />

<img width="871" height="604" alt="Screenshot 2026-02-10 232938" src="https://github.com/user-attachments/assets/e8858320-ff14-451b-82db-5a019fa59b60" />

## 📋 Overview

MedAssist Pro is an advanced Clinical Decision Support System (CDSS) that combines state-of-the-art Natural Language Processing with medical document analysis. The system helps analyze patient symptoms, extract medical entities, predict potential conditions, and provide evidence-based recommendations.

### Key Features

- 🔍 **Medical Entity Extraction** - Identifies diseases, symptoms, medications, and treatments from text
- 🎯 **Symptom Classification** - Zero-shot classification into disease categories
- 📄 **Document Processing** - Upload and analyze medical PDFs, prescriptions, and lab results
- ⚕️ **Risk Stratification** - ML-powered risk assessment based on multiple factors
- 💡 **Smart Recommendations** - Context-aware clinical recommendations
- 🌐 **Modern Web Interface** - Clean, responsive UI for easy interaction

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (for ML models)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/akshara12code/jhu-1
cd medassist-pro
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
uvicorn app.main:app --reload
```

5. **Access the interface**

Open your browser and navigate to:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🏗️ Architecture

```
medassist-pro/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic data models
│   ├── ml_service.py           # ML model inference
│   └── document_service.py     # Document processing
├── frontend/
│   ├── index.html              # Web interface
│   ├── styles.css              # Styling
│   └── script.js               # Frontend logic
├── requirements.txt            # Python dependencies
└── README.md
```

## 🤖 ML Models Used

| Model | Purpose | Provider |
|-------|---------|----------|
| **biomedical-ner-all** | Medical entity extraction | d4data/Hugging Face |
| **BART-large-MNLI** | Zero-shot symptom classification | Facebook AI |
| **BioBERT v1.1** | Medical question answering | dmis-lab |

### Model Download

Models are automatically downloaded on first startup. This requires:
- ~2GB disk space
- Internet connection
- 5-10 minutes initial setup time

## 📡 API Endpoints

### 1. Analyze Symptoms
```http
POST /api/analyze
Content-Type: application/json

{
  "symptoms_text": "Patient experiencing severe headache and fever",
  "age": 35,
  "severity": "moderate",
  "symptom_duration_days": 3
}
```

### 2. Upload Medical Document
```http
POST /api/upload-document
Content-Type: multipart/form-data

file: [PDF/Image file]
```

### 3. Enhanced Analysis with Documents
```http
POST /api/analyze-with-documents
Content-Type: application/json

{
  "symptoms_text": "Current symptoms...",
  "document_text": "Extracted medical history...",
  "age": 35,
  "severity": "moderate",
  "symptom_duration_days": 3
}
```

## 💻 Usage Examples

### Python SDK

```python
import requests

# Analyze symptoms
response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "symptoms_text": "Persistent cough, shortness of breath, and fatigue",
        "age": 45,
        "severity": "moderate",
        "symptom_duration_days": 7
    }
)

result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Predicted Conditions: {result['predicted_conditions']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms_text": "Severe chest pain radiating to left arm",
    "age": 55,
    "severity": "severe",
    "symptom_duration_days": 1
  }'
```

### Document Upload

```python
# Upload medical document
with open("lab_results.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/api/upload-document",
        files=files
    )

document_data = response.json()
print(f"Extracted Entities: {document_data['extracted_entities']}")
```

## 🎨 Web Interface

The web interface provides an intuitive way to:

1. **Enter Symptoms** - Describe symptoms in natural language
2. **Upload Documents** - Drop medical PDFs or images
3. **View Analysis** - See extracted entities, predictions, and risk assessment
4. **Get Recommendations** - Receive evidence-based clinical guidance

![Demo Screenshot](docs/screenshot.png) _(Add your screenshot)_

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000

# Model Configuration
MODEL_CACHE_DIR=./models
MAX_UPLOAD_SIZE_MB=10

# Logging
LOG_LEVEL=INFO
```

### Model Configuration

Customize models in `app/ml_service.py`:

```python
NER_MODEL = "d4data/biomedical-ner-all"
CLASSIFIER_MODEL = "facebook/bart-large-mnli"
QA_MODEL = "dmis-lab/biobert-v1.1"
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

## 📊 Performance

- **Average Response Time**: 2-5 seconds
- **Document Processing**: <10 seconds for PDFs up to 10MB
- **Concurrent Requests**: Supports up to 10 simultaneous analyses
- **Memory Usage**: ~2-3GB RAM with models loaded

## 🛡️ Security & Privacy

- ✅ No data storage - analyses are performed in-memory
- ✅ No external API calls for patient data
- ✅ File upload validation and size limits
- ✅ CORS protection enabled
- ⚠️ **Not HIPAA compliant** - educational use only



## 🗺️ Roadmap

- [ ] Add more medical domain models
- [ ] Implement conversation history
- [ ] Support for DICOM medical images
- [ ] Multi-language support
- [ ] Integration with FHIR standards
- [ ] Mobile application
- [ ] Docker containerization
- [ ] Kubernetes deployment configs

---

**Made with ❤️ for the medical AI community**

⭐ Star this repo if you find it helpful!
