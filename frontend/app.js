const API_BASE = "http://localhost:8000";

let uploadedDocumentText = null;

async function analyze() {
  const payload = {
    age: Number(document.getElementById("age").value),
    gender: document.getElementById("gender").value,
    symptoms_text: document.getElementById("symptoms").value,
    symptom_duration_days: Number(document.getElementById("duration").value),
    severity: document.getElementById("severity").value
  };

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  document.getElementById("output").textContent =
    JSON.stringify(data, null, 2);
}

async function uploadDocument() {
  const fileInput = document.getElementById("document");
  const file = fileInput.files[0];

  if (!file) {
    alert("Select a document first");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload-document`, {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  uploadedDocumentText = data.extracted_text;

  document.getElementById("docStatus").innerText =
    "Document processed successfully ✔";
}
