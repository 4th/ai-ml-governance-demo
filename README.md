# 🚀 AI + ML Governance Demo  
## **Technical Demonstration for Senior AI/ML Engineer Role (Transcend Staffing / Federal Client)**

This repository was intentionally developed as a **hands-on, end-to-end technical demonstration** aligned with the requirements of the **Senior AI/ML Engineer** position currently under consideration.

It showcases real-world, production-oriented capabilities across **machine learning**, **deep learning**, **generative AI**, **MLOps**, **cloud deployment**, and **AI governance**—directly mapping to the skill sets and responsibilities outlined in the role.

---

## 🎯 Purpose of This Repository

This project is designed to demonstrate:

- My ability to architect **full-stack AI systems** from data → model → API → UI → deployment  
- Practical expertise across **supervised/unsupervised ML**, neural networks, LLMs, and RAG  
- Production-ready engineering practices including MLOps, Docker, Kubernetes, CI/CD  
- Hands-on proficiency with **FastAPI, Streamlit, Databricks workflows, Python, PyTorch, scikit-learn**  
- Ability to integrate **policy-as-code**, governance, logging, and observability into ML systems  
- Strong **systems thinking**, with end-to-end lifecycle awareness  
- Experience mentoring and enabling ML teams through platform-style design  

This repository models what a **federal or enterprise AI/ML engineer** would deliver when building:

- a scalable ML inference service  
- a generative AI RAG pipeline  
- secure API endpoints  
- a user-facing interface  
- a cloud-ready deployment topology  
- governance + safety layers suitable for regulated environments  

---

## 🛠 What This Demo Covers (Mapped to Job Requirements)

### **AI/ML Development**
- Classical ML: regression, classification, clustering, time-series patterns  
- RandomForest baseline ready to upgrade to CNNs/RNNs/LSTMs  
- sklearn + PyTorch foundations  
- Feature engineering, metrics, evaluation  

### **Generative AI**
- RAG pipeline integrating:
  - retrieval  
  - prompt engineering  
  - model grounding  
- LLM client compatible with OpenAI / Azure OpenAI  
- Optional LoRA/PEFT extension hooks  
- Safety filters + provenance tracking  

### **Modern Code Development**
- Advanced Python with clean architecture  
- FastAPI microservices  
- Streamlit UI  
- Logging, configuration, dependency injection  
- Fully containerized (Docker) with Kubernetes deployment templates  

### **Model Management & Deployment**
- Model serialization via Pickle/Joblib  
- ONNX-ready export pipeline  
- FastAPI inference endpoints  
- Serverless-friendly folder structure  
- Streamlit tools for user interaction  

### **Platform Enablement (Databricks)**
- `databricks/` folder models notebook-driven development  
- Supports spark-based scaling and MLflow integration  
- Ready for AutoML and feature store integration  

### **Hands-On Data Work**
- pandas, polars, numpy, seaborn  
- Visualization and storytelling  
- Strong preprocessing and feature pipelines  

### **Systems Thinking**
- Complete data→model→API→UI→governance pipeline  
- Modular, maintainable architecture  
- Designed for scale, observability, and future enhancements  

---

## 🤝 Collaboration & Mentorship

The codebase demonstrates how I:

- Build reusable frameworks for other engineers  
- Document architecture and patterns  
- Enable self-service model deployment  
- Encourage best practices in testing, logging, and MLOps  
- Contribute to a healthy engineering culture  

---

Together, this repository functions as a **comprehensive, production-minded technical portfolio piece** demonstrating the exact capabilities required for a **Senior AI/ML Engineer** supporting a Washington DC–based federal client.

---

### 1. Overview

This project provides an end-to-end, production-ready example of a modern **enterprise AI/ML system** integrating classical machine learning, large language models (LLMs), Retrieval-Augmented Generation (RAG), FastAPI microservices, Streamlit UI, and Databricks-style training workflows.

The architecture demonstrates how real-world AI systems are engineered in organizations with requirements spanning:

- **Machine learning pipelines**
- **Generative AI integration**
- **Model deployment & microservices**
- **MLOps and artifact management**
- **Observability, logging, and governance**
- **Cloud-ready containerization**

---

### **Project Purpose**

This repository serves as a comprehensive engineering demonstration and portfolio example showing mastery of:

- End-to-end ML system design  
- Python engineering best practices  
- Model training and evaluation  
- Feature engineering  
- Building API-based inference services  
- Integrating LLM reasoning via a RAG pipeline  
- Deploying applications with Docker  
- Designing UI workflows for interacting with ML/LLM systems  

---

### **Key Capabilities Included**

- **Supervised ML Model (RandomForest)**  
  Trained on the Iris dataset with metrics, serialization, and production-safe loading.

- **LLM-Powered RAG Pipeline**  
  Includes a knowledge base, retrieval system, prompt builder, and OpenAI/Azure client.

- **FastAPI Service Layer**  
  Routes prediction and RAG requests, performs validation, orchestrates inference flows.

- **Streamlit User Interface**  
  Provides interactive prediction forms and natural-language Q&A.

- **Databricks-Compatible Training Notebooks**  
  Allows experimentation, feature engineering, and MLflow integration.

- **Central Logging & Configuration**  
  Via Loguru and structured settings for portability across environments.

- **Containerized Deployment**  
  Ready for Docker, Kubernetes, and all major cloud providers.

---

### **Why This Matters**

This system reflects real enterprise ML engineering patterns, demonstrating:

- Modular, well-structured Python code  
- Proper separation of training, inference, and UI layers  
- Extensibility to integrate governance, risk controls, and monitoring  
- Production-aligned architecture following industry best practices  

This project can be used for:

- Technical interviews  
- Portfolio demonstrations  
- Internal enterprise prototypes  
- ML engineering, MLOps, or LLM deployment templates  


## 2. Architecture

This project is structured as a **small but realistic enterprise AI system**. It combines:

- A **classical ML model** served behind a FastAPI microservice  
- A **RAG (Retrieval-Augmented Generation) pipeline** powered by an LLM  
- A **Streamlit UI** for interactive exploration  
- **Databricks-style notebooks** for data and model work  
- **Logging and configuration layers** suitable for production environments  
- A **Dockerized** deployment model ready for cloud platforms  

The architecture is intentionally modular so each layer can be scaled, swapped, or hardened independently.

---

### 2.1 High-Level System Diagram

```mermaid
flowchart LR
    U[User<br/>Browser] -->|HTTP| UI[Streamlit UI]

    UI -->|REST calls| API[FastAPI Service]

    API -->|/predict| MODEL[Sklearn Model<br/>(RandomForest)]
    API -->|/rag| RAG[RAG Pipeline]

    RAG --> KB[Knowledge Base<br/>(JSON / future vector store)]
    RAG --> LLM[LLM Client<br/>(OpenAI / Azure OpenAI)]

    API --> LOG[Logging & Metrics]
    MODEL --> LOG
    RAG --> LOG
    LLM --> LOG
 ```
 ---

### 2.2 Code-Level Architecture

The repository is organized into clear, responsibility-driven modules:

```
src/
  api/           # FastAPI app (HTTP interface)
  models/        # Model training and loading
  ml/            # Preprocessing & feature engineering
  generative/    # RAG + LLM integration
  ui/            # Streamlit frontend
  utils/         # Logging & metrics
  config.py      # Central configuration
  main.py        # Optional entrypoint for uvicorn
```

---

#### `src/api/fastapi_app.py`
- Defines API endpoints: `/health`, `/predict`, `/rag`
- Validates requests via Pydantic models
- Loads and caches the ML model
- Routes inference to ML or RAG components

---

#### `src/models/`
- `train_example_model.py`: trains RandomForest, evaluates metrics, serializes model
- `load_model.py`: safely loads serialized model for inference

---

#### `src/ml/`
- `preprocessing.py`: data cleaning and train/test splitting
- `features.py`: custom feature engineering utilities

---

#### `src/generative/`
- `knowledge_base.json`: small knowledge base for retrieval
- `retriever.py`: keyword-based document retrieval
- `llm_client.py`: LLM wrapper for OpenAI/Azure/OpenAI-compatible APIs
- `llm_rag_app.py`: RAG orchestration layer

---

#### `src/ui/streamlit_app.py`
- UI panel for ML prediction
- UI panel for RAG Q&A interactions

---

#### `src/utils/`
- `logging_utils.py`: Loguru logger configuration with rotation
- `metrics.py`: reusable metric functions (accuracy, precision, recall, F1)

---

#### `src/config.py`
- Environment-aware configuration for API keys, ports, paths, model names

---

#### `src/main.py`
- Optional uvicorn entrypoint for containerized execution

### 2.3 Data & Control Flow

This project contains two primary inference flows:

- **Supervised ML prediction** (`/predict`)
- **RAG / LLM augmented generation** (`/rag`)

---

### 2.3.1 ML Prediction Flow (`/predict`)

1. User enters numeric features in the Streamlit UI.
2. Streamlit sends a POST request to FastAPI:

```
{
  "feature1": 5.1,
  "feature2": 3.5,
  "feature3": 1.4
}
```

3. FastAPI:
   - Validates payload using `PredictRequest`
   - Converts features into NumPy array
   - Invokes the sklearn model’s `predict_proba`

4. Model returns class probabilities.

5. API computes and returns:

```
{
  "prediction": 0,
  "probability": 0.97
}
```

6. Streamlit displays the prediction and confidence.

---

### 2.3.2 RAG / LLM Flow (`/rag`)

1. User enters a natural-language question in the Streamlit UI.
2. Streamlit sends a POST request:

```
{
  "question": "What is model monitoring?"
}
```

3. FastAPI validates via `RAGRequest` and forwards to the RAG layer.

4. Retriever:
   - Loads documents from `knowledge_base.json`
   - Performs keyword-based scoring
   - Selects top-k relevant documents

5. RAG orchestrator:
   - Builds a structured prompt with system instructions + retrieved context
   - Calls the LLM client:
     - Uses OpenAI/Azure if API key is present  
     - Falls back to deterministic offline-safe response otherwise

6. RAG layer produces:

```
{
  "answer": "...",
  "sources": ["AI Governance Basics", "Model Monitoring"]
}
```

7. Streamlit displays answer + cited source titles.

---

### 2.3 Summary

These flows demonstrate a realistic enterprise pattern:

- deterministic classical ML inference  
- retrieval-augmented reasoning with LLMs  
- structured, logged, auditable request/response cycles


### 2.4 Deployment Topology

The system is designed to run in multiple deployment configurations, from simple local execution to cloud‑native microservices.

---

#### **Local Development**

- Run FastAPI using `uvicorn`
- Run Streamlit UI independently

```
uvicorn src.api.fastapi_app:app --reload
streamlit run src/ui/streamlit_app.py
```

---

#### **Single-Container Deployment**

FastAPI is packaged in a production-ready Docker image:

```
docker build -t ai-ml-governance-demo .
docker run -p 8000:8000 ai-ml-governance-demo
```

- Streamlit can run in its own container or be bundled together
- Image exposes port `8000` by default

---

#### **Microservices Deployment (Recommended)**

- **Container 1:** FastAPI inference + RAG API  
- **Container 2:** Streamlit UI  
- **Container 3 (optional):** Vector DB / embeddings service  
- **Container 4 (optional):** Logging/monitoring stack (ELK, Prometheus/Grafana)

---

#### **Supported Cloud Environments**

**Azure**
- Container Apps  
- AKS  
- App Service  
- Azure Functions

**AWS**
- ECS Fargate  
- EKS  
- Lambda + API Gateway

**Google Cloud**
- Cloud Run  
- GKE

**Kubernetes Anywhere**
- On‑prem  
- Hybrid cloud  
- Custom ML/GPU clusters


### 2.5 Databricks & Offline Training Environment

The `databricks/` directory models a realistic enterprise workflow for developing, training, and exporting machine learning models in a collaborative environment.

---

#### **Notebook-Centric Development**

In enterprise environments, data science teams often use Databricks notebooks for:

- Feature engineering  
- Data exploration  
- Model experimentation  
- Pipeline prototyping  
- MLflow logging  

This project mirrors that workflow by providing a Databricks-ready structure.

---

#### **Training & Artifact Lifecycle**

1. **Experiment & Train in Databricks**  
   Data scientists iterate on model architectures, hyperparameters, and evaluation steps inside notebooks.

2. **Generate Model Artifacts**  
   Artifacts may include:  
   - `model.pkl` (Pickle)  
   - `model.onnx` (ONNX runtime-compatible)  
   - `metrics.json` (evaluation metadata)  
   - `preprocessing.json` (optional schema, scalers, encoders)  

3. **Export Artifacts to Storage**  
   Models are exported to:  
   - Azure Blob Storage  
   - AWS S3  
   - GCP Cloud Storage  
   - or directly copied into `src/models/` for this demo  

4. **Deploy Artifacts into FastAPI**  
   The FastAPI service loads these stable, versioned files during startup and exposes them via `/predict`.

---

#### **Enterprise Alignment**

This design reflects the industry-standard pattern:

> **“Data scientists iterate in notebooks; platform engineers deploy the artifacts.”**

This separation improves:

- Reproducibility  
- Governance  
- Version control  
- Auditing  
- Model traceability  

---

#### **Advantages of This Setup**

- Compatible with MLflow for experiment tracking  
- Supports distributed training  
- Clean separation between research and production  
- Ideal for hybrid cloud + on-prem environments  
- Reduces coupling between training and serving layers  

### 3. Components

This section provides a detailed breakdown of each subsystem in the project, all implemented using clean, modular, and enterprise-friendly Python architecture.

---

### 3.1 Classical Machine Learning Pipeline

The ML pipeline handles training, evaluating, and serializing a supervised learning model.

#### **Key Files**
- `src/models/train_example_model.py`  
- `src/models/load_model.py`  
- `src/ml/preprocessing.py`  
- `src/ml/features.py`  
- `src/utils/metrics.py`

#### **Functionality**
- Loads and preprocesses training data  
- Applies feature engineering  
- Splits data into train/test sets  
- Trains a `RandomForestClassifier`  
- Computes evaluation metrics (accuracy, precision, recall, F1)  
- Saves serialized model artifacts (`model.pkl`)  
- Provides safe runtime loading with caching  

#### **Enterprise Characteristics**
- Deterministic pipeline  
- Clean separation of training vs inference  
- Production-safe serialization  
- Reusable utilities for future models  

---

### 3.2 FastAPI Inference Service

This microservice exposes inference endpoints and orchestrates ML + RAG workflows.

#### **Key File**
- `src/api/fastapi_app.py`

#### **Endpoints**
- `GET /health` — health check  
- `POST /predict` — classical ML inference  
- `POST /rag` — LLM-powered retrieval-augmented generation  

#### **Capabilities**
- Pydantic request/response models  
- Lifecycle startup events  
- Model caching across requests  
- Typed error responses  
- Logging and monitoring instrumentation  

---

### 3.3 Retrieval-Augmented Generation (RAG) System

Adds explainable, contextualized reasoning using LLMs + domain knowledge.

#### **Key Files**
- `src/generative/knowledge_base.json`  
- `src/generative/retriever.py`  
- `src/generative/llm_client.py`  
- `src/generative/llm_rag_app.py`

#### **Components**
- **Knowledge Base** — JSON-based mini-corpus  
- **Retriever** — keyword relevance scoring  
- **LLM Client** — unified interface for OpenAI/Azure/compatible APIs  
- **RAG Orchestrator** — builds contextual prompts and returns answers + sources  

#### **Features**
- Deterministic fallback mode when no API key exists  
- Modular retrieval layer (simple now, pluggable with vector DBs later)  
- Source attribution for transparency  
- Governance-ready prompt wrapper  

---

### 3.4 Streamlit User Interface

Provides an interactive, user-friendly front-end experience.

#### **Key File**
- `src/ui/streamlit_app.py`

#### **Capabilities**
- Feature input form for `/predict`  
- Textbox for natural-language `/rag` questions  
- Real-time result rendering  
- Error-safe UX patterns  

#### **Enterprise Value**
- Demonstrates integration of UI + APIs  
- Suitable for demos, stakeholder reviews, and lightweight internal tools  

---

### 3.5 Utilities & System Services

Reusable cross-cutting system modules.

#### **Key Files**
- `src/utils/logging_utils.py`
- `src/utils/metrics.py`
- `src/config.py`

#### **Logging**
- Structured Loguru logging  
- Rotating log files  
- Context-aware logging  

#### **Metrics**
- Accuracy  
- Precision  
- Recall  
- F1 Score  

#### **Configuration**
- `.env`-driven config  
- Centralized environment-aware variables  
- Cloud-friendly patterns for secrets and service URLs  

---

### 3.6 Databricks-Compatible Notebooks

Located in the `databricks/` directory.

#### **Purpose**
- Feature engineering  
- Experimentation  
- Training  
- MLflow tracking (optional)  

#### **Role in Architecture**
- Mirrors real enterprise workflows  
- Produces artifacts deployed by FastAPI  

---

### 3.7 Containerization & Deployment

Docker support enables cloud-ready deployment.

#### **Key File**
- `docker/Dockerfile`

#### **Capabilities**
- FastAPI container build  
- Production-level startup commands  
- Supports Kubernetes, ECS, ACI, Cloud Run, etc.  

---

### 3.8 End-to-End Summary

The system integrates:

- **Data preprocessing**  
- **Model training**  
- **RAG pipeline**  
- **LLM orchestration**  
- **API microservices**  
- **Interactive UI**  
- **Logging + metrics**  
- **Cloud deployment**  

This forms a complete, production-aligned demonstration of modern AI and ML engineering.

### 4. Setup Instructions

Follow these steps to install, configure, and run the full AI + ML Governance Demo system.

---

### 4.1 Prerequisites

Ensure you have the following installed:

- **Python 3.9+**
- **pip** (latest)
- **virtualenv** or `venv`
- **Git**
- **Docker** (optional, for container deployment)
- **OpenAI or Azure OpenAI API key** (optional; the RAG system will still run in offline-fallback mode)

---

### 4.2 Clone the Repository

```
git clone https://github.com/<your-username>/ai-ml-governance-demo.git
cd ai-ml-governance-demo
```

---

### 4.3 Create a Virtual Environment

```
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate        # Windows
```

---

### 4.4 Install Dependencies

```
pip install -r requirements.txt
```

This installs:

- FastAPI
- Uvicorn
- Streamlit
- Scikit-learn
- Loguru
- Requests
- Python-dotenv

---

### 4.5 Environment Configuration

Create a `.env` file at the project root:

```
OPENAI_API_KEY=<your-api-key-or-leave-empty>
MODEL_PATH=src/models/model.pkl
APP_ENV=development
```

If `OPENAI_API_KEY` is empty, the system will run with **deterministic offline-safe LLM responses**.

---

### 4.6 Train the Example Model

Before running inference, train the supervised ML model:

```
python -m src.models.train_example_model
```

This will:

- Load the dataset  
- Train the RandomForest model  
- Compute evaluation metrics  
- Save the serialized model to `src/models/model.pkl`  

---

### 4.7 Run the FastAPI Service

```
uvicorn src.api.fastapi_app:app --reload
```

Service starts at:

```
http://127.0.0.1:8000
```

API docs are automatically available at:

```
http://127.0.0.1:8000/docs
```

---

### 4.8 Run the Streamlit UI

In a separate terminal:

```
streamlit run src/ui/streamlit_app.py
```

The UI includes:

- ML Prediction panel  
- RAG Question-Answer panel  

---

### 4.9 Running the Entire System with Docker (Optional)

Build the container:

```
docker build -t ai-ml-governance-demo .
```

Run it:

```
docker run -p 8000:8000 ai-ml-governance-demo
```

---

### 4.10 Optional: Test Endpoints Manually

**ML Prediction:**

```
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"feature1": 5.1, "feature2": 3.5, "feature3": 1.4}'
```

**RAG Query:**

```
curl -X POST http://localhost:8000/rag   -H "Content-Type: application/json"   -d '{"question": "What is model governance?"}'
```

---

### 4.11 Summary

Your environment is now fully configured for:

- Supervised ML inference  
- RAG reasoning with LLM fallback  
- UI interaction  
- API microservice execution  
- Optional cloud deployment

This setup mirrors real-world enterprise AI workflows and ensures reproducibility across development, staging, and production environments.
### 5. Model Training

This section describes the full workflow for training, evaluating, and exporting the supervised machine learning model used in the AI + ML Governance Demo system.

---

### 5.1 Training Script Overview

The primary training script is located at:

```
src/models/train_example_model.py
```

This script is responsible for:

- Loading the dataset  
- Applying preprocessing  
- Performing feature engineering  
- Splitting into train/test sets  
- Training a `RandomForestClassifier`  
- Generating evaluation metrics  
- Serializing the trained model  
- Saving metadata for governance and observability  

---

### 5.2 Dataset

The example uses the **Iris dataset**, a clean, classic dataset suitable for demonstrating:

- Classification  
- Feature transformations  
- Metrics evaluation  
- End-to-end ML workflows  

The pipeline is designed to be easily replaced with a custom dataset.

---

### 5.3 Preprocessing and Feature Engineering

Preprocessing functions are defined in:

```
src/ml/preprocessing.py
```

Feature engineering utilities are in:

```
src/ml/features.py
```

These modules handle:

- Cleaning numeric inputs  
- Adding engineered features (e.g., sepal area, petal area)  
- Converting data into model-ready arrays  

This design ensures that both training and inference use **identical transformation logic**.

---

### 5.4 Model Training Process

The training pipeline performs the following steps:

1. **Load Dataset**  
2. **Apply Feature Engineering**  
3. **Split Into Train/Test Sets**  
4. **Train RandomForest Model**  
5. **Evaluate Model Metrics**  
6. **Save Model Artifact (`model.pkl`)**  
7. **Write Metrics & Metadata (`metrics.json`)**  
8. **Log Training Events**

The process ensures reproducibility and observability.

---

### 5.5 Evaluation Metrics

Model performance metrics are computed using utilities from:

```
src/utils/metrics.py
```

Metrics include:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

A summary is saved to:

```
src/models/metrics.json
```

These metrics support future governance, monitoring, drift detection, or MLflow integration.

---

### 5.6 Model Serialization

The trained model is serialized using Pickle:

```
src/models/model.pkl
```

This artifact is then:

- Loaded at FastAPI startup  
- Cached to avoid repeated loading  
- Versioned and stored for reproducibility  

The codebase also includes placeholders to support future ONNX export.

---

### 5.7 Running the Training Script

To train the model:

```
python -m src.models.train_example_model
```

This will generate:

- `model.pkl` — serialized classifier  
- `metrics.json` — evaluation statistics  
- Console logs showing training progress  

---

### 5.8 Integration With Databricks

For enterprise environments using Databricks:

- Model training can be migrated to notebooks under `databricks/`
- Artifacts exported to storage (Blob, S3, GCS)
- Only serialized artifacts need to be deployed with the API

This separates research workflows from production deployment.

---

### 5.9 Summary

The training subsystem is:

- Modular  
- Reproducible  
- Cloud-ready  
- Governance-friendly  
- Fully aligned with enterprise MLOps best practices  

It demonstrates a real-world ML lifecycle from dataset → preprocessing → training → evaluation → artifact generation → deployment.
### 6. API Usage

This section describes how to interact with the FastAPI service, including available endpoints, request/response formats, and example usage for both machine learning and RAG-based inference.

---

### 6.1 Base URL

When running locally:

```
http://127.0.0.1:8000
```

OpenAPI / Swagger docs:

```
http://127.0.0.1:8000/docs
```

---

### 6.2 Endpoints Overview

| Method | Endpoint      | Description                                   |
|--------|----------------|-----------------------------------------------|
| GET    | `/health`      | Health check for the API                      |
| POST   | `/predict`     | Runs classical ML inference                   |
| POST   | `/rag`         | Runs Retrieval-Augmented Generation workflow  |

---

### 6.3 `/health` Endpoint

**Description:**  
Confirms that the service is running and responsive.

**Example Request:**

```
GET /health
```

**Example Response:**

```
{
  "status": "ok"
}
```

---

### 6.4 `/predict` — Classical ML Prediction

**Description:**  
Runs inference using the trained RandomForest model.

#### **Request Body Format**

```
{
  "feature1": <float>,
  "feature2": <float>,
  "feature3": <float>
}
```

#### **Example Request**

```
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"feature1": 5.1, "feature2": 3.5, "feature3": 1.4}'
```

#### **Example Response**

```
{
  "prediction": 0,
  "probability": 0.97
}
```

#### **Notes**

- Uses the serialized model loaded during startup
- Validates input using Pydantic
- Returns probabilistic confidence score
- Logs request and output for governance and debugging

---

### 6.5 `/rag` — Retrieval-Augmented Generation

**Description:**  
Performs question answering using a knowledge base + LLM.

#### **Request Body Format**

```
{
  "question": "<your natural language question>"
}
```

#### **Example Request**

```
curl -X POST http://127.0.0.1:8000/rag   -H "Content-Type: application/json"   -d '{"question": "What is model governance?"}'
```

#### **Example Response**

```
{
  "answer": "Model governance refers to ...",
  "sources": [
    "AI Governance Basics",
    "Model Monitoring"
  ]
}
```

#### **Notes**

- Uses keyword-based retrieval over `knowledge_base.json`
- Builds a contextual prompt for the LLM
- Supports OpenAI, Azure OpenAI, and compatible APIs
- If no API key is provided, uses deterministic offline mode

---

### 6.6 Error Responses

The API provides structured error messages using Pydantic response models.

#### **Example Input Validation Error**

```
{
  "detail": [
    {
      "loc": ["body", "feature1"],
      "msg": "value is not a valid float",
      "type": "type_error.float"
    }
  ]
}
```

---

### 6.7 Interactive Testing

Use the built‑in Swagger UI to test endpoints interactively:

```
http://127.0.0.1:8000/docs
```

Features include:

- Live request/response preview
- Automatic schema validation
- Form‑based input for endpoints

---

### 6.8 Summary

The FastAPI service:

- Hosts both classical ML and RAG workflows  
- Provides typed, validated, auditable API interactions  
- Enables integration with UIs, microservices, and cloud applications  

This API layer acts as the central orchestration point for the entire AI system.
### 7. UI Usage

The project includes a Streamlit-based user interface that allows interactive testing of both the classical ML model and the RAG (Retrieval-Augmented Generation) system.

---

### 7.1 Launching the UI

To start the Streamlit interface:

```
streamlit run src/ui/streamlit_app.py
```

The UI will open in your default browser at:

```
http://localhost:8501
```

---

### 7.2 UI Layout

The interface is divided into **two primary sections**:

1. **ML Prediction Panel**
2. **RAG Question-Answer Panel**

Each panel interacts with the FastAPI backend using REST API calls.

---

### 7.3 ML Prediction Panel

This section allows users to input numeric features and receive predictions from the trained RandomForest model.

#### **Inputs**

Users must supply three floating-point values:

- `feature1`
- `feature2`
- `feature3`

These fields map directly to the model’s expected input vector.

#### **Process**

1. User enters feature values.
2. Clicks **Predict**.
3. UI sends POST request to:

```
/predict
```

4. Response displays:

- Predicted class label  
- Probability score  

#### **Example Output**

```
Prediction: 0
Confidence: 0.97
```

---

### 7.4 RAG Question-Answer Panel

This section enables natural language interaction with the RAG system.

#### **Inputs**

A single text field:

```
Your question about AI, ML, governance, risk, etc.
```

#### **Process**

1. User asks a question.
2. UI sends POST request to:

```
/rag
```

3. RAG layer:
   - Retrieves relevant documents
   - Builds contextualized prompt
   - Sends prompt to LLM (or offline fallback)
   - Returns answer + sources

#### **Example Output**

```
Answer:
Model governance provides transparency and oversight...

Sources:
- AI Governance Basics
- Model Monitoring
```

---

### 7.5 Error Handling & Feedback

The UI includes:

- Graceful error messages  
- Input validation  
- Automatic refresh on failed API calls  
- Clear indication of offline LLM fallback mode  

Errors from the FastAPI backend will be displayed in the UI.

---

### 7.6 Integration With API Layer

All UI actions call the backend API:

- `POST /predict`
- `POST /rag`

This ensures:

- UI and backend remain decoupled  
- The system is ready for microservices or distributed deployment  
- UI can be replaced by any frontend (React, Angular, etc.)  

---

### 7.7 Summary

The Streamlit UI provides a simple but powerful way to:

- Test classical ML inference  
- Interact with a RAG-enabled LLM pipeline  
- Validate backend API behavior  
- Demonstrate end-to-end AI workflows  

It is fully extensible and production-friendly for internal dashboards or proof-of-concept interfaces.

---
## 8. Deployment Guides

This section provides deployment examples for running the AI + ML Governance Demo across multiple environments, including Docker, Azure, AWS, GCP, and Kubernetes. Each guide is production-oriented but simple enough for fast setup.

---

## 8.1 Deploying with Docker

Docker is the recommended starting point for running the FastAPI service in a containerized environment.

### **Build the Docker Image**

```
docker build -t ai-governance-demo:latest -f docker/Dockerfile .
```

### **Run the Container**

```
docker run -p 8000:8000 ai-governance-demo:latest
```

The service is now live at:

```
http://127.0.0.1:8000/docs
```

### **Optional: Add Streamlit in Separate Container**

```
docker build -t streamlit-ui -f docker/Dockerfile.streamlit .
docker run -p 8501:8501 streamlit-ui
```

---

## 8.2 Deploying to Microsoft Azure

You can deploy the FastAPI service using **Azure Container Apps**, **App Service**, or **AKS**. Below is the simplest option: **Azure Container Apps**.

### **Step 1 — Push Image to Azure Container Registry**

```
az acr build --registry <ACR_NAME> --image ai-governance-demo:latest .
```

### **Step 2 — Deploy to Container Apps**

```
az containerapp create   --name ai-governance-demo   --resource-group <RG>   --image <ACR_NAME>.azurecr.io/ai-governance-demo:latest   --target-port 8000   --ingress external
```

You will get an external HTTPS endpoint URL.

---

## 8.3 Deploying to AWS

Recommended service: **AWS ECS Fargate** (serverless containers).

### **Step 1 — Push to ECR**

```
aws ecr create-repository --repository-name ai-governance-demo
aws ecr get-login-password | docker login --username AWS --password-stdin <AWS_ECR_URI>

docker build -t ai-governance-demo .
docker tag ai-governance-demo:latest <AWS_ECR_URI>/ai-governance-demo:latest
docker push <AWS_ECR_URI>/ai-governance-demo:latest
```

### **Step 2 — Deploy to ECS Fargate**

Create a Fargate Task Definition with:

- Container port: 8000  
- Memory: 1–2 GB  
- CPU: 0.5–1 vCPU  

### **Step 3 — Run Service**

Expose service via:

- ALB (Application Load Balancer), or  
- Public Fargate endpoint  

---

## 8.4 Deploying to Google Cloud (GCP)

Recommended path: **Cloud Run** (serverless container hosting).

### **Step 1 — Build and Push Image**

```
 gcloud builds submit --tag gcr.io/<PROJECT_ID>/ai-governance-demo
```

### **Step 2 — Deploy on Cloud Run**

```
gcloud run deploy ai-governance-demo   --image gcr.io/<PROJECT_ID>/ai-governance-demo   --platform managed   --region us-central1   --allow-unauthenticated
```

Cloud Run will output a public HTTPS endpoint.

---

## 8.5 Deploying to Kubernetes (Any Cloud)

You can deploy to any Kubernetes cluster: AKS, EKS, GKE, DigitalOcean, or local Minikube.

### **Deployment Manifest (example)**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-governance-demo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-governance-demo
  template:
    metadata:
      labels:
        app: ai-governance-demo
    spec:
      containers:
      - name: api
        image: ai-governance-demo:latest
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: ai-governance-service
spec:
  type: LoadBalancer
  selector:
    app: ai-governance-demo
  ports:
  - port: 80
    targetPort: 8000
```

### **Apply Manifest**

```
kubectl apply -f k8s/deployment.yaml
```

### **Scaling**

```
kubectl scale deployment ai-governance-demo --replicas=5
```

---

## 8.6 Summary

This project supports deployment to:

- Docker (single-container local development)  
- Azure (Container Apps, AKS)  
- AWS (ECR + ECS Fargate)  
- GCP (Cloud Run)  
- Kubernetes clusters (any cloud provider)

The codebase is designed so backend + UI components can be deployed independently, enabling true enterprise-grade microservice scaling.

---
## 9. Governance + Safety

This section outlines how the system incorporates governance, model risk controls, safety mechanisms, provenance tracking, and observability hooks.  
Even though this is a demo project, the design intentionally mirrors real-world enterprise AI governance architectures.

---

## 9.1 Model Risk Scoring

The architecture supports the integration of a simple but extensible **Model Risk Scoring Framework**.

### **Risk Factors Considered**
- **Model type** (RandomForest, LLM, RAG)
- **Use-case sensitivity** (high-risk vs low-risk applications)
- **Data quality** (missing values, drift signals)
- **Model drift** (input drift, prediction drift)
- **Confidence levels** (variance, low-confidence thresholds)
- **Operational metadata** (latency, failure events)

### **Example Scoring Dimensions**
| Dimension | Scored Range | Example Factors |
|----------|--------------|-----------------|
| Data Quality | 0–5 | Missingness, schema violations |
| Model Robustness | 0–5 | Stability, overfitting indicators |
| Transparency | 0–5 | Explainability features available |
| Deployment Risk | 0–5 | SLAs, monitoring coverage |
| Security | 0–5 | API hardening, access policies |

A total score categorizes the model as:

- **Low Risk** (0–7)  
- **Medium Risk** (8–15)  
- **High Risk** (16+)  

The project provides a natural integration point to plug in a full enterprise risk engine.

---

## 9.2 Provenance Hooks (Model + Data + LLM)

The system includes multiple provenance and traceability touch points.

### **Training-Time Provenance**
Stored when training the model in Databricks notebooks:

- Training dataset hash
- Feature schema snapshot
- Hyperparameters + model configuration
- Model version (semantic versioning)
- Training timestamp
- Library versions (Python, sklearn)

These are meant to be exported to:

- MLflow  
- AzureML Model Registry  
- S3 versioned buckets  
- or any enterprise model registry  

### **Inference-Time Provenance**
Each request through FastAPI logs:

- Timestamp
- Input schema and values (sanitized)
- Model version used
- Prediction output
- Confidence score
- Latency
- correlation_id
- Optional: user identity/context

These logs are structured JSON for downstream compliance pipelines.

---

## 9.3 Observability: Logging + Monitoring

Monitoring is built around the **Loguru** logging system to provide:

- Structured JSON logs  
- Rotating log files  
- Context-bound logger for each request  
- Rich error detail (tracebacks, stack traces)

### **Common Monitored Signals**
- Request latency  
- Prediction error rates  
- LLM response time  
- Token usage (if using OpenAI/Azure)  
- Model exception frequency  
- RAG retrieval quality  
- Knowledge base coverage  

### **Future Integrations**
The logging system is compatible with:

- **Azure Monitor**
- **AWS CloudWatch**
- **GCP Stackdriver**
- **Elastic / Kibana**
- **Prometheus + Grafana**

Metrics can be exported from `src/utils/metrics.py`.

---

## 9.4 Safety Controls (Classical ML + LLM Safety)

### **Classical ML Safety Guards**
- Confidence thresholds  
- Out-of-range input detection  
- Schema validation via Pydantic  
- Drift hooks for future detectors  
- Audit logs for every prediction  

### **RAG + LLM Safety Features**
- Restricted system prompt controlling behavior  
- Retrieval-augmented context grounding  
- Source attribution for transparency  
- Optional offline deterministic mode  
- Guards against hallucination by anchoring responses  
- API key validation and secure handling  

### **Additional Enterprise Controls (Extendable)**
- Rate limiting  
- Sensitive data filtering  
- Policy-as-code validation  
- Guardrail prompts  
- LLM output classifiers  

---

## 9.5 Governance Alignment (Enterprise Standards)

This project intentionally mirrors the structure used in:

- **ISO/IEC 42001 — AI Management Systems**
- **NIST AI Risk Management Framework (AI RMF)**
- **EU AI Act expectations for risk controls**
- **SOC 2 + HIPAA observability expectations**
- **Azure Responsible AI Toolkit patterns**

### Demonstrated Governance Principles:
| Principle | Demonstrated Via |
|----------|------------------|
| Accountability | Structured logging + version tracking |
| Transparency | Source attribution + interpretability scaffolding |
| Data Governance | Preprocessing, schema validation |
| Security | API hardening, environment variable controls |
| Reliability | Health checks, testing, microservice boundaries |
| Safety | RAG grounding, LLM guardrails, risk scoring |

---

## 9.6 Summary

This demo is engineered to demonstrate how AI systems can be built with:

- **Traceability**  
- **Accountability**  
- **Risk visibility**  
- **Operational monitoring**  
- **LLM safety constraints**  
- **Model lifecycle governance**  

All major components—ML model, RAG pipeline, LLM client, UI, and infrastructure—feed into a cohesive governance-ready architecture.

---
## 10. Conclusion & Next Steps

This project demonstrates a complete, enterprise‑grade example of how **classical machine learning**, **modern LLM‑based RAG systems**, and **MLOps governance practices** can be combined into a cohesive, production‑ready architecture.

---

## 10.1 What This Demo Achieves

This system shows how to operationalize AI responsibly by integrating:

### **✔ Classical ML Model Serving**
- Fully serialized model pipeline  
- FastAPI inference endpoint  
- Probabilities, schema validation, logging  

### **✔ RAG (Retrieval-Augmented Generation) Workflow**
- Local knowledge base retrieval  
- Prompt construction  
- LLM inference with fallback offline mode  
- Source attribution  

### **✔ Full UI Integration**
- Streamlit-based front-end  
- Live interaction with prediction and RAG endpoints  

### **✔ Governance & Safety Controls**
- Logging, traceability, observability  
- Risk scoring scaffolding  
- Provenance metadata  
- Safety guardrails for LLMs  

### **✔ Cloud‑Ready Deployment**
Deploys easily to:
- Docker  
- Azure  
- AWS  
- GCP  
- Kubernetes  

---

## 10.2 Recommended Enhancements

To evolve this into a full production enterprise platform:

### **🔧 Model Enhancements**
- Add ONNX exports for model optimization  
- Introduce drift detection (data + prediction drift)  
- Add explainability (SHAP, LIME)  

### **🧠 RAG Improvements**
- Replace keyword retriever with vector embeddings  
- Add Milvus, Pinecone, Qdrant, or Azure AI Search  
- Introduce synthetic data evaluation for grounding  

### **🔐 Governance & Observability**
- Connect logs to ELK / Azure Monitor  
- Implement request correlation IDs end‑to‑end  
- Add policy-as-code (OPA / Oso / custom engine)  
- Add audit log export pipelines  

### **⚙️ DevOps & MLOps**
- Implement CI/CD  
- Add unit tests + integration tests  
- Train models in Databricks with MLflow tracking  
- Register model versions in a Model Registry  

---

## 10.3 Intended Value

This project serves as a **portfolio‑grade demonstration** of Dr. Freeman Jackson ability to build:

- High-quality ML/LLM solutions  
- Enterprise-ready architecture  
- Governed and safe AI systems  
- Real-world pipelines for production environments  

It is designed to position you strongly for roles in:

- AI/ML Engineering  
- MLOps  
- Generative AI Engineering  
- AI Governance  
- Applied Machine Learning Architecture  

---

## 10.4 Final Summary

This repository shows how to design, implement, deploy, and govern AI systems in a modern cloud-native environment.  
You now have an extensible foundation that demonstrates:

**🔹 Technical excellence**  
**🔹 Architecture discipline**  
**🔹 AI safety + governance awareness**  
**🔹 Production mindset**  

Use this project as a launching point for interviews, demos, technical discussions, and future expansion into a full enterprise AI platform.

---
## 11. Testing & Quality Assurance (QA)

This section details the testing philosophy, structure, and recommended practices for validating the AI + ML Governance Demo system.  
While this is a lightweight portfolio project, the testing design follows **enterprise-grade QA patterns** suitable for production ML/LLM systems.

---

## 11.1 Testing Strategy Overview

A complete AI system requires testing at multiple layers:

### **✔ Unit Tests**
Validate individual functions (e.g., preprocessing, feature engineering, retriever scoring).

### **✔ Integration Tests**
Confirm that components communicate correctly:
- FastAPI ↔ ML model  
- FastAPI ↔ RAG pipeline  
- UI ↔ API  

### **✔ Model Validation Tests**
Evaluate:
- Accuracy, precision, recall  
- Model drift signals  
- Schema consistency  

### **✔ LLM / RAG Evaluation**
Since LLM outputs are non-deterministic:
- Use deterministic offline fallback mode for automated testing  
- Validate retrieval correctness  
- Compare grounding quality  

### **✔ Governance & Safety Validation**
Ensure:
- Logs are generated  
- Inputs are validated  
- Outputs include required metadata  
- API rejects malformed requests  

---

## 11.2 Test Directory Structure

A recommended layout:

```
tests/
  unit/
    test_preprocessing.py
    test_features.py
    test_retriever.py
  integration/
    test_api_predict.py
    test_api_rag.py
  model/
    test_model_loading.py
    test_model_accuracy.py
  governance/
    test_logging.py
    test_schema_validation.py
```

---

## 11.3 Running Tests

Install test dependencies:

```
pip install pytest
```

Run the entire suite:

```
pytest -vv
```

Run a specific folder:

```
pytest tests/unit -vv
```

Run with coverage:

```
pytest --cov=src --cov-report=term-missing
```

---

## 11.4 Example Unit Test

**tests/unit/test_retriever.py**

```python
from src.generative.retriever import retrieve_documents

def test_retriever_finds_relevant_docs():
    kb = [
        {"title": "AI Governance", "content": "Risk, oversight, transparency."},
        {"title": "Cats", "content": "Cats are animals."}
    ]

    results = retrieve_documents("governance", kb, top_k=1)

    assert len(results) == 1
    assert results[0]["title"] == "AI Governance"
```

---

## 11.5 Example Integration Test for FastAPI

**tests/integration/test_api_predict.py**

```python
from fastapi.testclient import TestClient
from src.api.fastapi_app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "feature1": 5.1,
        "feature2": 3.5,
        "feature3": 1.4
    })

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "probability" in body
```

---

## 11.6 Model Quality Assurance

Model QA should include:

### **Performance validation**
- Accuracy  
- Precision / recall  
- F1 score  

### **Robustness**
- Test extreme values  
- Test noisy input  
- Validate predictions remain stable  

### **Versioning**
Record:
- Model artifact hash  
- Training dataset hash  
- sklearn version  

---

## 11.7 RAG/LLM Quality Assurance

### **Deterministic Testing Mode**
When no API key is present, the LLM client returns predictable outputs, enabling reproducible tests.

### **Retrieval Tests**
Ensure:
- Relevant documents are selected  
- Stopwords are ignored  
- Context construction is correct  

### **Prompt Verification**
Confirm:
- System prompt is consistent  
- Retrieved context is injected correctly  

---

## 11.8 Governance QA

Validate that:

- Logs include required metadata  
- Errors generate rich diagnostic logs  
- Correlation IDs appear consistently  
- API rejects malformed or unsafe inputs  
- Output fields comply with governance checklists  

---

## 11.9 Summary

This section provides a template for creating a robust, enterprise-grade QA strategy that ensures:

- Model correctness  
- API reliability  
- LLM safety  
- Governance compliance  
- Reproducibility  

Implementing these practices transforms this portfolio project into a production-ready AI engineering foundation.
---
## 12. Roadmap & Future Enhancements

This section outlines the long-term development roadmap for evolving the AI + ML Governance Demo into a mature, production-grade enterprise platform.  
These enhancements represent common next steps in organizations scaling AI responsibly, integrating governance, and expanding system capabilities.

---

## 12.1 Short-Term Enhancements (0–30 Days)

These improvements deliver immediate value and strengthen the system’s core capabilities.

### **🔧 ML & RAG Enhancements**
- Add ONNX export for optimized model serving  
- Replace keyword retriever with embeddings-based semantic search  
- Integrate a vector database (Milvus, Pinecone, Qdrant, or Azure AI Search)  
- Expand the knowledge base with additional documents  

### **🖥 UI Enhancements**
- Add multi-page Streamlit navigation  
- Include charts for prediction probability distribution  
- Add debugging panel for RAG pipeline visualization  

### **⚙ API & Backend**
- Add pagination for logs  
- Add caching layer for repeated RAG calls  
- Introduce feature flags for ML/LLM switching  

---

## 12.2 Medium-Term Enhancements (1–3 Months)

These changes evolve the demo into a robust enterprise AI platform.

### **🤖 ML Pipeline Improvements**
- Add drift detection (input drift, prediction drift)  
- Add SHAP or LIME for interpretability  
- Experiment tracking with MLflow  
- Automated hyperparameter search (Optuna or Hyperopt)  

### **🧠 Advanced RAG Capabilities**
- Add LLM function-calling support  
- Cross-encoder re-ranking for improved retrieval  
- Support multiple knowledge bases  
- Implement document chunking + embedding pipeline  

### **📊 Observability & Logging**
- Route all logs to ELK / Azure Monitor  
- Add Prometheus metrics exporter  
- Implement distributed tracing (OpenTelemetry)  

### **🔐 Security & Governance**
- Add API authentication with OAuth2 / JWT  
- Implement role-based access control (RBAC)  
- Add Data Loss Prevention (DLP) scanner for inputs  
- Add LLM output filtering and classifier-based moderation  

---

## 12.3 Long-Term Enhancements (3–12 Months)

These items transform the project into a true enterprise-grade AI governance platform.

### **🏗 Platform Architecture**
- Split system into microservices  
- Deploy via CI/CD on Azure DevOps or GitHub Actions  
- Add scalable autoscaling tiers using Kubernetes HPA  

### **🗄 Data & Storage**
- Add a full data lakehouse architecture (Delta Lake)  
- Introduce feature store (Feast or Databricks Feature Store)  
- Add historical prediction logging to a governed storage layer  

### **🔁 Full MLOps Lifecycle**
- Automated retraining pipelines  
- Canary deployments for model updates  
- Shadow deployment testing for new model versions  

### **📚 Governance as Code**
- Integrate policy-as-code using OPA or custom rules engine  
- Add audit trail dashboards  
- Integrate risk scoring into all model and LLM events  
- Track provenance across training, inference, and RAG  
- Add NIST AI RMF + ISO 42001 compliance mappings  

### **🎛 AI Safety Expansion**
- Add red-teaming framework  
- Automated toxicity, bias, and hallucination detection  
- Reinforcement alignment workflows for LLM guardrails  

---

## 12.4 Future AI Agent Capabilities

Once the system matures, add multi-agent orchestration:

- Use LangChain, LangGraph, or Microsoft AutoGen  
- Add task planning + memory systems  
- Build workflow agents (research, governance, summarization)  
- Add agentic evaluation + safety layers  

This extends the system into modern, agent-driven architectures used in advanced generative AI systems.

---

## 12.5 Vision: The Fully Featured Enterprise AI Platform

If fully developed, this project becomes a real-world AI governance platform with:

- **Classical ML inference + LLM capabilities**  
- **Full provenance and accountability layer**  
- **Risk & governance scoring integrated with every event**  
- **Scalable microservices architecture with security controls**  
- **Databricks-native training workflows**  
- **Policy-as-code guardrails for responsible AI**  

This serves as a solid foundation for:

- Enterprise AI teams  
- AI governance & compliance groups  
- ML engineers & platform engineers  
- R&D and safety evaluation teams  

---

## 12.6 Summary

Section 12 provides a clear roadmap for turning this portfolio project into a full enterprise platform incorporating:

- Scalable infrastructure  
- Deep MLOps  
- Responsible AI guardrails  
- Multi-agent generative AI systems  
- Full governance and compliance lifecycle  

