# MLOps / DevOps Agentic RAG Agent

A **Level 3 Agentic RAG** pipeline for MLOps and DevOps Q&A, featuring query rewriting, iterative retrieval, self-reflection, source citation, and fallback handling.

Live demo: [HuggingFace Space — atulkrs/mlops-rag-agent](https://huggingface.co/spaces/atulkrs/mlops-rag-agent)

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│               Level 3 Agentic RAG Pipeline               │
│                                                         │
│  1. Query Rewriting (Flan-T5-base)                      │
│     └─ Reformulates query for better retrieval          │
│                                                         │
│  2. Semantic Retrieval                                  │
│     ├─ Embeddings: all-MiniLM-L6-v2 (384-dim)          │
│     ├─ Vector Store: ChromaDB (cosine similarity)       │
│     └─ Indexing: LlamaIndex (SentenceSplitter)          │
│                                                         │
│  3. Relevance Filtering (Self-Reflection)               │
│     └─ Flan-T5 checks if each chunk is relevant        │
│                                                         │
│  4. Answer Generation (Flan-T5-base)                   │
│     └─ Context-grounded answer synthesis               │
│                                                         │
│  5. Answer Self-Reflection (Flan-T5-base)              │
│     └─ Judges completeness → triggers re-retrieval     │
│                                                         │
│  6. Iterative Retrieval (up to 2 iterations)           │
│     └─ Follow-up queries if answer is inadequate       │
│                                                         │
│  7. Fallback Handling                                   │
│     └─ General answer when no relevant docs found      │
│                                                         │
│  8. Source Citation                                     │
│     └─ Source doc + cosine similarity per chunk        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Gradio UI (Answer + Query Info + Reflection + Citations)
```

## Tech Stack

| Component     | Technology                              |
|---------------|-----------------------------------------|
| Orchestration | LlamaIndex Core                         |
| Vector Store  | ChromaDB (persistent)                   |
| Embeddings    | sentence-transformers/all-MiniLM-L6-v2  |
| Generator     | google/flan-t5-base                     |
| UI            | Gradio                                  |

## Project Structure

```
rag-agent/
├── app.py              # Gradio UI entry point
├── agent.py            # Agentic RAG orchestration loop
├── rag_engine.py       # LlamaIndex + ChromaDB retrieval
├── generator.py        # Flan-T5 query rewriting & generation
├── requirements.txt
└── knowledge_base/
    ├── sagemaker.txt       # AWS SageMaker (training, endpoints, auto scaling, pipelines)
    ├── kubernetes.txt      # Kubernetes (HPA/KEDA, GPU, KServe, Helm)
    ├── cicd.txt            # CI/CD (GitHub Actions, Jenkins, ArgoCD, DVC)
    ├── terraform.txt       # Terraform (HCL, remote state, EKS, SageMaker)
    └── model_monitoring.txt# Drift detection, Evidently, Prometheus/Grafana
```

## Knowledge Base Topics

- **AWS SageMaker** — Training jobs, HPO, real-time endpoints, auto scaling (target tracking, InvocationsPerInstance, cooldown periods), Pipelines, Feature Store, Model Monitor, Clarify
- **Kubernetes** — Pods, Deployments, HPA/KEDA, GPU management, Kubeflow, KServe, Helm
- **CI/CD** — GitHub Actions, Jenkins, ArgoCD/GitOps, DVC, MLflow model registry, testing strategies
- **Terraform** — HCL syntax, remote state, modules, EKS/SageMaker provisioning, Terragrunt
- **Model Monitoring** — Data drift, concept drift, Evidently AI, Alibi Detect, Prometheus/Grafana, Great Expectations

## Running Locally

```bash
git clone https://github.com/atulkrsdevops/huggingface-workshop.git
cd huggingface-workshop/rag-agent
pip install -r requirements.txt
python app.py
```

The app starts a Gradio server at `http://localhost:7860`.

## Environment Variables

| Variable    | Purpose                                          |
|-------------|--------------------------------------------------|
| `HF_TOKEN`  | Hugging Face token — required only when pushing updates to the HF Space via `clean_push.py`. Never hardcode; pass via environment. |

## Example Questions

- How do I configure auto-scaling for a SageMaker inference endpoint?
- What is the difference between data drift and concept drift?
- How do I use Terraform to provision an EKS cluster with GPU nodes?
- Explain how to set up a CI/CD pipeline for model retraining with GitHub Actions.
- How do I deploy a model with canary rollout on Kubernetes using KServe?
- What metrics should I monitor for ML models in production?

## License

MIT
