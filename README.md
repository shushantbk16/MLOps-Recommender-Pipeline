# 🚀 MLOps Recommender Pipeline: Nexus News Feed

### End-to-End Hybrid Recommendation System Orchestrated with Docker & Kubernetes

---

## 💡 Executive Summary

This project implements a **production-grade, two-stage hybrid recommendation system** designed to solve cold-start and scalability challenges for a high-volume news platform (MIND dataset).

The system uses advanced deep learning (SBERT) and collaborative filtering (LightFM) and blends their results using a powerful ranker (XGBoost) to deliver personalized content. The entire project is packaged as a **containerized, auditable MLOps solution** ready for cloud deployment via Kubernetes.

---

## 🛠️ Technology Stack & Architecture

| Category | Tools Used | Key Architectural Concept |
| :--- | :--- | :--- |
| **Orchestration & Deployment** | **Docker**, **Kubernetes (K8s)**, GitHub Actions | Multi-Stage Build, Containerization, Scalable Pod Definition, CI/CD Automation. |
| **Retrieval (Stage 1)** | **SBERT** (Sentence-BERT), **FAISS** (Facebook AI Similarity Search), **LightFM** | Blends content-based vector search (ANN) with matrix factorization (Collaborative Filtering). |
| **Ranking (Stage 2)** | **XGBoost** | Final **Learning-to-Rank** model that blends the two retrieval scores for precision. |
| **MLOps & Data** | **MLflow**, Git LFS, `pandas`/`pyarrow` | Experiment tracking, model versioning, and massive data handling. |
| **Demo Interface** | **Streamlit** | Interactive UI for visual demonstration of model logic. |

---

## 📈 Achievements & Technical Insights

This project demonstrates expertise in solving real-world production challenges:

| Achievement Area | Insight | MLOps Proof |
| :--- | :--- | :--- |
| **Scalability & K8s** | Authored **Kubernetes Manifests** defining **2 replicas (HA)** and setting explicit **resource limits (`memory: 4Gi`, `cpu: 2`)** to manage the large memory footprint of the XGBoost/SBERT models in a production cluster. | `k8s/*.yaml` committed. |
| **Cold-Start Solution** | Solved **Item Cold-Start** using SBERT/FAISS and **User Cold-Start** (after 1 click) by training the XGBoost ranker to rely on the non-zero SBERT score when the LightFM score is zero (New User). | **Demo 2** visually confirms this logic. |
| **Container Optimization** | Successfully built a **Multi-Stage Docker Image** for the 3.2 GB project, solving complex C++ compilation failures (`lightfm`/`faiss`) and pushing the image to Docker Hub. | Image built with optimized layers. |
| **Data Efficiency** | Converted a multi-GB dataset (`behaviors.tsv`) to lightweight `.parquet` and implemented **memory-safe Python generators** for feature engineering, preventing system-level crashes (`zsh: killed`). | Scripts `03` and `04` use memory-safe iteration. |

---

## 🚀 How to Run the Project (Showcase)

### 1. **Run the Full Containerized Application (Proof of Docker)**

You can pull the published image from Docker Hub and run the app locally.

1.  **Pull the Image:**
    ```bash
    docker pull shushantbk16/mlops-recommender-pipeline:latest
    ```

2.  **Run the Container:**
    ```bash
    # Runs the container and maps internal port 8501 to host port 8501
    docker run -d -p 8501:8501 --name recommender-app shushantbk16/mlops-recommender-pipeline
    ```
3.  **Access:** Open your browser to `http://localhost:8501`.

### 2. **The Deployment Blueprint (Kubernetes)**

The final deployable architecture is defined in the `k8s/` directory. These files are ready to be executed on any cloud cluster:

```bash
# Apply the Deployment (runs the app)
kubectl apply -f k8s/deployment.yaml

# Apply the Service (creates the public LoadBalancer endpoint)
kubectl apply -f k8s/service.yaml
```

---

### 📂 Repository Structure

```
.
├── .github/                   # GitHub Actions CI/CD workflow
├── k8s/                       # Kubernetes deployment and service manifests
├── data/models/               # Final, trained models (.pkl, .index, etc.)
├── demo.py                    # Streamlit Interactive Demo Application (Main Entry Point)
├── Dockerfile                 # Multi-stage Docker build recipe
└── requirements.txt           # Python dependencies list
```
