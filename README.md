# Autonomous Financial Crime Intelligence Platform

An end-to-end fraud intelligence system that combines machine learning, graph intelligence, explainability, decision automation, investigator search, and local RAG support with Ollama.

## What It Does

- Scores suspicious accounts with ML and rule-based signals
- Detects fraud rings and connected risky accounts using graph intelligence
- Generates human-readable explanations for each case
- Recommends automated actions such as freeze, escalate, or limit reduction
- Estimates financial impact and investigation priority
- Supports investigator queries through a retriever + knowledge store layer
- Answers grounded questions locally with Ollama `phi3`

## Project Structure

- `main.py` - pipeline entrypoint
- `app_gradio.py` - Gradio frontend for the same fraud intelligence workflow
- `src/feature_engineering.py` - transaction feature generation
- `src/model.py` - model training and prediction
- `src/suspicious_window.py` - suspicious period detection
- `src/graph_intelligence.py` - graph/community intelligence
- `src/decision_engine.py` - action recommendation engine
- `src/explainability.py` - explanation generation
- `src/impact_metrics.py` - case impact calculations
- `src/investigator_ai.py` - investigator query layer
- `src/retriever_layer.py` - retrieval-ready case search
- `src/knowledge_store.py` - vector index / knowledge store
- `src/ollama_rag.py` - local RAG answer layer
- `src/schema_cleaner.py` - CSV cleanup utility

## Setup

### 1. Install dependencies

Use your preferred Python environment and install the project requirements.

### 2. Install Ollama

Install Ollama for Windows from:

- https://ollama.com/download/windows

Then pull the local model:

```powershell
ollama pull phi3
```

### 3. Run the pipeline

From the project root:

```powershell
python main.py
```

This generates the final `data/submission.csv` and saves the trained models and artifacts used by the pipeline.

### 4. Clean the submission schema

If you want the final CSV to contain only canonical graph/community columns:

```powershell
python src\schema_cleaner.py
```

### 5. Run terminal-only mode

If you want everything in the terminal, use:

```powershell
python terminal_app.py
```

This mode:

- loads `data/submission.csv`
- lets you pick any account
- prints the full case detail and explainability
- runs the Ollama-backed investigator/RAG query flow
- shows top accounts and overview stats in the terminal

### 6. Run the Gradio frontend

If you want the browser UI rebuilt in Gradio, run:

```powershell
python app_gradio.py
```

The app will automatically open in your browser at `http://127.0.0.1:7860` or `http://localhost:7860`.

This reads `data/submission.csv`, keeps the same case-selection and investigator flow, and uses the same Ollama-backed RAG layer for queries.

**Note:** Make sure you've run `python main.py` first to generate the submission.csv file.

<img width="1898" height="911" alt="Screenshot 2026-06-07 170816" src="https://github.com/user-attachments/assets/0272ae0b-e35c-45b6-b2e0-1b417547c6a9" />

<img width="1881" height="801" alt="Screenshot 2026-06-07 182808" src="https://github.com/user-attachments/assets/5ba97605-f82e-4aaa-8acf-2a5937e03454" />

<img width="1896" height="911" alt="Screenshot 2026-06-07 183017" src="https://github.com/user-attachments/assets/61e948ed-cc9c-487f-a480-53356b4894d7" />

<img width="1882" height="597" alt="Screenshot 2026-06-07 183046" src="https://github.com/user-attachments/assets/1d4e8173-c7aa-40d0-9733-4ea23a39b22e" />

## Demo Video (2 minutes 35 seconds)

https://drive.google.com/file/d/1vsvz0OrsOzojo34Aa3uSIh-1tbO3f5yu/view?usp=drivesdk



## Final Output

The main deliverable is `data/submission.csv`, which includes:

- risk and action fields
- explanation fields
- graph/community intelligence
- impact metrics
- investigator-ready retrieval text
- retriever and knowledge-store readiness flags

## Architecture Summary

```text
Transaction Data
    -> Feature Engineering
    -> ML Scoring
    -> Suspicious Window Detection
    -> Graph Intelligence
    -> Decision Engine
    -> Explainability
    -> Impact Metrics
    -> Investigator Layer
    -> Retriever + Knowledge Store
    -> Ollama RAG Assistant
    -> Final Submission Output
```

### Layer Breakdown

- `Feature Engineering`: builds behavioral transaction features and graph-ready context
- `Modeling`: predicts fraud risk and produces the base score
- `Suspicious Window`: detects abnormal activity periods and burst behavior
- `Graph Intelligence`: identifies connected risky clusters and fraud-ring signals
- `Decision Engine`: recommends operational actions
- `Explainability`: turns model outputs into readable reasons
- `Impact Metrics`: estimates fraud prevented and investigation priority
- `Investigator Layer`: creates case briefs and query-friendly narratives
- `Retriever + Knowledge Store`: finds relevant cases fast for investigator search
- `Ollama RAG`: answers grounded questions using retrieved case context

## Demo Flow

Use this sequence for a live presentation:

1. Open the dashboard overview and show the top KPIs.
2. Click a critical account and show:
   - risk score
   - decision action
   - explanation
   - connected risky accounts
   - impact estimate
3. Show a medium-risk account to demonstrate that not all cases are frozen.
4. Open the investigator panel and ask:
   - `Why is account <id> suspicious?`
   - `Find connected risky users for <id>`
   - `Show top risky accounts`
5. Show the RAG response from Ollama and point out the grounded evidence.
6. Finish by showing that the system links detection to action and investigation.

### Suggested Demo Accounts

- One critical freeze case with a very high risk score
- One medium-risk limit-reduction case
- One account with strong connected-account context for the graph page
- One account for investigator query and grounded response demo

## Recommended Talking Points

- This is not only a fraud detector.
- It is a fraud decision intelligence system.
- It combines prediction, explanation, graph reasoning, and investigation support.
- The RAG layer is grounded in case evidence instead of generic chatbot output.

## Notes

- The current local LLM integration uses `phi3`.
- If you want a stronger local model later, you can switch to `phi3.5`.
- Large generated outputs are stored in compressed format (.gz) to reduce repository size.
  To access the full results, extract submission.csv.gz.
- Dataset used in this project
  https://www.kaggle.com/datasets/abhyudayrbih/rbih-nfpc-phase-2
  

  



