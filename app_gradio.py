from pathlib import Path
import json
import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
import traceback

import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

from src.investigator_ai import InvestigatorAssistant
# Note: We use our own check_ollama_status() function instead of summarize_ollama_integration
# to avoid accidentally starting Ollama during import


ROOT = Path(__file__).resolve().parent
SUBMISSION_PATH = ROOT / "data" / "submission.csv"


def _request_json(url, method="GET", payload=None, timeout=5):
    """Make HTTP request and return JSON response"""
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_installed_ollama_models():
    """Return installed Ollama model names from the local CLI without starting the service."""
    if shutil.which("ollama") is None:
        return []

    try:
        completed = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            return []

        models = []
        for raw_line in completed.stdout.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("NAME") or line.startswith("MODEL"):
                continue
            match = re.match(r"^([\w\-.:]+)\s+", line)
            if match:
                models.append(match.group(1).strip())
        return list(dict.fromkeys(models))
    except Exception:
        return []


def check_ollama_status(base_url="http://localhost:11434/api", model="phi3", timeout=2):
    """
    Check if Ollama is running by attempting to connect to its API.
    Returns dict with status information.
    
    IMPORTANT: 
    - This function ONLY checks if port 11434 is open and queries the API
    - It does NOT start Ollama or run any system commands
    - On Windows, Ollama may be configured to auto-start when applications try to connect
    - To disable auto-start: Check Windows Services and disable "Ollama" service if present
    """
    import socket as sock
    
    try:
        # First try a simple socket connection to see if port is open
        # This is faster and more reliable than HTTP request
        # NOTE: This check will NOT start Ollama - it only tests if port is already open
        host, port = "localhost", 11434
        sock_test = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        sock_test.settimeout(1)
        result = sock_test.connect_ex((host, port))
        sock_test.close()
        
        if result != 0:
            available_models = _get_installed_ollama_models()
            model_names = set(available_models)
            model_available = model in model_names or any(name.startswith(f"{model}:") for name in model_names)
            preferred_model = model if model_available else (available_models[0] if available_models else model)

            return {
                "ollama_available": bool(available_models),
                "ollama_running": False,
                "ollama_base_url": base_url,
                "ollama_model": model,
                "preferred_model": preferred_model,
                "available_models": available_models,
                "ollama_model_installed": model_available,
                "ollama_model_count": len(available_models),
                "ollama_error": "Port 11434 is not open - Ollama is not running"
            }
        
        # Port is open, now try API call
        try:
            response = _request_json(f"{base_url}/tags", method="GET", timeout=timeout)
            models = response.get("models", [])
            available_models = []
            for item in models:
                if item.get("name"):
                    available_models.append(item.get("name"))
                if item.get("model"):
                    available_models.append(item.get("model"))
            available_models = list(dict.fromkeys(available_models))
        except Exception:
            available_models = _get_installed_ollama_models()
            models = available_models

        model_names = set(available_models)
        model_available = model in model_names or any(name.startswith(f"{model}:") for name in model_names)
        preferred_model = model if model_available else (available_models[0] if available_models else model)

        return {
            "ollama_available": True,
            "ollama_running": True,
            "ollama_base_url": base_url,
            "ollama_model": model,
            "preferred_model": preferred_model,
            "available_models": available_models,
            "ollama_model_installed": model_available,
            "ollama_model_count": len(models),
            "ollama_error": None
        }
    except (urllib.error.URLError, ConnectionRefusedError, OSError) as e:
        # Connection refused or timeout - Ollama is definitely NOT running
        error_msg = str(e.reason) if hasattr(e, 'reason') else str(e)
        return {
            "ollama_available": False,
            "ollama_running": False,
            "ollama_base_url": base_url,
            "ollama_model": model,
            "ollama_model_installed": False,
            "ollama_model_count": 0,
            "ollama_error": f"Connection failed: {error_msg}"
        }
    except socket.timeout:
        # Explicit timeout - Ollama is NOT running
        return {
            "ollama_available": False,
            "ollama_running": False,
            "ollama_base_url": base_url,
            "ollama_model": model,
            "ollama_model_installed": False,
            "ollama_model_count": 0,
            "ollama_error": "Connection timeout - Ollama is not running"
        }
    except Exception as e:
        # Other connection errors - Ollama is NOT running
        return {
            "ollama_available": False,
            "ollama_running": False,
            "ollama_base_url": base_url,
            "ollama_model": model,
            "ollama_model_installed": False,
            "ollama_model_count": 0,
            "ollama_error": f"Connection error: {str(e)}"
        }


class AppState:
    """Centralized state management for the application"""
    def __init__(self):
        self.df = None
        self.assistant = None
        self.ollama_status = None
        self.selected_account = None
        self.load_data()
    
    def load_data(self):
        """Load and normalize data"""
        try:
            if not SUBMISSION_PATH.exists():
                raise FileNotFoundError(f"Missing required file: {SUBMISSION_PATH}")
            
            self.df = pd.read_csv(SUBMISSION_PATH)
            self._normalize_columns()
            self.assistant = InvestigatorAssistant(self.df)
            
            # Get Ollama status using our improved check function
            self.ollama_status = check_ollama_status()
            
            # Set default selected account to highest risk
            if not self.df.empty:
                self.selected_account = str(self.df.sort_values("risk_score", ascending=False).iloc[0]["account_id"])
            
            return True, "Data loaded successfully"
        except Exception as e:
            return False, f"Error loading data: {str(e)}\n{traceback.format_exc()}"
    
    def _normalize_columns(self):
        """Normalize duplicate column names from merges"""
        rename_map = {
            "graph_risk_score_x": "graph_risk_score",
            "graph_risk_score_y": "graph_risk_score",
            "community_risk_score_x": "community_risk_score",
            "community_risk_score_y": "community_risk_score",
            "fraud_ring_flag_x": "fraud_ring_flag",
            "fraud_ring_flag_y": "fraud_ring_flag",
        }
        for old, new in rename_map.items():
            if old in self.df.columns and new not in self.df.columns:
                self.df[new] = self.df[old]
    
    def refresh(self):
        """Refresh data from disk"""
        return self.load_data()
    
    def get_accounts(self):
        """Get list of all account IDs"""
        if self.df is None or self.df.empty:
            return []
        return self.df["account_id"].astype(str).tolist()
    
    def get_top_accounts(self, n=10):
        """Get top N accounts by risk score"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        return self.df.sort_values("risk_score", ascending=False).head(n)


# Global app state
app_state = AppState()


def as_text(value, default="N/A"):
    """Safely convert value to text"""
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def as_money(value):
    """Format value as currency"""
    try:
        return f"INR {float(value):,.2f}"
    except Exception:
        return "N/A"


def as_pct(value):
    """Format value as percentage"""
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def pick_row(account_id):
    """Get row for specific account ID from app state"""
    if app_state.df is None or app_state.df.empty:
        return None
    
    if not account_id:
        return app_state.df.sort_values("risk_score", ascending=False).iloc[0]
    
    match = app_state.df[app_state.df["account_id"].astype(str) == str(account_id)]
    if match.empty:
        return app_state.df.sort_values("risk_score", ascending=False).iloc[0]
    return match.iloc[0]


def format_markdown_case(row):
    """Format account case details as markdown"""
    if row is None:
        return "No data available"
    
    return f"""
### Account: {as_text(row.get("account_id"))}

#### Core Metrics
- **Risk score:** `{row.get("risk_score", "N/A")}`
- **Risk level:** `{as_text(row.get("risk_level"))}`
- **Primary action:** `{as_text(row.get("primary_action"))}`
- **Action priority:** `{as_text(row.get("action_priority"))}`
- **Action owner:** `{as_text(row.get("action_owner"))}`
- **Decision confidence:** `{as_pct(row.get("decision_confidence"))}`

#### Impact Assessment
- **Impact band:** `{as_text(row.get("impact_band"))}`
- **Impact priority score:** `{row.get("impact_priority_score", "N/A")}`
- **Estimated exposure:** `{as_money(row.get("estimated_exposure_inr"))}`
- **Estimated fraud prevented:** `{as_money(row.get("estimated_fraud_prevented_inr"))}`

#### Graph Intelligence
- **Graph risk score:** `{row.get("graph_risk_score", "N/A")}`
- **Community risk score:** `{row.get("community_risk_score", "N/A")}`
- **Community ID:** `{as_text(row.get("community_id"))}`
- **Community size:** `{row.get("community_size", "N/A")}`
- **Fraud ring flag:** `{as_text(row.get("fraud_ring_flag"))}`
- **Influential node flag:** `{as_text(row.get("influential_node_flag"))}`

#### Suspicious Activity Window
- **Start:** `{as_text(row.get("suspicious_start"))}`
- **End:** `{as_text(row.get("suspicious_end"))}`

#### Decision Summary
{as_text(row.get("decision_summary"))}

#### Decision Reasons
{as_text(row.get("decision_reasons"))}

#### Explanation Summary
{as_text(row.get("explanation_summary"))}

#### Human-Readable Reasoning
{as_text(row.get("human_readable_reasoning"))}

#### Connected Risky Accounts
{as_text(row.get("connected_risky_accounts"))}
"""


def overview_panel():
    """Generate overview dashboard HTML"""
    df = app_state.df
    if df is None or df.empty:
        return "<div class='hero'><h1>No data loaded</h1><p>Please run the pipeline to generate data</p></div>"
    
    screened = len(df)
    cases = int((df.get("case_generated", pd.Series(dtype=str)).astype(str).str.upper() == "YES").sum()) if "case_generated" in df.columns else 0
    auto = int((df.get("auto_action_eligible", pd.Series(dtype=str)).astype(str).str.upper() == "YES").sum()) if "auto_action_eligible" in df.columns else 0
    ring = int((df.get("fraud_ring_flag", pd.Series(dtype=str)).astype(str).str.upper() == "YES").sum()) if "fraud_ring_flag" in df.columns else 0
    prevented = pd.to_numeric(df.get("estimated_fraud_prevented_inr", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if "estimated_fraud_prevented_inr" in df.columns else 0
    top = df.sort_values("risk_score", ascending=False).head(1).iloc[0]
    
    case_rate = (cases / screened * 100) if screened > 0 else 0
    auto_rate = (auto / cases * 100) if cases > 0 else 0

    html = f"""
    <div class="hero">
      <div class="eyebrow">Fraud Decision Intelligence Platform</div>
      <h1>Autonomous Financial Crime Intelligence</h1>
      <p>Real-time detection • Graph intelligence • AI explainability • Autonomous decisions</p>
    </div>
    <div class="metrics">
      <div class="metric">
        <span>Accounts Screened</span>
        <b>{screened:,}</b>
      </div>
      <div class="metric">
        <span>Cases Generated</span>
        <b>{cases:,}</b>
        <div style="color: var(--muted); font-size: 0.85rem; margin-top: 4px;">{case_rate:.1f}% detection rate</div>
      </div>
      <div class="metric">
        <span>Auto Actions</span>
        <b>{auto:,}</b>
        <div style="color: var(--success); font-size: 0.85rem; margin-top: 4px;">{auto_rate:.1f}% automated</div>
      </div>
      <div class="metric">
        <span>Fraud Ring Cases</span>
        <b>{ring:,}</b>
      </div>
      <div class="metric">
        <span>Fraud Prevented</span>
        <b>{as_money(prevented)}</b>
      </div>
      <div class="metric">
        <span>Top Risk Account</span>
        <b>{as_text(top.get("account_id"))}</b>
        <div style="color: var(--error); font-size: 0.85rem; margin-top: 4px;">Score: {top.get('risk_score', 0):.1f}</div>
      </div>
    </div>
    """
    return html


def account_detail_panel(account_id):
    """Generate account detail panel"""
    row = pick_row(account_id)
    app_state.selected_account = account_id
    return format_markdown_case(row)


def graph_panel(account_id):
    """Generate graph intelligence panel and visualization"""
    df = app_state.df
    if df is None or df.empty:
        return "<div class='panel-card'><p>No data available</p></div>", None
    
    row = pick_row(account_id)
    app_state.selected_account = account_id
    
    connected = [x.strip() for x in as_text(row.get("connected_risky_accounts"), "").split("|") if x.strip()]
    peers = df.copy()
    if "community_id" in df.columns and as_text(row.get("community_id"), "") != "N/A":
        try:
            peers = peers[df["community_id"] == row.get("community_id")]
        except:
            pass
    peers = peers.sort_values("risk_score", ascending=False).head(10)

    # Create simple bar chart with better layout to avoid overlapping
    bars = px.bar(
        peers,
        x="account_id",
        y="risk_score" if "risk_score" in peers.columns else peers.columns[0],
        title=f"Community Risk Distribution (Community {as_text(row.get('community_id'))})",
        template="plotly_white",
        labels={"account_id": "Account ID", "risk_score": "Risk Score"}
    ) if not peers.empty else go.Figure()
    
    bars.update_layout(
        xaxis_tickangle=-45,
        height=450,  # Increased height to prevent overlap
        showlegend=False,
        margin=dict(t=50, b=80, l=60, r=30),  # More bottom margin for labels
        font=dict(size=10),
        xaxis=dict(
            tickmode='linear',
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            title="Risk Score",
            tickfont=dict(size=10)
        )
    )
    bars.update_traces(
        marker_color='#6366f1',
        text=peers["risk_score"].round(1),  # Show values on bars
        textposition='outside',  # Position text outside bars
        textfont=dict(size=9)
    )
    
    # Determine risk level badge
    graph_risk = row.get("graph_risk_score", 0)
    if isinstance(graph_risk, (int, float)) and graph_risk >= 75:
        risk_badge = '<span class="badge error">HIGH RISK</span>'
    elif isinstance(graph_risk, (int, float)) and graph_risk >= 50:
        risk_badge = '<span class="badge warning">MEDIUM RISK</span>'
    else:
        risk_badge = '<span class="badge success">LOW RISK</span>'

    html = f"""
    <div class="panel-card">
      <div class="section-title">Graph Intelligence: {as_text(row.get("account_id"))}</div>
      <div style="margin-bottom: 16px;">{risk_badge}</div>
      <div class="grid">
        <div><span>Graph risk score</span><b>{row.get("graph_risk_score", "N/A")}</b></div>
        <div><span>Community risk score</span><b>{row.get("community_risk_score", "N/A")}</b></div>
        <div><span>Community ID</span><b>{as_text(row.get("community_id"))}</b></div>
        <div><span>Community size</span><b>{row.get("community_size", "N/A")}</b></div>
        <div><span>Fraud ring flag</span><b>{as_text(row.get("fraud_ring_flag"))}</b></div>
        <div><span>Influential node</span><b>{as_text(row.get("influential_node_flag"))}</b></div>
      </div>
      <div class="subhead">Connected Risky Accounts ({len(connected)} found)</div>
      <div class="chips">{''.join(f'<span class="chip">{c}</span>' for c in connected[:10]) if connected else '<span class="chip">None retrieved</span>'}</div>
    </div>
    """
    return html, bars


def impact_panel(account_id):
    """Generate impact assessment panel"""
    df = app_state.df
    if df is None or df.empty:
        return "<div class='panel-card'><p>No data available</p></div>"
    
    row = pick_row(account_id)
    app_state.selected_account = account_id
    
    # Calculate impact severity
    impact_score = row.get("impact_priority_score", 0)
    if isinstance(impact_score, (int, float)):
        if impact_score >= 75:
            impact_badge = '<span class="badge error">CRITICAL IMPACT</span>'
        elif impact_score >= 50:
            impact_badge = '<span class="badge warning">HIGH IMPACT</span>'
        else:
            impact_badge = '<span class="badge success">MODERATE IMPACT</span>'
    else:
        impact_badge = '<span class="badge info">N/A</span>'
    
    return f"""
    <div class="panel-card">
      <div class="section-title">Impact Assessment: {as_text(row.get("account_id"))}</div>
      <div style="margin-bottom: 16px;">{impact_badge}</div>
      <div class="metrics compact">
        <div class="metric">
          <span>Estimated Exposure</span>
          <b>{as_money(row.get("estimated_exposure_inr"))}</b>
        </div>
        <div class="metric">
          <span>Fraud Prevented</span>
          <b>{as_money(row.get("estimated_fraud_prevented_inr"))}</b>
        </div>
        <div class="metric">
          <span>Impact Priority</span>
          <b>{row.get("impact_priority_score", "N/A")}</b>
        </div>
        <div class="metric">
          <span>Review SLA</span>
          <b>{row.get("review_sla_hours", "N/A")}h</b>
        </div>
        <div class="metric">
          <span>Ring Case Flag</span>
          <b>{as_text(row.get("ring_case_flag"))}</b>
        </div>
        <div class="metric">
          <span>Investigation Queue</span>
          <b>{as_text(row.get("investigation_queue"))}</b>
        </div>
      </div>
      <div class="subhead">Impact Analysis</div>
      <div class="text-block">
        This account has been assessed with {'a ' + as_text(row.get("impact_band", "UNKNOWN")).lower() + ' impact band' if row.get("impact_band") else 'no impact band'}. 
        The estimated financial exposure represents potential fraud magnitude, while fraud prevented indicates the value protected by detection.
        Priority score: <b>{row.get("impact_priority_score", "N/A")}</b> • 
        Action: <b>{as_text(row.get("primary_action", "PENDING"))}</b>
      </div>
    </div>
    """


def investigator_panel(account_id):
    """Generate investigator AI panel"""
    df = app_state.df
    if df is None or df.empty:
        return "<div class='panel-card'><p>No data available</p></div>"
    
    row = pick_row(account_id)
    app_state.selected_account = account_id
    
    brief = as_text(row.get("investigator_brief"))
    kw = [k.strip() for k in as_text(row.get("investigator_keywords"), "").split("|") if k.strip()]
    ret = as_text(row.get("retrieval_text"))[:500] + "..." if len(as_text(row.get("retrieval_text"))) > 500 else as_text(row.get("retrieval_text"))
    
    return f"""
    <div class="panel-card">
      <div class="section-title">Investigator AI: {as_text(row.get("account_id"))}</div>
      
      <div class="subhead">Investigator Brief</div>
      <div class="text-block">{brief}</div>
      
      <div class="subhead">Keywords Analysis ({len(kw)} total)</div>
      <div class="chips">{''.join(f'<span class="chip">{k}</span>' for k in kw[:20]) if kw else '<span class="chip">No keywords</span>'}</div>
      
      <div class="subhead">Retrieval Context Preview</div>
      <div class="text-block" style="max-height: 200px; overflow-y: auto;">{ret}</div>
    </div>
    """


def query_answer(query, account_id):
    """Process RAG query and return formatted answer"""
    df = app_state.df
    if df is None or df.empty:
        return "<div class='panel-card'><p>No data available</p></div>"
    
    if not query or not query.strip():
        return """
        <div class="panel-card">
          <div class="section-title">Investigator Query</div>
          <div class="text-block">Please enter a question above and click Ask.</div>
        </div>
        """
    
    assistant = app_state.assistant
    ollama_status = app_state.ollama_status
    
    # Check if Ollama is actually running
    if not ollama_status.get("ollama_running"):
        row = pick_row(account_id)
        error_message = ollama_status.get('ollama_error', 'Ollama service is not running')
        return f"""
        <div class="panel-card">
          <div class="section-title">Investigator Query</div>
          <div class="alert error">
            <b>❌ Ollama Status:</b> Not running<br>
            <b>Model:</b> {ollama_status.get('ollama_model', 'phi3')}<br>
            <b>Error:</b> {error_message}<br><br>
            <b>💡 To use RAG queries:</b><br>
            1. Start Ollama: <code>ollama serve</code><br>
            2. Install model: <code>ollama pull {ollama_status.get('ollama_model', 'phi3')}</code><br>
            3. Click 'Refresh Ollama' button above<br><br>
            Using fallback grounded answer from retrieved case data.
          </div>
          <div class="subhead">Fallback Context</div>
          <div class="text-block">{as_text(row.get("investigator_brief"))}</div>
        </div>
        """

    try:
        model_name = ollama_status.get("preferred_model") or ollama_status.get("ollama_model", "phi3")
        result = assistant.answer_query_with_ollama(query, model=model_name, top_k=5)
        
        if result.get("error"):
            return f"""
            <div class="panel-card">
              <div class="section-title">Investigator Query</div>
              <div class="alert error">
                <b>Error:</b> {result.get("error")}
              </div>
            </div>
            """

        answer = as_text(result.get("answer"), "N/A").replace("\n", "<br>")
        status = as_text(result.get("grounding_status"), "N/A")
        fallback = as_text(result.get("fallback_reason"), "")
        ctxs = result.get("retrieved_contexts", [])
        
        # Status badge
        status_class = "success" if "llm_grounded" in status else "warning" if "fallback" in status else "info"
        
        chips = "".join(
            f"<span class='chip'>{as_text(c.get('account_id'))} • {as_text(c.get('primary_action'))} • Score: {round(float(c.get('risk_score', 0)), 1)}</span>"
            for c in ctxs[:8]
        )
        
        return f"""
        <div class="panel-card">
          <div class="section-title">Investigator Response</div>
          <div class="query-display">
            <b>Query:</b> {query}
          </div>
          <div class="subhead">Answer</div>
          <div class="text-block answer">{answer}</div>
          <div class="subhead">Grounding Information</div>
          <div class="meta">
            <span class="badge {status_class}">Status: {status}</span>
            {f'<br><b>Fallback reason:</b> {fallback}' if fallback else ''}
          </div>
          <div class="subhead">Retrieved Contexts ({len(ctxs)} documents)</div>
          <div class="chips">{chips if chips else '<span class="chip">No contexts</span>'}</div>
        </div>
        """
    except Exception as e:
        return f"""
        <div class="panel-card">
          <div class="section-title">Investigator Query</div>
          <div class="alert error">
            <b>Exception:</b> {str(e)}<br>
            <pre>{traceback.format_exc()}</pre>
          </div>
        </div>
        """


def refresh_ollama_status():
    """Refresh only Ollama status"""
    try:
        app_state.ollama_status = check_ollama_status()
        
        ol = app_state.ollama_status
        is_running = ol.get('ollama_running', False)
        is_installed = ol.get('ollama_model_installed', False)
        error = ol.get('ollama_error')
        model_count = ol.get('ollama_model_count', 0)
        
        status_icon = '✅ ONLINE' if is_running else '❌ OFFLINE'
        model_icon = '✅ YES' if is_installed else '❌ NO'
        
        status_text = f"""
### System Status

- **Ollama Service:** {status_icon}
- **Model ({ol.get('ollama_model', 'phi3')}):** {model_icon}
- **Models Available:** `{model_count}`
- **Total Accounts:** `{len(app_state.get_accounts()):,}`
- **API Endpoint:** `{ol.get('ollama_base_url', 'http://localhost:11434/api')}`

{f"**⚠️ Connection Error:** {error}" if error else '**✅ Connected to Ollama successfully**'}

**Note:** Status shows ONLINE only when Ollama service is actually running and responding on port 11434.
{f"**💡 To start Ollama:** If you closed it, run `ollama serve` in a terminal" if not is_running else '**ℹ️ Ollama is active:** To stop it, close the terminal running ollama serve'}
        """
        
        success_msg = f"✅ Ollama is ONLINE ({model_count} models available)" if is_running else "❌ Ollama is OFFLINE"
        return status_text, success_msg
    except Exception as e:
        return f"### System Status\n\n**Error refreshing:** {str(e)}", f"Error: {str(e)}"


def refresh_data():
    """Refresh data from disk"""
    success, message = app_state.refresh()
    
    # Get updated status
    ol = app_state.ollama_status or {}
    is_running = ol.get('ollama_running', False)
    is_installed = ol.get('ollama_model_installed', False)
    error = ol.get('ollama_error')
    
    status_icon = '✅ ONLINE' if is_running else '❌ OFFLINE'
    model_icon = '✅ YES' if is_installed else '❌ NO'
    
    status_text = f"""
### System Status

- **Ollama Running:** {status_icon}
- **Model ({ol.get('ollama_model', 'phi3')}) Installed:** {model_icon}
- **Total Accounts:** `{len(app_state.get_accounts()):,}`

{f"**⚠️ Error:** {error}" if error else ''}
{f"**💡 To start Ollama:** Run `ollama serve` in a terminal" if not is_running else ''}
    """
    
    if success:
        return (
            overview_panel(),
            gr.Dropdown(choices=app_state.get_accounts(), value=app_state.selected_account),
            f"✅ {message}",
            create_analytics_charts(),
            status_text
        )
    else:
        return (
            f"<div class='alert error'>{message}</div>",
            gr.Dropdown(choices=[]),
            f"❌ {message}",
            go.Figure(),
            status_text
        )


def create_analytics_charts():
    """Create optimized analytics visualizations"""
    df = app_state.df
    if df is None or df.empty:
        return go.Figure()
    
    # Create subplot figure
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Top 20 Risk Scores", "Risk Level Distribution", 
                       "Primary Actions", "Fraud Ring Cases"),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.18,  # Increased spacing
        horizontal_spacing=0.15
    )
    
    # Top 20 accounts - simple bar chart with better spacing
    top20 = df.nsmallest(20, 'risk_score', keep='first') if len(df) > 20 else df
    top20 = top20.sort_values("risk_score", ascending=False)
    fig.add_trace(
        go.Bar(
            x=top20["account_id"].astype(str), 
            y=top20["risk_score"],
            marker_color='#6366f1',
            hovertemplate='%{x}<br>Risk: %{y:.1f}<extra></extra>',
            text=top20["risk_score"].round(1),
            textposition='outside',
            textfont=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Risk level distribution
    if "risk_level" in df.columns:
        risk_dist = df["risk_level"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=risk_dist.index,
                values=risk_dist.values,
                marker_colors=['#ef4444', '#f59e0b', '#6366f1', '#10b981'],
                hovertemplate='%{label}<br>%{value} (%{percent})<extra></extra>',
                textfont=dict(size=10)
            ),
            row=1, col=2
        )
    
    # Primary actions - limit to top 8
    if "primary_action" in df.columns:
        action_dist = df["primary_action"].value_counts().head(8)
        fig.add_trace(
            go.Bar(
                x=action_dist.index,
                y=action_dist.values,
                marker_color='#8b5cf6',
                hovertemplate='%{x}<br>Count: %{y}<extra></extra>',
                text=action_dist.values,
                textposition='outside',
                textfont=dict(size=8)
            ),
            row=2, col=1
        )
    
    # Fraud rings - limit to top 10
    if "fraud_ring_flag" in df.columns and "community_id" in df.columns:
        ring_df = df[df["fraud_ring_flag"] == "YES"].groupby("community_id").size()
        ring_df = ring_df.nlargest(10).reset_index(name="count")
        fig.add_trace(
            go.Bar(
                x=ring_df["community_id"].astype(str),
                y=ring_df["count"],
                marker_color='#ef4444',
                hovertemplate='Community %{x}<br>Cases: %{y}<extra></extra>',
                text=ring_df["count"],
                textposition='outside',
                textfont=dict(size=8)
            ),
            row=2, col=2
        )
    
    # Better layout to prevent overlaps
    fig.update_layout(
        height=800,  # Increased height
        showlegend=False,
        template="plotly_white",
        margin=dict(t=80, b=60, l=60, r=40),
        font=dict(size=10)
    )
    
    # Update all x-axes with better spacing
    fig.update_xaxes(
        tickangle=-45,
        tickfont=dict(size=9),
        row=1, col=1
    )
    fig.update_xaxes(
        tickangle=-45,
        tickfont=dict(size=9),
        row=2, col=1
    )
    fig.update_xaxes(
        tickangle=-45,
        tickfont=dict(size=9),
        row=2, col=2
    )
    
    # Update y-axes
    fig.update_yaxes(tickfont=dict(size=9))
    
    return fig


def make_app():
    """Create Gradio application"""
    accounts = app_state.get_accounts()
    default_account = app_state.selected_account or (accounts[0] if accounts else "")
    ol = app_state.ollama_status or {}
    
    is_running = ol.get('ollama_running', False)
    is_installed = ol.get('ollama_model_installed', False)
    error = ol.get('ollama_error')
    
    status_icon = '✅ ONLINE' if is_running else '❌ OFFLINE'
    model_icon = '✅ YES' if is_installed else '❌ NO'

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --border: #e2e8f0;
      --text: #1e293b;
      --muted: #64748b;
      --primary: #6366f1;
      --success: #10b981;
      --warning: #f59e0b;
      --error: #ef4444;
    }
    
    body, .gradio-container {
      background: var(--bg) !important;
      color: var(--text) !important;
      font-family: 'Inter', system-ui, sans-serif !important;
    }
    
    .hero, .panel-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 16px;
    }
    
    .hero {
      border-left: 4px solid var(--primary);
    }
    
    .hero h1 {
      margin: 8px 0 0;
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--text);
    }
    
    .hero p {
      color: var(--muted);
      margin-top: 8px;
    }
    
    .eyebrow {
      display: inline-block;
      padding: 6px 12px;
      border-radius: 6px;
      background: var(--primary);
      color: white;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }
    
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    
    .metrics.compact {
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    
    .metric {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
    }
    
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 6px;
    }
    
    .metric b {
      display: block;
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--primary);
    }
    
    .section-title {
      font-size: 1.2rem;
      font-weight: 700;
      margin-bottom: 12px;
      color: var(--text);
      padding-left: 10px;
      border-left: 3px solid var(--primary);
    }
    
    .subhead {
      margin-top: 16px;
      margin-bottom: 8px;
      font-size: 11px;
      text-transform: uppercase;
      color: var(--primary);
      font-weight: 700;
    }
    
    .text-block {
      margin-top: 8px;
      padding: 12px;
      border-radius: 6px;
      background: #f1f5f9;
      border: 1px solid var(--border);
      line-height: 1.6;
      color: var(--text);
    }
    
    .text-block.answer {
      background: #eef2ff;
      border-color: var(--primary);
    }
    
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    
    .chip {
      padding: 6px 12px;
      border-radius: 6px;
      background: var(--surface);
      border: 1px solid var(--border);
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
    }
    
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    
    .grid > div {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
    }
    
    .grid span {
      display: block;
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 6px;
    }
    
    .grid b {
      display: block;
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--primary);
    }
    
    .meta {
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
      padding: 8px;
      background: #f8fafc;
      border-radius: 6px;
    }
    
    .alert {
      padding: 12px;
      border-radius: 6px;
      margin: 12px 0;
      background: #fef3c7;
      border: 1px solid #fbbf24;
      color: var(--text);
    }
    
    .alert.error {
      background: #fee2e2;
      border-color: #ef4444;
    }
    
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }
    
    .badge.success {
      background: #d1fae5;
      color: #065f46;
      border: 1px solid #10b981;
    }
    
    .badge.warning {
      background: #fef3c7;
      color: #92400e;
      border: 1px solid #f59e0b;
    }
    
    .badge.info {
      background: #dbeafe;
      color: #1e40af;
      border: 1px solid #3b82f6;
    }
    
    .badge.error {
      background: #fee2e2;
      color: #991b1b;
      border: 1px solid #ef4444;
    }
    
    .query-display {
      padding: 12px;
      background: #eef2ff;
      border-radius: 6px;
      margin: 12px 0;
      font-weight: 600;
      border: 1px solid var(--primary);
      color: var(--text);
    }
    
    button {
      border-radius: 6px !important;
      font-weight: 600 !important;
    }
    
    @media (max-width: 768px) {
      .metrics, .grid {
        grid-template-columns: 1fr;
      }
      .hero h1 {
        font-size: 1.4rem;
      }
    }
    """

    with gr.Blocks(css=css, title="Fraud Decision Intelligence Platform", theme=gr.themes.Soft()) as demo:
        # Header
        overview_html = gr.HTML(overview_panel())
        
        # System status
        with gr.Row():
            with gr.Column(scale=2):
                system_status_md = gr.Markdown(
                    f"""
                    ### System Status
                    
                    - **Ollama Running:** {status_icon}
                    - **Model ({ol.get('ollama_model', 'phi3')}) Installed:** {model_icon}
                    - **Total Accounts:** `{len(accounts):,}`
                    
                    {f"**⚠️ Error:** {error}" if error else ''}
                    {f"**💡 To start Ollama:** Run `ollama serve` in a terminal, then click 'Refresh Ollama'" if not is_running else '**✅ Ollama is ready for RAG queries**'}
                    """
                )
            with gr.Column(scale=1):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Data", variant="primary", scale=1)
                    refresh_ollama_btn = gr.Button("Refresh Ollama", variant="secondary", scale=1)
                refresh_status = gr.Textbox(label="Status", interactive=False, visible=False)
        
        # Main account selector (shared across tabs) - Optimized for performance
        with gr.Row():
            # Limit dropdown to top 100 accounts initially for better performance
            top_accounts_list = app_state.df.sort_values("risk_score", ascending=False).head(100)["account_id"].astype(str).tolist() if not app_state.df.empty else accounts[:100]
            
            account_selector = gr.Dropdown(
                choices=top_accounts_list, 
                value=default_account, 
                label="🔍 Select Account (Showing top 100 by risk score - type to search all)",
                interactive=True,
                allow_custom_value=True,
                filterable=True,  # Enable search/filter
                elem_classes="account-selector"
            )

        # Tabs
        with gr.Tabs():
            with gr.Tab("📋 Case Details"):
                gr.Markdown("### 📊 Comprehensive Account Analysis")
                case_output = gr.Markdown(value=account_detail_panel(default_account))
                with gr.Row():
                    gr.Examples(
                        examples=[[a] for a in app_state.get_top_accounts(10)["account_id"].astype(str).tolist()],
                        inputs=account_selector,
                        label="⭐ Top 10 High-Risk Accounts (Click to load)"
                    )

            with gr.Tab("🕸️ Graph Intelligence"):
                gr.Markdown("### � Network Analysis & Community Detection")
                graph_html = gr.HTML()
                graph_plot = gr.Plot()
                gr.Markdown("""
                **Graph Intelligence** uses network analysis to identify fraud rings and connected suspicious accounts.
                - **Community Detection**: Groups related accounts based on transaction patterns
                - **Influence Analysis**: Identifies key players in fraud networks
                - **Risk Propagation**: Tracks how risk spreads through connections
                """)

            with gr.Tab("💰 Impact Assessment"):
                gr.Markdown("### 💵 Financial Impact & Priority Scoring")
                impact_html = gr.HTML()
                gr.Markdown("""
                **Impact Metrics** quantify the financial consequences and operational priorities:
                - **Exposure**: Potential fraud magnitude
                - **Prevention**: Value protected by detection
                - **Priority Score**: Investigation urgency ranking
                """)

            with gr.Tab("🔍 Investigator AI"):
                gr.Markdown("### 🤖 AI-Powered Investigation Briefing")
                investigator_html = gr.HTML()
                gr.Markdown("""
                **Investigator AI** provides human-readable summaries and keyword extraction for rapid case assessment.
                Keywords are categorized by risk level, action type, and behavioral patterns.
                """)

            with gr.Tab("🤖 RAG Query"):
                gr.Markdown("### 💬 Retrieval-Augmented Generation (RAG) Query System")
                gr.Markdown("""
                Ask natural language questions about accounts and fraud patterns. 
                The system retrieves relevant case context and uses **Ollama** (local LLM) to generate grounded answers.
                """)
                with gr.Row():
                    query_input = gr.Textbox(
                        label="💬 Ask a question",
                        placeholder="e.g., Why is this account suspicious? Find connected risky users...",
                        lines=3
                    )
                query_btn = gr.Button("🚀 Ask Question", variant="primary", size="lg")
                query_output = gr.HTML()
                
                with gr.Accordion("📝 Example Queries (Click to expand)", open=False):
                    gr.Examples(
                        examples=[
                            [f"Why is account {default_account} suspicious?"],
                            [f"Find connected risky users for {default_account}"],
                            ["Show top risky accounts"],
                            ["What fraud patterns are detected?"],
                            [f"Summarize the case for {default_account}"],
                            ["Which accounts are in fraud rings?"],
                            ["Show me high-priority cases"],
                        ],
                        inputs=query_input,
                        label="Click any example to use it"
                    )

            with gr.Tab("📊 Analytics Dashboard"):
                gr.Markdown("### 📈 Visual Intelligence & Trends")
                analytics_plot = gr.Plot(value=create_analytics_charts())
                gr.Markdown("""
                **Analytics Dashboard** provides visual insights across the entire dataset:
                - **📊 Risk Distribution**: See how risk scores spread across accounts
                - **🎯 Risk Categories**: Breakdown by severity level
                - **⚡ Action Types**: What actions are being recommended
                - **🕸️ Fraud Rings**: Communities with connected fraud cases
                
                *All charts are interactive - hover for details, click legend to toggle visibility*
                """)
                with gr.Row():
                    gr.Button("📊 Export Analytics Report", variant="secondary")
                    gr.Button("📄 Generate PDF Summary", variant="secondary")

        # Event handlers - Optimized to reduce lag
        account_selector.change(
            fn=lambda acc: (
                account_detail_panel(acc),
                *graph_panel(acc),
                impact_panel(acc),
                investigator_panel(acc)
            ),
            inputs=[account_selector],
            outputs=[case_output, graph_html, graph_plot, impact_html, investigator_html],
            show_progress="minimal"  # Reduce visual overhead
        )
        
        query_btn.click(
            fn=query_answer,
            inputs=[query_input, account_selector],
            outputs=[query_output]
        )
        
        query_input.submit(
            fn=query_answer,
            inputs=[query_input, account_selector],
            outputs=[query_output]
        )
        
        refresh_btn.click(
            fn=refresh_data,
            inputs=[],
            outputs=[overview_html, account_selector, refresh_status, analytics_plot, system_status_md]
        )
        
        refresh_ollama_btn.click(
            fn=refresh_ollama_status,
            inputs=[],
            outputs=[system_status_md, refresh_status]
        )
        
        # Initialize panels on load
        demo.load(
            fn=lambda: (
                *graph_panel(default_account),
                impact_panel(default_account),
                investigator_panel(default_account)
            ),
            outputs=[graph_html, graph_plot, impact_html, investigator_html]
        )

    return demo


if __name__ == "__main__":
    print("=" * 70)
    print("Starting Autonomous Financial Crime Intelligence Platform")
    print("=" * 70)
    
    # Check if data exists
    if not SUBMISSION_PATH.exists():
        print(f"❌ ERROR: {SUBMISSION_PATH} not found!")
        print("\nPlease run the main pipeline first to generate data:")
        print("  python main.py")
        print("\nThis will create the submission.csv file with fraud intelligence data.")
        print("=" * 70)
        input("\nPress Enter to exit...")
        exit(1)
    
    # Display status
    try:
        print(f"✅ Data file found: {SUBMISSION_PATH}")
        print(f"✅ Accounts loaded: {len(app_state.get_accounts()):,}")
        
        ol_status = app_state.ollama_status or {}
        ollama_running = ol_status.get('ollama_running', False)
        ollama_installed = ol_status.get('ollama_model_installed', False)
        ollama_model = ol_status.get('ollama_model', 'phi3')
        ollama_error = ol_status.get('ollama_error', '')
        
        if ollama_running:
            print(f"✅ Ollama is running")
            print(f"✅ Model ({ollama_model}) installed: {ollama_installed}")
            if not ollama_installed:
                print(f"   ⚠️ Model not found - install with: ollama pull {ollama_model}")
        else:
            print(f"❌ Ollama is not running")
            if ollama_error:
                print(f"   Error: {ollama_error}")
            print(f"   📝 RAG queries will show error messages")
            print(f"   💡 To enable RAG functionality:")
            print(f"      1. Start Ollama: ollama serve")
            print(f"      2. Install model: ollama pull {ollama_model}")
            print(f"      3. Click 'Refresh Ollama' in the app")
    except Exception as e:
        print(f"⚠️ Warning during initialization: {e}")
    
    print("=" * 70)
    print("\n🚀 Launching Gradio interface...")
    print("📍 Local URL: http://127.0.0.1:7860")
    print("📍 Network URL: http://localhost:7860")
    print("\n💡 Tips:")
    print("   - Click 'Refresh Ollama Status' if you start Ollama after launching")
    print("   - Use 'Refresh Data' to reload submission.csv without restarting")
    print("=" * 70)
    
    app = make_app()
    app.launch(
        share=False,
        server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0 for Windows
        server_port=7860,
        show_error=True,
        inbrowser=True  # Automatically open browser
    )
