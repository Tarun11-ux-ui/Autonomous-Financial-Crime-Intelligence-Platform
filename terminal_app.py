import os
import traceback
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SUBMISSION_PATH = ROOT / "data" / "submission.csv"
STARTUP_LOG = ROOT / "terminal_app.log"
APP_NAME = "Autonomous Financial Crime Intelligence Platform"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def pause():
    input("\nPress Enter to continue...")


def fmt_money(value):
    try:
        return f"INR {float(value):,.2f}"
    except Exception:
        return "N/A"


def fmt_pct(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def print_header(title):
    print("\n" + "=" * 96)
    print(APP_NAME)
    print(title)
    print("=" * 96)


def print_subheader(title):
    print(f"\n--- {title} " + "-" * max(0, 78 - len(title)))


def print_kv(label, value):
    print(f"{label:<28} {value}")


def pick_column(df, *candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def load_data():
    if not SUBMISSION_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {SUBMISSION_PATH}")
    return pd.read_csv(SUBMISSION_PATH)


def normalize_columns(df):
    cleaned = df.copy()
    mappings = {
        "graph_risk_score": ["graph_risk_score", "graph_risk_score_x", "graph_risk_score_y"],
        "community_risk_score": ["community_risk_score", "community_risk_score_x", "community_risk_score_y"],
        "fraud_ring_flag": ["fraud_ring_flag", "fraud_ring_flag_x", "fraud_ring_flag_y"],
    }
    for canonical, candidates in mappings.items():
        for column in candidates:
            if column in cleaned.columns:
                cleaned[canonical] = cleaned[column]
                break
    return cleaned


def _lazy_imports():
    from src.investigator_ai import InvestigatorAssistant
    from src.ollama_rag import summarize_ollama_integration
    return pd, InvestigatorAssistant, summarize_ollama_integration


def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_overview(df):
    print_header("OVERVIEW")
    print_kv("Accounts screened:", f"{len(df):,}")
    if "case_generated" in df.columns:
        print_kv("Cases generated:", f"{(df['case_generated'].astype(str).str.upper() == 'YES').sum():,}")
    if "auto_action_eligible" in df.columns:
        print_kv("Auto actions:", f"{(df['auto_action_eligible'].astype(str).str.upper() == 'YES').sum():,}")
    if "fraud_ring_flag" in df.columns:
        print_kv("Fraud ring cases:", f"{(df['fraud_ring_flag'].astype(str).str.upper() == 'YES').sum():,}")
    if "estimated_fraud_prevented_inr" in df.columns:
        total = pd.to_numeric(df["estimated_fraud_prevented_inr"], errors="coerce").fillna(0).sum()
        print_kv("Estimated fraud prevented:", fmt_money(total))
    print_subheader("Top priority cases")

    columns = [
        "account_id", "risk_score", "risk_level", "primary_action",
        "impact_priority_score", "fraud_ring_flag", "connected_risky_accounts"
    ]
    available = [c for c in columns if c in df.columns]
    print(df[available].sort_values("risk_score", ascending=False).head(10).to_string(index=False))


def print_case(row):
    print_header(f"ACCOUNT {row.get('account_id', 'N/A')}")
    print_subheader("Core decision")
    fields = [
        ("Risk score", row.get("risk_score")),
        ("Risk level", row.get("risk_level")),
        ("Primary action", row.get("primary_action")),
        ("Action priority", row.get("action_priority")),
        ("Action owner", row.get("action_owner")),
        ("Decision confidence", fmt_pct(row.get("decision_confidence"))),
        ("Impact band", row.get("impact_band")),
        ("Impact priority", row.get("impact_priority_score")),
        ("Estimated exposure", fmt_money(row.get("estimated_exposure_inr"))),
        ("Estimated fraud prevented", fmt_money(row.get("estimated_fraud_prevented_inr"))),
        ("Graph risk score", row.get("graph_risk_score")),
        ("Community risk score", row.get("community_risk_score")),
        ("Community ID", row.get("community_id")),
        ("Fraud ring flag", row.get("fraud_ring_flag")),
        ("Connected risky accounts", row.get("connected_risky_accounts")),
        ("Suspicious window", f"{row.get('suspicious_start', 'N/A')} -> {row.get('suspicious_end', 'N/A')}"),
    ]
    for label, value in fields:
        print_kv(label, value)

    print_subheader("Decision summary")
    print(row.get("decision_summary", "N/A"))
    print_subheader("Decision reasons")
    print(row.get("decision_reasons", "N/A"))
    print_subheader("Explainability")
    print(row.get("explanation_summary", "N/A"))
    print_subheader("Human reasoning")
    print(row.get("human_readable_reasoning", "N/A"))
    print_subheader("Investigator brief")
    print(row.get("investigator_brief", "N/A"))
    print_subheader("Retrieval text")
    print(row.get("retrieval_text", "N/A"))


def print_graph_view(row, df):
    print_header(f"GRAPH VIEW FOR {row.get('account_id', 'N/A')}")
    print_subheader("Graph intelligence")
    print_kv("Graph risk score:", row.get('graph_risk_score', 'N/A'))
    print_kv("Community risk score:", row.get('community_risk_score', 'N/A'))
    print_kv("Community ID:", row.get('community_id', 'N/A'))
    print_kv("Community size:", row.get('community_size', 'N/A'))
    print_kv("Fraud ring flag:", row.get('fraud_ring_flag', 'N/A'))
    print_kv("Influential node flag:", row.get('influential_node_flag', 'N/A'))

    connected = [c.strip() for c in str(row.get('connected_risky_accounts', '')).split('|') if c.strip()]
    print_subheader("Connected risky accounts")
    if connected:
        for idx, acct in enumerate(connected, start=1):
            print(f"  {idx}. {acct}")
    else:
        print("  None retrieved")

    print_subheader("Mini network view")
    center = row.get('account_id', 'CENTER')
    print(f"      [ {center} ]")
    if connected:
        for acct in connected[:8]:
            print(f"         |--> {acct}")
    else:
        print("         |--> no direct connected accounts available")

    if 'community_id' in df.columns:
        community = row.get('community_id', None)
        if pd.notna(community):
            peers = df[df['community_id'] == community].sort_values('risk_score', ascending=False).head(6)
            print_subheader("Top community peers")
            for _, peer in peers.iterrows():
                print(f"  - {peer.get('account_id')} | risk={peer.get('risk_score')} | action={peer.get('primary_action')}")


def print_all_features_for_account(row, df):
    print_case(row)
    print_graph_view(row, df)


def list_accounts(df, limit=20):
    cols = [c for c in ["account_id", "risk_score", "risk_level", "primary_action", "fraud_ring_flag"] if c in df.columns]
    print_header(f"TOP {limit} ACCOUNTS")
    print(df[cols].sort_values("risk_score", ascending=False).head(limit).to_string(index=False))


def ask_query(assistant, current_account_id):
    print_header("INVESTIGATOR / RAG QUERY")
    print(f"Current account: {current_account_id}")
    print_subheader("Example prompts")
    print(f"- Why is account {current_account_id} suspicious?")
    print(f"- Find connected risky users for {current_account_id}")
    print("- Show top risky accounts")
    print("- Summarize this case")
    print("\nType a question, or press Enter to go back.")
    query = input("> ").strip()
    if not query:
        return

    print("\nRunning Ollama-backed investigator query...\n")
    result = assistant.answer_query_with_ollama(query, model="phi3", top_k=5)
    print_header("ASSISTANT RESPONSE")
    if result.get("error"):
        print("Answer source: Ollama unavailable")
        print("Error:", result["error"])
    else:
        grounding_status = str(result.get("grounding_status", "")).lower()
        fallback_reason = result.get("fallback_reason")
        if grounding_status.startswith("llm"):
            print("Answer source: Ollama phi3")
        elif grounding_status.startswith("fallback"):
            print("Answer source: Fallback grounded answer")
        else:
            print("Answer source: Unknown")
        print(result.get("answer", "N/A"))
        print("\nGrounding status:", result.get("grounding_status", "N/A"))
        if result.get("fallback_reason"):
            print("Fallback reason:", result.get("fallback_reason"))
        print("\nRetrieved contexts:")
        for ctx in result.get("retrieved_contexts", []):
            print(
                f"- {ctx.get('account_id')} | {ctx.get('primary_action')} | "
                f"score={ctx.get('risk_score')} | relation={ctx.get('context_relation')}"
            )

    pause()


def choose_account(df):
    print_header("SELECT ACCOUNT")
    print("1. Search by account ID")
    print("2. Show top 25 by risk")
    print("3. Pick from full list (paged)")
    print("Press Enter to use the top-risk account.")
    mode = input("> ").strip()

    if not mode:
        top = df.sort_values("risk_score", ascending=False).head(1)
        return str(top.iloc[0]["account_id"])

    if mode == "1":
        query = input("Enter full or partial account ID: ").strip().lower()
        matches = df[df["account_id"].astype(str).str.lower().str.contains(query, na=False)]
        if matches.empty:
            print("No matches found. Falling back to top-risk account.")
            return str(df.sort_values("risk_score", ascending=False).iloc[0]["account_id"])
        matches = matches.sort_values("risk_score", ascending=False).head(25)
        for idx, acct in enumerate(matches["account_id"].astype(str).tolist(), start=1):
            print(f"{idx:>2}. {acct}")
        pick = input("Select a number or exact account ID: ").strip()
        if pick.isdigit():
            i = int(pick) - 1
            if 0 <= i < len(matches):
                return str(matches.iloc[i]["account_id"])
        if pick in set(matches["account_id"].astype(str)):
            return pick
        print("Invalid selection. Using top match.")
        return str(matches.iloc[0]["account_id"])

    if mode == "2":
        top = df.sort_values("risk_score", ascending=False).head(25)
        available = top["account_id"].astype(str).tolist()
        for idx, acct in enumerate(available, start=1):
            print(f"{idx:>2}. {acct}")
        pick = input("Select a number or exact account ID: ").strip()
        if pick.isdigit():
            i = int(pick) - 1
            if 0 <= i < len(available):
                return available[i]
        if pick in set(available):
            return pick
        print("Invalid selection. Using top-risk account.")
        return str(top.iloc[0]["account_id"])

    if mode == "3":
        page_size = 25
        total = len(df)
        page = 0
        while True:
            start = page * page_size
            end = min(start + page_size, total)
            chunk = df.iloc[start:end]
            print_header(f"FULL LIST {start + 1}-{end} OF {total}")
            for idx, acct in enumerate(chunk["account_id"].astype(str).tolist(), start=start + 1):
                print(f"{idx:>5}. {acct}")
            print("\nEnter a number, next, prev, or exact account ID.")
            pick = input("> ").strip()
            if pick.lower() == "next":
                if end < total:
                    page += 1
                else:
                    print("Already at end.")
                continue
            if pick.lower() == "prev":
                if page > 0:
                    page -= 1
                else:
                    print("Already at start.")
                continue
            if pick.isdigit():
                idx = int(pick) - 1
                if 0 <= idx < total:
                    return str(df.iloc[idx]["account_id"])
            if pick in set(df["account_id"].astype(str)):
                return pick
            print("Invalid selection.")

    top = df.sort_values("risk_score", ascending=False).head(1)
    return str(top.iloc[0]["account_id"])


def current_selected_account(df, selected_account_id):
    if selected_account_id and not df[df["account_id"].astype(str) == str(selected_account_id)].empty:
        return str(selected_account_id)
    return str(df.sort_values("risk_score", ascending=False).iloc[0]["account_id"])


def choose_or_keep_account(df, selected_account_id, prompt="Use current account or choose a new one?"):
    print_header(prompt.upper())
    current = current_selected_account(df, selected_account_id)
    print(f"Current account: {current}")
    print("1. Keep current account")
    print("2. Choose a different account")
    choice = input("> ").strip()
    if choice == "2":
        return choose_account(df)
    return current


def quick_query_mode(assistant, df, acct):
    clear()
    print_header("QUICK QUERY")
    print(f"Selected account: {acct}")
    print("Type your question and press Enter. Leave blank to return.")
    query = input("> ").strip()
    if not query:
        return
    print("\nRunning Ollama-backed investigator query...\n")
    result = assistant.answer_query_with_ollama(query, model="phi3", top_k=5)
    print_header("ASSISTANT RESPONSE")
    if result.get("error"):
        print("Answer source: Ollama unavailable")
        print("Error:", result["error"])
    else:
        grounding_status = str(result.get("grounding_status", "")).lower()
        if grounding_status.startswith("llm"):
            print("Answer source: Ollama phi3")
        elif grounding_status.startswith("fallback"):
            print("Answer source: Fallback grounded answer")
        else:
            print("Answer source: Unknown")
        print(result.get("answer", "N/A"))
        print("\nGrounding status:", result.get("grounding_status", "N/A"))
        if result.get("fallback_reason"):
            print("Fallback reason:", result.get("fallback_reason"))
        print("\nRetrieved contexts:")
        for ctx in result.get("retrieved_contexts", []):
            print(
                f"- {ctx.get('account_id')} | {ctx.get('primary_action')} | "
                f"score={ctx.get('risk_score')} | relation={ctx.get('context_relation')}"
            )
    input("\nPress Enter to continue...")


def main():
    try:
        pd, InvestigatorAssistant, summarize_ollama_integration = _lazy_imports()
        clear()
        df = normalize_columns(load_data())
        assistant = InvestigatorAssistant(df)
        ol = summarize_ollama_integration()
        selected_account_id = current_selected_account(df, None)

        print_header("TERMINAL OPERATIONS CONSOLE")
        print_kv("Mode:", "Terminal-only demo")
        print_kv("Ollama available:", ol.get('ollama_available'))
        print_kv("Ollama model:", f"{ol.get('ollama_model')} | installed: {ol.get('ollama_model_installed')}")

        while True:
          print_subheader("Main menu")
          print("1. Overview")
          print("2. Select account and view full details")
          print("3. Ask investigator / RAG query")
          print("4. Top accounts")
          print("5. Graph visualization for selected account")
          print("6. Show all features for selected account")
          print("7. Refresh data")
          print("q. Quick query")
          print("0. Exit")
          choice = input("> ").strip()

          if choice == "0":
              break
          if choice == "1":
              clear()
              print_overview(df)
          elif choice == "2":
              acct = choose_account(df)
              selected_account_id = acct
              row = df[df["account_id"].astype(str) == str(acct)]
              if row.empty:
                  print("Account not found.")
                  continue
              clear()
              print_case(row.iloc[0])
          elif choice == "3":
              acct = current_selected_account(df, selected_account_id)
              clear()
              ask_query(assistant, acct)
          elif choice == "4":
              clear()
              list_accounts(df)
          elif choice == "5":
              acct = choose_or_keep_account(df, selected_account_id, "Graph view")
              row = df[df["account_id"].astype(str) == str(acct)]
              if row.empty:
                  print("Account not found.")
                  continue
              selected_account_id = acct
              clear()
              print_graph_view(row.iloc[0], df)
          elif choice == "6":
              acct = choose_or_keep_account(df, selected_account_id, "Full account feature view")
              row = df[df["account_id"].astype(str) == str(acct)]
              if row.empty:
                  print("Account not found.")
                  continue
              selected_account_id = acct
              clear()
              print_all_features_for_account(row.iloc[0], df)
              pause()
          elif choice == "7":
              clear()
              df = normalize_columns(load_data())
              assistant = InvestigatorAssistant(df)
              selected_account_id = current_selected_account(df, selected_account_id)
              print("Data refreshed.")
          elif choice.lower() == "q":
              quick_query_mode(assistant, df, current_selected_account(df, selected_account_id))
          else:
              print("Invalid choice.")
    except Exception as exc:
        STARTUP_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        print_header("STARTUP ERROR")
        print(str(exc))
        print(f"\nA full traceback was saved to: {STARTUP_LOG}")
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
