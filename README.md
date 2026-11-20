# Policy Evidence Agent

An AI agent that answers security assessments from policy documents, cites its
evidence, and sends uncertain answers for human review. Policy embeddings,
response cache, review decisions, and workflow checkpoints persist in PostgreSQL.

## Run

Requires Docker Compose and the existing LiteLLM/PostgreSQL stack. The proxy
environment file must define `DB_PASSWORD` and `LITELLM_MASTER_KEY`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

The script starts PostgreSQL, LiteLLM, the gateway, and the agent, then ingests
the policy corpus. Open http://127.0.0.1:8001/ for the browser UI; `/health`
checks the API and PostgreSQL connection. Set `LLM_MODE=record` in the proxy
environment file before running the script to make model calls and save
responses. The default, `replay`, requires responses already stored in
PostgreSQL.
