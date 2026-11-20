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
the policy corpus. Open http://127.0.0.1:8001/ for the chat UI. Sessions and
messages persist in PostgreSQL and can be reopened from the sidebar; `/health`
checks the API and database connection. Set `LLM_MODE=record` in the proxy
environment file before running the script to make model calls and save
responses. The default, `replay`, requires responses already stored in
PostgreSQL. Use Human review and Activity log in the UI for review decisions
and audit events. Application request, workflow, and model logs go to the API
container output:
`docker compose --env-file '..\litellm-enterprise-proxy\.env' -f 'compose.yaml' logs -f api`.
