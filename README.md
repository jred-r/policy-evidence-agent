# Policy Evidence Agent

An AI chat assistant that answers policy and security questions with citations.
It handles ordinary conversation and gently redirects unrelated questions toward
policy topics. Chat history, policy embeddings, response cache, and audit events
persist in PostgreSQL.

## Run

Requires Docker Compose and the existing LiteLLM/PostgreSQL stack. The proxy
environment file must define `DB_PASSWORD` and `LITELLM_MASTER_KEY`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

The script builds the React chat UI in an isolated Node container, starts
PostgreSQL, LiteLLM, the gateway, and the agent, then ingests the policy corpus.
Open http://127.0.0.1:8001/ for the chat UI. The UI uses the
[assistant-ui](https://www.assistant-ui.com/) React chat components (MIT
licensed), connected to the existing FastAPI endpoints. To work on the UI
directly, install Node.js and run `npm ci` and `npm run build` in `web/`.
Sessions and messages persist in PostgreSQL and can be reopened from the
sidebar; `/health` checks the API and database connection. Set
`LLM_MODE=record` in the proxy environment file before running the script to
make model calls and save responses. The default, `replay`, requires responses
already stored in PostgreSQL. Policy-related answers with grounding F1 below
50% are automatically withheld from the chat and sent to **Human review**.
A reviewer can edit, approve, or reject the draft; only an approved answer is
shown in that turn.
Grounding F1 compares answer tokens with the cited policy text. Other messages
return immediately; review is automatic for low-grounding policy answers.

Use the Activity log in the UI to inspect operational events. Existing audit
records are retained, so older entries may reflect the former automatic review
routing. Application request and model logs go to the API container output:
`docker compose --env-file '..\litellm-enterprise-proxy\.env' -f 'compose.yaml' logs -f api`.
