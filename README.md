# Policy Evidence Agent

An AI agent that answers security assessments from policy documents, cites its
evidence, and sends uncertain answers for human review.

## Run

Requires Docker Compose and the existing LiteLLM/PostgreSQL stack. The proxy
environment file must define `DB_PASSWORD` and `LITELLM_MASTER_KEY`.

```powershell
$proxyEnv = 'C:\path\to\litellm-enterprise-proxy\.env'
docker compose --env-file $proxyEnv up -d --build api
docker compose --env-file $proxyEnv run --rm api qa ingest
Invoke-RestMethod http://127.0.0.1:8001/health
docker compose --env-file $proxyEnv run --rm api qa answer Q001
```

Set `LLM_MODE=record` in the proxy environment file before starting the API to
make model calls and save responses for replay. The default, `replay`, requires
responses already stored in PostgreSQL.
