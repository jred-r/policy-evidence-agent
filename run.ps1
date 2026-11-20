[CmdletBinding()]
param(
    [string]$ProxyDirectory
)

$ErrorActionPreference = 'Stop'
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProxyDirectory)) {
    $ProxyDirectory = Join-Path (Split-Path -Parent $scriptDirectory) 'litellm-enterprise-proxy'
}

$proxyEnv = Join-Path $ProxyDirectory '.env'
$proxyCompose = Join-Path $ProxyDirectory 'docker-compose.yml'
$proxyOverride = Join-Path $ProxyDirectory 'docker-compose.override.yml'

foreach ($requiredPath in @($proxyEnv, $proxyCompose, $proxyOverride)) {
    if (-not (Test-Path $requiredPath -PathType Leaf)) {
        throw "Required proxy file not found: $requiredPath"
    }
}

$webDirectory = Join-Path $scriptDirectory 'web'
$staticDirectory = Join-Path $scriptDirectory 'src\qa\interfaces\static'
$webBuildArgs = @(
    'run',
    '--rm',
    '--mount', "type=bind,source=$webDirectory,target=/web",
    '--mount', "type=bind,source=$staticDirectory,target=/output",
    '--workdir', '/web',
    '--env', 'STATIC_OUTPUT_DIR=/output',
    'node:22-alpine',
    'sh', '-c', 'npm ci && npm run build'
)
& docker @webBuildArgs
if ($LASTEXITCODE -ne 0) {
    throw "Building the chat UI failed with exit code $LASTEXITCODE."
}

$proxyArgs = @(
    'compose',
    '--env-file', $proxyEnv,
    '-f', $proxyCompose,
    '-f', $proxyOverride,
    'up', '-d'
)
& docker @proxyArgs
if ($LASTEXITCODE -ne 0) {
    throw "Starting the LiteLLM/PostgreSQL stack failed with exit code $LASTEXITCODE."
}

$agentCompose = Join-Path $scriptDirectory 'compose.yaml'
& docker compose --env-file $proxyEnv -f $agentCompose up -d --build api
if ($LASTEXITCODE -ne 0) {
    throw "Starting the agent API failed with exit code $LASTEXITCODE."
}

& docker compose --env-file $proxyEnv -f $agentCompose exec -T api qa ingest
if ($LASTEXITCODE -ne 0) {
    throw "Policy ingestion failed with exit code $LASTEXITCODE."
}

$healthy = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if (Test-NetConnection -ComputerName '127.0.0.1' -Port 8001 -InformationLevel Quiet -WarningAction SilentlyContinue) {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/health' -TimeoutSec 5
        if ($health.status -ne 'healthy' -or $health.database -ne 'connected') {
            throw 'The agent API responded, but its PostgreSQL health check failed.'
        }
        $healthy = $true
        break
    }
    Start-Sleep -Seconds 2
}

if (-not $healthy) {
    throw 'The agent API did not become available on 127.0.0.1:8001.'
}

Write-Host 'Policy Evidence Agent is ready at http://127.0.0.1:8001/'
