# run.ps1 — Clears VS Code injected env vars then runs a command.
# Usage:
#   .\run.ps1 test          → python test_critics.py
#   .\run.ps1 api           → uvicorn api.app:app --reload --port 8000
#   .\run.ps1 ui            → streamlit run ui/app.py
#   .\run.ps1 dryrun        → python test_critics.py --dry-run

# Clear stale VS Code-injected values so .env is the source of truth
$staleVars = @(
    "GOOGLE_API_KEY","GROQ_API_KEY","OPENROUTER_API_KEY",
    "LOGIC_MODEL","SAFETY_MODEL","ACCURACY_MODEL","ADJUDICATOR_MODEL",
    "COMPLETENESS_MODEL","STYLE_MODEL",
    "ACCURACY_FALLBACK_MODEL","ACCURACY_FALLBACK_PROVIDER",
    "LOGIC_FALLBACK_MODEL","LOGIC_FALLBACK_PROVIDER",
    "SAFETY_FALLBACK_MODEL","SAFETY_FALLBACK_PROVIDER"
)
foreach ($v in $staleVars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }

switch ($args[0]) {
    "test"   { .venv\Scripts\python.exe test_critics.py }
    "dryrun" { .venv\Scripts\python.exe test_critics.py --dry-run }
    "api"    { .venv\Scripts\uvicorn.exe api.app:app --reload --port 8000 }
    "ui"     { .venv\Scripts\streamlit.exe run ui/app.py }
    default  { Write-Host "Usage: .\run.ps1 [test|dryrun|api|ui]" }
}
