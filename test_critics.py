"""
test_critics.py
---------------
Smoke-test and demo for both Phase 1 (individual critics) and Phase 2
(full LangGraph orchestration pipeline).

Usage:
    python test_critics.py              # Full Phase 2 pipeline
    python test_critics.py --dry-run    # Config check only, no LLM calls
    python test_critics.py --critic accuracy   # Single Phase 1 critic
    python test_critics.py --phase 1    # Run individual critics (Phase 1)
    python test_critics.py --phase 2    # Run full pipeline (Phase 2, default)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from typing import Optional

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from config import settings, configure_logging
from models.critique import (
    ArbitrationRequest,
    CriticDimension,
    Critique,
    FinalVerdict,
    Severity,
)
from critics import (
    AccuracyCritic,
    LogicCritic,
    CompletenessCritic,
    SafetyCritic,
    StyleCritic,
    CRITIC_REGISTRY,
)

console = Console(force_terminal=True, highlight=False)

# ─────────────────────────────────────────────────────────────────────
#  Sample evaluation target (intentional bugs across all dimensions)
# ─────────────────────────────────────────────────────────────────────

SAMPLE_PROMPT = (
    "Explain how HTTPS works, including the TLS handshake process, "
    "and describe the difference between symmetric and asymmetric encryption "
    "with an example of each."
)

SAMPLE_RESPONSE = (
    "HTTPS works by using SSL to encrypt web traffic. When you visit a website, "
    "your browser and the server shake hands using a process invented by Google in 2010. "
    "During this handshake, they agree on an encryption key.\n\n"
    "Asymmetric encryption uses two different keys (public and private). "
    "RSA is an example -- it uses prime number magic to create unbreakable keys. "
    "The public key is used to encrypt, only the private key can decrypt.\n\n"
    "Symmetric encryption uses the same key for both -- AES is a good example. "
    "It's much slower than asymmetric encryption and therefore is only used "
    "for the initial handshake, after which both sides switch to asymmetric.\n\n"
    "HTTPS is 100% secure and no man-in-the-middle attack can ever succeed against it. "
    "You should always trust any website with HTTPS."
)
# Intentional bugs:
#   ACCURACY:     TLS not invented by Google in 2010; symmetric IS faster, not slower;
#                 HTTPS is not 100% secure.
#   LOGIC:        Claims symmetric is slower then says both sides "switch to asymmetric"
#                 (backwards logic).
#   COMPLETENESS: Never actually explains the TLS handshake steps.
#   SAFETY:       "always trust any website with HTTPS" is dangerous advice.
#   STYLE:        "prime number magic" is unprofessional.


# ─────────────────────────────────────────────────────────────────────
#  Rendering helpers
# ─────────────────────────────────────────────────────────────────────

SCORE_STYLE = {"A": "bold green", "B": "green", "C": "yellow", "D": "red", "F": "bold red"}
SEV_STYLE   = {
    Severity.CRITICAL: "bold red",
    Severity.MAJOR:    "red",
    Severity.MINOR:    "yellow",
    Severity.INFO:     "dim",
}


def render_critique(critique: Critique) -> None:
    grade_style  = SCORE_STYLE.get(critique.grade, "white")
    passed_text  = Text("PASS", style="green") if critique.passed else Text("FAIL", style="red")
    header = (
        f"[bold]{critique.dimension.value.upper()} CRITIC[/bold]  "
        f"[{grade_style}]Grade {critique.grade} ({critique.score}/100)[/{grade_style}]  "
        f"Conf: {critique.confidence:.0%}  "
        f"Latency: {critique.latency_ms or 0:.0f}ms  "
        f"Model: [italic]{critique.model_used}[/italic]"
    )
    console.print(Panel(header, expand=False))
    console.print(f"  {passed_text}  {critique.summary}\n")
    if critique.issues:
        for issue in critique.issues:
            sty = SEV_STYLE.get(issue.severity, "white")
            console.print(f"  [{sty}][{issue.severity.upper()}][/{sty}]  {issue.explanation}")
            console.print(f"    Quote: \"{issue.quote[:80]}\"")
            console.print(f"    Fix:   {issue.recommendation}")
            console.print()
    else:
        console.print("  [green]No issues found.[/green]\n")


def render_verdict(verdict: FinalVerdict) -> None:
    rec_style = {"APPROVE": "bold green", "REVISE": "yellow", "REJECT": "bold red"}
    sty = rec_style.get(verdict.recommendation, "white")

    # Quality score badge (1-10, Phase 3)
    qs = verdict.quality_score
    qs_style = "bold green" if qs >= 7 else ("yellow" if qs >= 5 else "bold red")
    qs_display = f"  Quality: [{qs_style}]{qs}/10[/{qs_style}]" if qs > 0 else ""

    console.print(Panel(
        f"[bold]FINAL VERDICT[/bold]   "
        f"[{sty}]{verdict.recommendation}[/{sty}]   "
        f"Score: [bold]{verdict.overall_score}/100[/bold]{qs_display}   "
        f"Confidence: {verdict.overall_confidence:.0%}   "
        f"Agreement: {verdict.critic_agreement:.0%}\n\n"
        f"{verdict.explanation}",
        title="Arbitration Result",
        border_style="bold blue",
    ))

    # --- Executive Summary (Phase 3) ---
    if verdict.executive_summary:
        console.print(Panel(
            verdict.executive_summary,
            title="[bold cyan]Executive Summary[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        ))

    # Dimension breakdown table
    table = Table(title="Dimension Breakdown", box=box.SIMPLE_HEAVY)
    table.add_column("Dimension", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("Passed")
    for dim, score in verdict.dimension_scores.items():
        passed = verdict.dimension_passed.get(dim, False)
        table.add_row(
            dim,
            str(score),
            Text("PASS", style="green") if passed else Text("FAIL", style="red"),
        )
    console.print(table)

    # --- Strengths (Phase 3) ---
    if verdict.strengths:
        console.print("\n[bold green]Strengths:[/bold green]")
        for strength in verdict.strengths:
            console.print(f"  [green]+[/green]  {strength}")

    # Confirmed key issues
    if verdict.key_issues:
        console.print("\n[bold]Key Issues (Confirmed):[/bold]")
        for issue in verdict.key_issues:
            sty2 = SEV_STYLE.get(issue.severity, "white")
            console.print(
                f"  [{sty2}][{issue.severity.upper()}][/{sty2}]  {issue.explanation}"
            )
            if issue.evidence:
                console.print(f"    [dim]Evidence: {issue.evidence}[/dim]")

    # --- Dismissed Issues (Phase 3) ---
    if verdict.dismissed_issues:
        console.print(
            f"\n[bold yellow]Dismissed Issues ({len(verdict.dismissed_issues)}):[/bold yellow]"
        )
        for di in verdict.dismissed_issues:
            orig = di.original_issue if len(di.original_issue) <= 80 else di.original_issue[:77] + "..."
            console.print(
                f"  [yellow dim][{di.dimension.value.upper()}][/yellow dim]  [dim]{orig}[/dim]"
            )
            console.print(f"    [dim]Dismissed: {di.dismissal_reason}[/dim]")

    # Provenance
    if verdict.fallbacks_used:
        console.print(f"\n[yellow]Fallbacks used: {len(verdict.fallbacks_used)}[/yellow]")
        for fb in verdict.fallbacks_used:
            console.print(f"  {fb.dimension.value}: {fb.primary_model} -> {fb.fallback_model}")

    flags = []
    if verdict.fast_path:
        flags.append("[green]fast-path[/green]")
    if verdict.adjudication_used:
        flags.append("[cyan]adjudicated[/cyan]")
    if flags:
        console.print(f"\nFlags: {', '.join(flags)}")

    console.print(
        f"\n[dim]Session: {verdict.session_id}  "
        f"Latency: {verdict.total_latency_ms or 0:.0f}ms[/dim]"
    )


def check_config() -> bool:
    table = Table(title="Configuration Status", box=box.ROUNDED, show_lines=True)
    table.add_column("Critic",    style="bold")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Fallback")
    table.add_column("Status")

    checks = [
        ("Accuracy",     "Google AI Studio", settings.accuracy_model,
         f"{settings.accuracy_fallback_provider}/{settings.accuracy_fallback_model}",
         settings.google_api_key, "your_google_ai_studio_api_key_here"),
        ("Logic",        "Groq",             settings.logic_model,
         f"{settings.logic_fallback_provider}/{settings.logic_fallback_model}",
         settings.groq_api_key, "your_groq_api_key_here"),
        ("Completeness", "Ollama (local)",   settings.completeness_model, "N/A",
         settings.ollama_base_url, None),
        ("Safety",       "OpenRouter",       settings.safety_model,
         f"{settings.safety_fallback_provider}/{settings.safety_fallback_model}",
         settings.openrouter_api_key, "your_openrouter_api_key_here"),
        ("Style",        "Ollama (local)",   settings.style_model, "N/A",
         settings.ollama_base_url, None),
    ]

    all_ok = True
    for critic, provider, model, fallback, key, placeholder in checks:
        if placeholder is None:
            status  = Text("Local", style="green")
            display = key
        elif key and key != placeholder:
            status  = Text("Configured", style="green")
            display = key[:8] + "..."
        else:
            status  = Text("Missing key", style="red")
            display = "(not set)"
            all_ok  = False
        table.add_row(critic, provider, model, fallback, status)

    console.print(table)
    return all_ok


# ─────────────────────────────────────────────────────────────────────
#  Phase 1 runner (individual critics)
# ─────────────────────────────────────────────────────────────────────

async def run_phase1(request: ArbitrationRequest, selected: Optional[str] = None) -> list[Critique]:
    if selected:
        dim = CriticDimension(selected.lower())
        cls = CRITIC_REGISTRY.get(dim)
        if not cls:
            console.print(f"[red]Unknown critic: {selected}[/red]")
            sys.exit(1)
        critics_to_run = [cls()]
    else:
        critics_to_run = [cls() for cls in CRITIC_REGISTRY.values()]

    console.print(f"\n[bold cyan]Running {len(critics_to_run)} critic(s) in parallel...[/bold cyan]\n")
    results = await asyncio.gather(*[c.evaluate(request) for c in critics_to_run], return_exceptions=True)

    critiques = []
    for r in results:
        if isinstance(r, Exception):
            console.print(f"[red]  Critic raised exception: {r}[/red]")
        else:
            critiques.append(r)
    return critiques


# ─────────────────────────────────────────────────────────────────────
#  Phase 2 runner (full pipeline)
# ─────────────────────────────────────────────────────────────────────

def run_phase2(request: ArbitrationRequest) -> FinalVerdict:
    from orchestration import run_arbitration_sync
    console.print("\n[bold cyan]Running full LangGraph pipeline...[/bold cyan]\n")
    return run_arbitration_sync(request)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

async def main(dry_run: bool, critic: Optional[str], phase: int) -> None:
    configure_logging()

    console.print(Panel.fit(
        "[bold]LLM Output Arbitration System[/bold] -- Phase 2 Smoke Test",
        style="bold blue",
    ))

    console.print("\n[bold]Step 1 -- Configuration[/bold]")
    ready = check_config()

    if dry_run:
        console.print("\n[yellow]--dry-run: skipping LLM calls.[/yellow]")
        if not ready:
            console.print(
                "[dim]Fill in .env with your API keys, then run without --dry-run.[/dim]"
            )
        sys.exit(0)

    if not ready:
        console.print(
            "\n[yellow]Some API keys are missing. Affected critics will use "
            "fallback models or return error critiques.[/yellow]\n"
        )

    console.print("\n[bold]Step 2 -- Evaluation Target[/bold]")
    console.print(Panel(SAMPLE_PROMPT, title="Prompt", border_style="dim"))
    console.print(Panel(SAMPLE_RESPONSE, title="LLM Response (has intentional bugs)", border_style="dim"))

    request = ArbitrationRequest(
        original_prompt=SAMPLE_PROMPT,
        llm_response=SAMPLE_RESPONSE,
        session_id="smoke-test-p2",
    )

    if phase == 1 or critic:
        # Phase 1: run critics individually
        console.print(f"\n[bold]Step 3 -- Phase 1: Individual Critics[/bold]")
        critiques = await run_phase1(request, selected=critic)
        console.print("\n[bold]Step 4 -- Results[/bold]\n")
        for c in critiques:
            render_critique(c)

        # Summary table
        table = Table(title="Summary", box=box.SIMPLE_HEAVY)
        table.add_column("Dimension", style="bold")
        table.add_column("Score", justify="center")
        table.add_column("Grade", justify="center")
        table.add_column("Issues", justify="center")
        table.add_column("Verdict")
        total = 0
        for c in critiques:
            sty = SCORE_STYLE.get(c.grade, "white")
            v   = Text("PASS", style="green") if c.passed else Text("FAIL", style="red")
            table.add_row(c.dimension.value, str(c.score), Text(c.grade, style=sty), str(len(c.issues)), v)
            total += c.score
        console.print(table)
        if critiques:
            console.print(f"\n  [bold]Aggregate score: {total // len(critiques)}/100[/bold]")

    else:
        # Phase 2: full pipeline
        console.print(f"\n[bold]Step 3 -- Phase 2: Full LangGraph Pipeline[/bold]")
        try:
            verdict = run_phase2(request)
        except Exception as exc:
            console.print(f"\n[bold red]Pipeline error: {exc}[/bold red]")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        console.print("\n[bold]Step 4 -- Individual Critic Results[/bold]\n")
        for c in verdict.critiques:
            render_critique(c)

        console.print("\n[bold]Step 5 -- Final Verdict[/bold]\n")
        render_verdict(verdict)

    console.print("\n[dim]Smoke test complete.[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Arbitration System smoke test")
    parser.add_argument("--dry-run",  action="store_true", help="Config check only")
    parser.add_argument("--critic",   type=str, default=None,
                        help="Run one critic: accuracy|logic|completeness|safety|style")
    parser.add_argument("--phase",    type=int, default=2, choices=[1, 2],
                        help="1=individual critics, 2=full pipeline (default)")
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, critic=args.critic, phase=args.phase))
