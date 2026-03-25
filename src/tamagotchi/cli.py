"""CLI entry point — Typer-based commands for Tamagotchi."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tamagotchi.memory.store import MemoryStore

app = typer.Typer(
    name="tamagotchi",
    help="당신만의 AI 다마고치를 키워보세요!",
    no_args_is_help=True,
)
profile_app = typer.Typer(help="학습된 프로필 관리")
app.add_typer(profile_app, name="profile")

console = Console()


def _get_store() -> MemoryStore:
    return MemoryStore()


@app.command()
def chat(
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="사용할 모델. Anthropic: sonnet/haiku/opus. OpenAI: 모델 이름 (예: Qwen/Qwen3-8B)",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="LLM 제공자 (anthropic 또는 openai). 환경변수 TAMAGOTCHI_PROVIDER도 가능.",
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url",
        help="OpenAI-compatible 엔드포인트 URL (예: http://localhost:8000/v1)",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="API 키 (self-hosted는 보통 불필요)",
    ),
) -> None:
    """다마고치와 대화를 시작합니다."""
    from tamagotchi.agent.core import TamagotchiAgent

    store = _get_store()
    try:
        agent = TamagotchiAgent(
            store, model=model, provider=provider,
            base_url=base_url, api_key=api_key,
        )
        agent.chat_loop()
    finally:
        store.close()


@app.command()
def models() -> None:
    """사용 가능한 모델 목록을 표시합니다."""
    from tamagotchi.config import MODELS, get_chat_model

    current = get_chat_model()

    console.print("[bold]Anthropic 모델:[/bold]")
    table = Table(show_header=True)
    table.add_column("이름", style="cyan")
    table.add_column("설명", style="white")
    table.add_column("대화당 비용", style="yellow", justify="right")
    table.add_column("", style="green")

    for key, info in MODELS.items():
        marker = "← 현재" if info["id"] == current else ""
        table.add_row(key, info["description"], info["cost_per_chat"], marker)

    console.print(table)

    console.print("\n[bold]Self-hosted (OpenAI-compatible):[/bold]")
    console.print("[dim]vLLM, SGLang, Ollama 등으로 띄운 모델을 사용할 수 있습니다.[/dim]")
    console.print()
    console.print("[bold cyan]사용법:[/bold cyan]")
    console.print("  [dim]# Anthropic[/dim]")
    console.print("  tamagotchi chat --model haiku")
    console.print()
    console.print("  [dim]# Self-hosted (vLLM / SGLang)[/dim]")
    console.print("  tamagotchi chat --provider openai --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B")
    console.print()
    console.print("  [dim]# Ollama[/dim]")
    console.print("  tamagotchi chat --provider openai --base-url http://localhost:11434/v1 --model llama3")
    console.print()
    console.print("  [dim]# 환경변수로 설정[/dim]")
    console.print("  export TAMAGOTCHI_PROVIDER=openai")
    console.print("  export TAMAGOTCHI_BASE_URL=http://localhost:8000/v1")
    console.print("  export TAMAGOTCHI_MODEL=Qwen/Qwen3-8B")
    console.print("  tamagotchi chat")


@app.command()
def status() -> None:
    """다마고치의 성장 상태를 확인합니다."""
    from tamagotchi.growth.display import show_status
    from tamagotchi.growth.state import GrowthManager

    store = _get_store()
    try:
        growth = GrowthManager(store)
        show_status(growth, console, store=store)
    finally:
        store.close()


@profile_app.callback(invoke_without_command=True)
def profile_show(ctx: typer.Context) -> None:
    """학습된 사용자 프로필을 조회합니다."""
    if ctx.invoked_subcommand is not None:
        return

    from tamagotchi.memory.profile import ProfileManager

    store = _get_store()
    try:
        mgr = ProfileManager(store)
        profile = mgr.load()

        if not profile.preferences:
            console.print("[dim]아직 학습된 선호도가 없습니다. 대화를 시작해보세요![/dim]")
            return

        table = Table(title="학습된 선호도", show_lines=True)
        table.add_column("카테고리", style="cyan")
        table.add_column("항목", style="white")
        table.add_column("내용", style="green")
        table.add_column("확신도", style="yellow", justify="right")
        table.add_column("출처", style="dim")

        for p in profile.preferences:
            conf_bar = "●" * int(p.confidence * 5) + "○" * (5 - int(p.confidence * 5))
            table.add_row(p.category, p.key, p.value, conf_bar, p.source)

        console.print(table)
    finally:
        store.close()


@profile_app.command("forget")
def profile_forget(
    category: str = typer.Argument(help="삭제할 카테고리 (예: food, shopping)"),
    key: Optional[str] = typer.Argument(None, help="삭제할 특정 항목 (생략하면 카테고리 전체 삭제)"),
) -> None:
    """특정 선호도를 삭제합니다."""
    from tamagotchi.memory.profile import ProfileManager

    store = _get_store()
    try:
        mgr = ProfileManager(store)
        count = mgr.forget(category, key)
        if count > 0:
            target = f"{category}/{key}" if key else category
            console.print(f"[green]{target} 관련 선호도 {count}개를 삭제했습니다.[/green]")
        else:
            console.print("[yellow]해당하는 선호도가 없습니다.[/yellow]")
    finally:
        store.close()


@app.command()
def history(limit: int = typer.Option(10, help="표시할 대화 수")) -> None:
    """최근 대화 히스토리를 조회합니다."""
    store = _get_store()
    try:
        episodes = store.get_recent_episodes(limit)
        if not episodes:
            console.print("[dim]대화 기록이 없습니다.[/dim]")
            return

        for ep in episodes:
            console.print(Panel(
                f"[white]{ep['summary']}[/white]\n[dim]{ep['created_at'][:19]}[/dim]",
                border_style="blue",
            ))
    finally:
        store.close()


@app.command()
def reset(
    confirm: bool = typer.Option(False, "--yes", "-y", help="확인 없이 리셋"),
) -> None:
    """모든 데이터를 초기화합니다."""
    if not confirm:
        confirm = typer.confirm("정말로 모든 데이터를 초기화하시겠습니까?")
    if not confirm:
        console.print("[yellow]취소되었습니다.[/yellow]")
        return

    import shutil
    from tamagotchi.memory.store import DEFAULT_DB_DIR

    if DEFAULT_DB_DIR.exists():
        shutil.rmtree(DEFAULT_DB_DIR)
    console.print("[green]모든 데이터가 초기화되었습니다. 다마고치가 알(Egg)에서 다시 시작합니다![/green]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="서버 호스트"),
    port: int = typer.Option(8000, help="서버 포트"),
) -> None:
    """웹 서버를 시작합니다."""
    try:
        from tamagotchi.api.server import run_server
    except ImportError:
        console.print("[red]웹 서버를 시작하려면 web 의존성을 설치하세요: uv sync --extra web[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Tamagotchi 웹 서버 시작: http://{host}:{port}[/bold green]")
    console.print("[dim]Ctrl+C로 중지합니다.[/dim]")
    run_server(host=host, port=port)


@app.command("export")
def export_data(
    output: str = typer.Argument("tamagotchi_backup.json", help="출력 파일 경로"),
) -> None:
    """모든 데이터를 JSON 파일로 내보냅니다."""
    from pathlib import Path
    from tamagotchi.data_manager import export_to_file

    store = _get_store()
    try:
        path = Path(output)
        export_to_file(store, path)
        console.print(f"[green]데이터를 {path}에 내보냈습니다.[/green]")
    finally:
        store.close()


@app.command("import")
def import_data(
    input_file: str = typer.Argument(help="가져올 JSON 파일 경로"),
) -> None:
    """JSON 파일에서 데이터를 가져옵니다."""
    from pathlib import Path
    from tamagotchi.data_manager import import_from_file

    path = Path(input_file)
    if not path.exists():
        console.print(f"[red]파일을 찾을 수 없습니다: {path}[/red]")
        raise typer.Exit(1)

    store = _get_store()
    try:
        counts = import_from_file(store, path)
        console.print("[green]데이터 가져오기 완료:[/green]")
        for key, count in counts.items():
            if count > 0:
                console.print(f"  {key}: {count}개")
    finally:
        store.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
