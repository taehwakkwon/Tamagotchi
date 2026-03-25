"""Terminal display for Tamagotchi status — ASCII art, stats, and personality."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns

from tamagotchi.growth.personality import PersonalityManager, PersonalityTraits
from tamagotchi.growth.state import LEVELS, GrowthManager
from tamagotchi.memory.store import MemoryStore

# ASCII art for each growth level
ASCII_ART: dict[int, str] = {
    1: r"""
      ___
     /   \
    | . . |
     \ _ /
      \_/
    """,
    2: r"""
      ___
     / o \
    | ^.^ |
     \_v_/
      |_|
    """,
    3: r"""
     \\ //
      (o.o)
     /(   )\
      |   |
     _/   \_
    """,
    4: r"""
     \\   //
      (^.^)
    </(   )\>
      || ||
     _//  \\_
    """,
    5: r"""
    *\\   //*
      (*.*)
   ⚡/(   )\⚡
      || ||
    _///  \\\_
    """,
    6: r"""
    ★ \\   // ★
       (@.@)
    ⚡⚡/(   )\⚡⚡
       || ||
   ★_///  \\\_ ★
      ~MASTER~
    """,
}

# Mood expressions based on personality
MOOD_FACES: dict[str, str] = {
    "따뜻한": "(♥.♥)",
    "유머러스한": "(^▽^)",
    "호기심 많은": "(◉.◉)",
    "격식있는": "(−.−)",
    "활발한": "(★.★)",
    "공감적인": "(◡.◡)",
    "쿨한": "(−_−)",
    "진지한": "(⊙_⊙)",
    "차분한": "(─.─)",
}


def show_status(
    growth: GrowthManager,
    console: Console | None = None,
    store: MemoryStore | None = None,
) -> None:
    """Display the full Tamagotchi status panel in the terminal."""
    if console is None:
        console = Console()

    state = growth.get_state()
    level = state["level"]
    level_info = LEVELS[level]
    xp_to_next = growth.xp_to_next_level()

    # Load personality if store available
    personality: PersonalityTraits | None = None
    if store:
        personality = PersonalityManager(store).load()

    # ASCII art panel
    art = ASCII_ART.get(level, ASCII_ART[1])

    # Add mood face if personality has dominant traits
    subtitle = ""
    if personality:
        dominant = personality.get_dominant_traits()
        if dominant:
            face = MOOD_FACES.get(dominant[0], "")
            subtitle = f"[dim]{face}[/dim]" if face else ""

    console.print(Panel(
        Text(art, style="bold yellow", justify="center"),
        title=f"[bold magenta]Lv.{level} {level_info['name']}[/bold magenta]",
        subtitle=subtitle,
        border_style="magenta",
    ))

    # Stats table
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("stat", style="bold cyan")
    stats_table.add_column("value", style="white")

    stats_table.add_row("레벨", f"{level} - {level_info['name']}")
    stats_table.add_row("설명", level_info["description"])
    stats_table.add_row("XP", f"{state['xp']:,}")
    if xp_to_next is not None:
        stats_table.add_row("다음 레벨까지", f"{xp_to_next:,} XP")
    else:
        stats_table.add_row("다음 레벨까지", "MAX LEVEL")

    # Progress bar
    if xp_to_next is not None and level + 1 in LEVELS:
        next_req = LEVELS[level + 1]["min_xp"]
        curr_req = LEVELS[level]["min_xp"]
        progress = (state["xp"] - curr_req) / max(1, next_req - curr_req)
        progress = min(1.0, max(0.0, progress))
        bar_len = 20
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        stats_table.add_row("진행도", f"[{bar}] {progress:.0%}")

    stats_table.add_row("", "")
    stats_table.add_row("총 대화", f"{state['total_conversations']:,}회")
    stats_table.add_row("학습한 선호도", f"{state['total_preferences']:,}개")
    stats_table.add_row("추천 수락", f"{state['total_recommendations_accepted']:,}회")

    console.print(Panel(stats_table, title="[bold cyan]Stats[/bold cyan]", border_style="cyan"))

    # Personality panel
    if personality:
        _show_personality(personality, console)


def _show_personality(traits: PersonalityTraits, console: Console) -> None:
    """Display personality traits as a visual panel."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("trait", style="bold", width=8)
    table.add_column("bar", width=24)
    table.add_column("label", style="dim", width=14)

    trait_display = [
        ("따뜻함", traits.warmth, "쿨한", "따뜻한"),
        ("유머", traits.humor, "진지한", "유머러스"),
        ("호기심", traits.curiosity, "과묵한", "호기심"),
        ("격식", traits.formality, "편한", "격식있는"),
        ("에너지", traits.energy, "차분한", "활발한"),
        ("공감", traits.empathy, "이성적", "공감적"),
    ]

    for name, val, low_label, high_label in trait_display:
        bar_len = 16
        pos = int(val * bar_len)
        bar = "░" * pos + "●" + "░" * (bar_len - pos)
        label = high_label if val > 0.6 else (low_label if val < 0.4 else "보통")
        table.add_row(name, f"[yellow]{bar}[/yellow]", label)

    summary = traits.summary()
    console.print(Panel(
        table,
        title=f"[bold yellow]Personality[/bold yellow]",
        subtitle=f"[dim]{summary}[/dim]",
        border_style="yellow",
    ))
