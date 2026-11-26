#!/usr/bin/env python3
"""fold_profile.py - summarize Rust Windows sampling profiles."""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SYSTEM_WAIT_MODULES = {"ntdll.dll", "kernelbase.dll", "win32u.dll"}


@dataclass
class ThreadRecord:
    tid: int
    start: str
    cpu_ms: int
    samples: int
    name: str = ""


@dataclass
class StackRecord:
    tid: int
    phase: int
    count: int
    frames: List[str]


def parse_profile(path: str) -> Tuple[Optional[int], Dict[int, int], Dict[int, ThreadRecord], List[StackRecord]]:
    """Parse the input profile file. Unknown/blank lines are ignored."""
    rounds = None
    phases: Dict[int, int] = {}
    threads: Dict[int, ThreadRecord] = {}
    stacks: List[StackRecord] = []

    with open(path, "r", encoding="utf-8-sig", newline=None) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            command_parts = line.split()
            if not command_parts:
                continue

            cmd = command_parts[0].lower()

            if cmd == "rounds":
                try:
                    if len(command_parts) >= 2:
                        rounds = int(command_parts[1])
                except ValueError:
                    pass

            elif cmd == "phase":
                try:
                    if len(command_parts) >= 3:
                        phase_id = int(command_parts[1])
                        count = int(command_parts[2])
                        phases[phase_id] = count
                except ValueError:
                    pass

            elif cmd == "thread":
                # Name is the rest of the line and may contain spaces or be
                # missing; the start is empty for a thread that could not be
                # opened (two spaces in a row).
                m = re.match(r"thread (\d+) (\S*) (\d+) (\d+) ?(.*)$", raw_line.rstrip("\r\n"))
                if m:
                    tid = int(m.group(1))
                    threads[tid] = ThreadRecord(tid, m.group(2), int(m.group(3)), int(m.group(4)), m.group(5).strip())

            elif cmd == "stack":
                sp = line.split(maxsplit=4)
                try:
                    if len(sp) >= 5:
                        tid = int(sp[1])
                        phase = int(sp[2])
                        count = int(sp[3])
                        frame_str = sp[4].strip()
                        frames = [f.strip() for f in frame_str.split(";") if f.strip()]
                        stacks.append(StackRecord(tid, phase, count, frames))
                except ValueError:
                    pass

    return rounds, phases, threads, stacks


def split_module(frame: str) -> Optional[str]:
    """Return module part of `module+0x...`, else None."""
    if "+" not in frame:
        return None
    module = frame.split("+", 1)[0].strip()
    return module or None


def fallback_label(frame: str) -> str:
    """Collapse other modules to module name; keep raw hex and raw victory frames."""
    module = split_module(frame)
    return module if module else frame


def is_victory_frame(frame: str, exe: Optional[str]) -> bool:
    if not exe:
        return False
    module = split_module(frame)
    if not module:
        return False
    exe_base = os.path.basename(exe.replace("\\", "/")).lower()
    mod_base = os.path.basename(module.replace("\\", "/")).lower()
    return mod_base == exe_base


def strip_rust_hash(name: str) -> str:
    """Strip trailing Rust hash suffixes like ::h0123456789abcdef."""
    return re.sub(r"::h[0-9a-fA-F]{16}$", "", name)


_LLVM_PDB_SUPPORT: Optional[bool] = None


def llvm_supports_pdb() -> bool:
    global _LLVM_PDB_SUPPORT
    if _LLVM_PDB_SUPPORT is None:
        try:
            proc = subprocess.run(
                ["llvm-symbolizer", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            _LLVM_PDB_SUPPORT = "--pdb" in out
        except Exception:
            _LLVM_PDB_SUPPORT = False
    return _LLVM_PDB_SUPPORT


def parse_symbolizer_output(text: str, frames: List[str]) -> Dict[str, str]:
    """Parse llvm-symbolizer output: function line, then file:line:col, blank-separated."""
    mapping: Dict[str, str] = {}
    blocks: List[List[str]] = []
    current: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(stripped)

    if current:
        blocks.append(current)

    for i, frame in enumerate(frames):
        if i >= len(blocks):
            break
        block = blocks[i]
        if not block:
            continue
        name = block[0]
        if not name or name == "??":
            continue
        name = strip_rust_hash(name)
        if name and name != "??":
            mapping[frame] = name

    return mapping


def run_llvm_symbolizer(
    frames: List[str],
    exe: str,
    image_base: int,
    pdb: Optional[str],
) -> Dict[str, str]:
    """Run llvm-symbolizer once in batch mode for all requested frames."""
    try:
        addresses = []
        for frame in frames:
            rva_str = frame.split("+", 1)[1].strip()
            rva = int(rva_str, 16)
            addresses.append(f"0x{image_base + rva:x}")

        cmd = [
            "llvm-symbolizer",
            f"--obj={exe}",
            "--functions=linkage",
            "--demangle",
            "--inlines=false",
            "--output-style=LLVM",
        ]

        if pdb and llvm_supports_pdb():
            cmd.append(f"--pdb={pdb}")

        proc = subprocess.run(
            cmd,
            input="\n".join(addresses) + "\n",
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return {}

        return parse_symbolizer_output(proc.stdout, frames)

    except Exception:
        return {}


def create_display_map(
    threads: Dict[int, ThreadRecord],
    stacks: List[StackRecord],
    exe: Optional[str],
    image_base: int = 0x140000000,
    pdb: Optional[str] = None,
) -> Dict[str, str]:
    """Map every raw frame/start to its aggregation label."""
    all_frames = set()
    for t in threads.values():
        all_frames.add(t.start)
    for s in stacks:
        all_frames.update(s.frames)

    symbol_map: Dict[str, str] = {}

    if exe:
        victory_frames = sorted([f for f in all_frames if is_victory_frame(f, exe)])
        if victory_frames:
            symbol_map = run_llvm_symbolizer(victory_frames, exe, image_base, pdb)

    display_map: Dict[str, str] = {}
    for frame in all_frames:
        if exe and is_victory_frame(frame, exe):
            symbol = symbol_map.get(frame)
            display_map[frame] = symbol if symbol else frame
        else:
            display_map[frame] = fallback_label(frame)

    return display_map


class Aggregation:
    def __init__(self):
        self.self_total = defaultdict(int)
        self.inclusive_total = defaultdict(int)
        self.self_by_thread = defaultdict(lambda: defaultdict(int))
        self.inclusive_by_thread = defaultdict(lambda: defaultdict(int))
        self.total_by_thread = defaultdict(int)


def aggregate(stacks: List[StackRecord], display_map: Dict[str, str]) -> Aggregation:
    agg = Aggregation()
    for s in stacks:
        agg.total_by_thread[s.tid] += s.count
        labels = [display_map.get(f, fallback_label(f)) for f in s.frames]
        if not labels:
            continue

        self_label = labels[0]  # innermost frame is first
        agg.self_total[self_label] += s.count
        agg.self_by_thread[s.tid][self_label] += s.count

        # Inclusive: count a function once per stack, even with recursion.
        for label in set(labels):
            agg.inclusive_total[label] += s.count
            agg.inclusive_by_thread[s.tid][label] += s.count

    return agg


def waiting_percent(agg: Aggregation, tid: int) -> float:
    total = agg.total_by_thread.get(tid, 0)
    if total <= 0:
        return 0.0
    waiting = sum(
        count
        for label, count in agg.self_by_thread[tid].items()
        if label.lower() in SYSTEM_WAIT_MODULES
    )
    return 100.0 * waiting / total


def print_header(rounds: Optional[int], phases: Dict[int, int]) -> None:
    print(f"rounds: {rounds if rounds is not None else '?'}")
    total_phase = sum(phases.values())
    base = rounds if rounds and rounds > 0 else total_phase
    if base <= 0:
        base = 1
    for phase_id in sorted(phases):
        count = phases[phase_id]
        pct = 100.0 * count / base
        print(f"phase {phase_id}: {count} ({pct:.1f}%)")


def print_threads_table(threads: Dict[int, ThreadRecord], display_map: Dict[str, str]) -> None:
    print("\nThreads (sorted by cpu_ms desc):")
    print(f"{'id':<6}{'cpu_ms':<10}{'samples':<10}{'start':<45}{'name'}")
    for t in sorted(threads.values(), key=lambda t: (-t.cpu_ms, t.tid)):
        start = display_map.get(t.start, fallback_label(t.start))
        print(f"{t.tid:<6}{t.cpu_ms:<10}{t.samples:<10}{start:<45}{t.name}")


def print_top_counts(counts: Dict[str, int], total: int, top_n: int, indent: str = "  ") -> None:
    if not counts:
        print(f"{indent}(no samples)")
        return
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"{indent}{'count':>8} {'pct':>7}  function")
    for label, count in items[:top_n]:
        pct = 100.0 * count / total if total else 0.0
        print(f"{indent}{count:>8} {pct:6.1f}%  {label}")


def print_report(
    rounds: Optional[int],
    phases: Dict[int, int],
    threads: Dict[int, ThreadRecord],
    agg: Aggregation,
    display_map: Dict[str, str],
    top_n: int = 40,
    thread_top_n: int = 8,
) -> None:
    print_header(rounds, phases)
    print_threads_table(threads, display_map)

    thread_list = sorted(threads.values(), key=lambda t: (-t.cpu_ms, t.tid))
    for t in thread_list[:thread_top_n]:
        total = agg.total_by_thread.get(t.tid, 0)
        name = t.name if t.name else "(unnamed)"
        print(f"\nThread {t.tid}: {name} (cpu_ms={t.cpu_ms}, samples={t.samples}, stack_samples={total})")

        print("Top self:")
        print_top_counts(agg.self_by_thread[t.tid], total, top_n)

        print("Top inclusive:")
        print_top_counts(agg.inclusive_by_thread[t.tid], total, top_n)

        wp = waiting_percent(agg, t.tid)
        print(f"waiting %: {wp:.1f}%")

    total_all = sum(agg.total_by_thread.values())
    print(f"\nCombined top self (all threads, total={total_all}):")
    print_top_counts(agg.self_total, total_all, top_n)

    print(f"\nCombined top inclusive (all threads, total={total_all}):")
    print_top_counts(agg.inclusive_total, total_all, top_n)


def write_folded(
    path: str,
    threads: Dict[int, ThreadRecord],
    stacks: List[StackRecord],
    display_map: Dict[str, str],
) -> None:
    folded = defaultdict(int)

    for s in stacks:
        labels = [display_map.get(f, fallback_label(f)) for f in s.frames]
        t = threads.get(s.tid)
        thread_name = t.name if t and t.name else f"thread-{s.tid}"
        thread_name = thread_name.replace(";", "_")

        # Input order is innermost first; folded format wants outermost first.
        folded_line = ";".join([thread_name] + list(reversed(labels)))
        folded[folded_line] += s.count

    with open(path, "w", encoding="utf-8") as fh:
        for folded_line, count in sorted(folded.items()):
            fh.write(f"{folded_line} {count}\n")


def run_selftest() -> None:
    profile = """rounds 100
phase 0 60
phase 1 40
thread 1 0x140001000 80 70 main
thread 2 ntdll.dll+0x2000 20 50 worker
stack 1 0 30 0x1000;0x2000;victory.exe+0x3000;
stack 1 0 40 0x2000;victory.exe+0x3000;
stack 2 1 30 ntdll.dll+0x2000;ntdll.dll+0x3000;0x4000;
stack 2 1 20 0x5000;0x5000;0x6000;
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "profile.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(profile)

        rounds, phases, threads, stacks = parse_profile(path)
        # No --exe means no llvm-symbolizer invocation.
        display_map = create_display_map(threads, stacks, exe=None, image_base=0x140000000, pdb=None)
        agg = aggregate(stacks, display_map)

        assert rounds == 100
        assert phases == {0: 60, 1: 40}
        assert set(threads.keys()) == {1, 2}
        assert agg.total_by_thread[1] == 70
        assert agg.total_by_thread[2] == 50

        assert dict(agg.self_by_thread[1]) == {"0x1000": 30, "0x2000": 40}
        assert dict(agg.inclusive_by_thread[1]) == {
            "0x1000": 30,
            "0x2000": 70,
            "victory.exe": 70,
        }

        assert dict(agg.self_by_thread[2]) == {"ntdll.dll": 30, "0x5000": 20}
        assert dict(agg.inclusive_by_thread[2]) == {
            "ntdll.dll": 30,
            "0x4000": 30,
            "0x5000": 20,
            "0x6000": 20,
        }

        # Recursion on 0x5000: appears twice in one stack but counted once.
        assert agg.inclusive_by_thread[2]["0x5000"] == 20
        assert abs(waiting_percent(agg, 2) - 60.0) < 1e-9

    print("selftest OK")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a Rust Windows sampling-profile text file."
    )
    parser.add_argument("profile", nargs="?", help="Path to PROFILE.txt")
    parser.add_argument("--exe", help="Path to the main executable, e.g. victory.exe")
    parser.add_argument("--pdb", help="Optional PDB path to pass to llvm-symbolizer")
    parser.add_argument("--image-base", default="0x140000000", help="Image base address (default: 0x140000000)")
    parser.add_argument("--top", type=int, default=40, help="Number of top functions to show (default: 40)")
    parser.add_argument("--thread-top", type=int, default=8, help="Number of top threads to show (default: 8)")
    parser.add_argument("--folded", help="Write folded stacks to this path")
    parser.add_argument("--selftest", action="store_true", help="Run built-in synthetic selftest")
    args = parser.parse_args(argv)

    if args.selftest:
        run_selftest()
        return 0

    if not args.profile:
        parser.error("PROFILE is required unless --selftest")

    image_base = int(args.image_base, 0)

    rounds, phases, threads, stacks = parse_profile(args.profile)
    display_map = create_display_map(threads, stacks, args.exe, image_base, args.pdb)
    agg = aggregate(stacks, display_map)

    print_report(rounds, phases, threads, agg, display_map, args.top, args.thread_top)

    if args.folded:
        write_folded(args.folded, threads, stacks, display_map)

    return 0


if __name__ == "__main__":
    sys.exit(main())
