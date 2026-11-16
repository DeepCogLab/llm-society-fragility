r"""Write the two LaTeX table bodies of plan A from numbers.json, so that no number is typed by hand.
Output: paper/tables/tab_main.tex and tab_nl.tex (read by main.tex).

Styling conventions (macros and colours are defined in main.tex):
  \band{n}{title}   shaded group header spanning n columns
  \mk{colour}       small square in the colour the condition has in the figures
  \cellcolor{heat!p} background that scales with the excess error
  bold estimate      95% CI excludes zero
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)
N = json.loads((HERE / "numbers.json").read_text(encoding="utf-8"))
MODELS = ["Qwen2.5-7B", "Qwen3-14B", "Qwen3-30B", "GLM-4-9B", "Llama-3.1-8B"]
RED, BLUE, GREY = r"\mk{onered}", r"\mk{recblue}", r"\mk{relaygrey}"


def e(m, grp, s, a):
    return N["effects"].get(f"{m}|{grp}|{s}|{a}")


def num(x):
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s


def heat(x):
    p = int(round(min(55.0, max(0.0, x) * 0.75)))
    return rf"\cellcolor{{heat!{p}}}" if p >= 3 else ""


def cell(v, with_ci=True, shade=True, dagger=False):
    """v = [estimate, lo, hi]."""
    est = num(v[0])
    if dagger:  # control arm at chance: neither shaded nor marked as significant
        body = rf"\ci{{{est}}}{{{num(v[1])}}}{{{num(v[2])}}}" if with_ci else est
        return body + r"\rlap{$^\dagger$}"
    if v[1] > 0 or v[2] < 0:
        est = rf"\textbf{{{est}}}"
    body = rf"\ci{{{est}}}{{{num(v[1])}}}{{{num(v[2])}}}" if with_ci else est
    return (heat(v[0]) if shade else "") + body


def row(label, cells):
    return f"    {label} & " + " & ".join(cells) + r" \\"


def delta_row(label, grp, s, a, key="tail", dagger_llama=False, with_ci=True, shade=True):
    return row(label, [cell(e(m, grp, s, a)[key], with_ci, shade, dagger_llama and m == "Llama-3.1-8B") for m in MODELS])


# ------------------------------------------------------------------ Table 1: abstract task
L = []
L.append(r"    \band{6}{Reference values} \\")
L.append(row(r"Error without attack (\%), one-time signal", [num(e(m, "abs", "initial_only", "treated")["ctrl_tail"]) for m in MODELS]))
L.append(row(r"Error without attack (\%), recurrent signal", [num(e(m, "abs", "persistent", "treated")["ctrl_tail"]) for m in MODELS]))
L.append(delta_row(r"Shock at round 3 (pp), one-time signal", "abs", "initial_only", "treated", key="peak3", shade=False))
L.append(r"    \band{6}{Persistence: excess error $\Delta$ in rounds 6--8 (pp)} \\")
L.append(delta_row(RED + r" One-time signal", "abs", "initial_only", "treated", dagger_llama=True))
L.append(delta_row(RED + r" \quad with full answer history", "abs", "init_fullmem", "treated", dagger_llama=True))
L.append(delta_row(RED + r" \quad with six seed agents", "abs", "init_seed6", "treated", dagger_llama=True))
L.append(delta_row(GREY + r" \quad with relay blocked", "abs", "initial_only", "no_relay_treated"))
L.append(delta_row(RED + r" Signal listed in history", "abs", "in_history", "treated"))
L.append(delta_row(BLUE + r" Recurrent signal", "abs", "persistent", "treated"))
L.append(delta_row(BLUE + r" \quad with six seed agents", "abs", "persist_seed6", "treated"))
(OUT / "tab_main.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ Table 2: natural-language task
kw = dict(with_ci=False)
L = []
L.append(r"    \band{6}{One-time signal} \\")
L.append(delta_row(RED + r" Authority claim", "nl", "initial_only", "treated", **kw))
L.append(delta_row(RED + r" Neutral wording", "nl", "initial_only", "treated_neutral", **kw))
L.append(delta_row(GREY + r" Relay blocked", "nl", "initial_only", "no_relay_treated", **kw))
L.append(r"    \band{6}{Recurrent signal} \\")
L.append(delta_row(BLUE + r" Authority claim", "nl", "persistent", "treated", **kw))
L.append(delta_row(GREY + r" Relay blocked", "nl", "persistent", "no_relay_treated", **kw))
(OUT / "tab_nl.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
hw = max((v["tail"][2] - v["tail"][1]) / 2 for k, v in N["effects"].items() if "|nl|" in k)
print((OUT / "tab_main.tex").read_text(encoding="utf-8"))
print((OUT / "tab_nl.tex").read_text(encoding="utf-8"))
print("max CI half-width in table 2:", round(hw, 1))
