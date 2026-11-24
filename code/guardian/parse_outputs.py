"""Parse GUARDIAN run logs (outputs/*.txt) into tidy CSVs.

Writes two files:
  questions.csv  one row per (run, question): run settings, accuracy, attack outcome,
                 pruning/detection metrics, consensus behaviour, per-round agent agreement
  agents.csv     one row per (run, question, round, agent): the agent's parsed answer,
                 whether it is correct / the injected answer, its anomaly score, pruning

Usage:
    python parse_outputs.py                       # reads outputs/*.txt, writes outputs/parsed/
    python parse_outputs.py --out-dir results --include-text   # also keep full agent replies

Agent answers are re-extracted from the printed reply table with the same parsers the run
used (extract_math_answer / parse_single_choice), so they match what the debate saw.
"""
import argparse
import glob
import json
import math
import os
import re
import shlex
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import extract_math_answer, parse_single_choice, answers_equiv, most_frequent  # noqa: E402
from tasks import parse_choice  # noqa: E402

try:
    from wcwidth import wcwidth
except ImportError:
    wcwidth = None

ROUNDS = 3
NO_PRUNE_K = 1000  # --prune threshold with a huge k (9999) is how the "nopruning" runs were made
INJECTED_TOL = 1e-3  # relative tolerance for "agent adopted the injected number" (it is a 6-dp float)

RE_QUESTION = re.compile(r"^\[(.+?)\] Question (\d+)/(\d+)$")
RE_SCORES = re.compile(r"^\[.+?\] Round (\d+) anomaly scores: (.*?)(?:, threshold ([-\d.e]+))? -> pruned \[(.*)\]$")
RE_AGENT_SCORE = re.compile(r"agent (\d+): ([-\d.eE+na]+)")
RE_FINISHED = re.compile(r"Query finished in ([\d.]+)s")


# ---------------------------------------------------------------- run settings

def parse_run_settings(header, path):
    argv = shlex.split(header.split("Run started:", 1)[1])
    opts = {"task": "math", "attack": "none", "detector": "temporal", "model": "gpt-4o-mini",
            "agents": "4", "n-samples": "100", "math-levels": "all", "attack-agents": "0",
            "prune": "top1", "prune-k": "1.0", "min-agents": "2"}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                opts[key] = argv[i + 1]
                i += 2
                continue
            opts[key] = "true"
        i += 1

    n_agents = int(opts["agents"])
    aa = opts["attack-agents"]
    if opts["attack"] != "agent":
        attackers = []
    elif aa.startswith("random:"):
        attackers = None  # resolved from the "Attack agents:" lines
    else:
        attackers = sorted({int(p) for p in aa.split(",")})
    prune_k = float(opts["prune-k"])
    prune_mode = "none" if opts["prune"] == "threshold" and prune_k >= NO_PRUNE_K else opts["prune"]
    return {
        "run_file": os.path.basename(path),
        "run_label": os.path.splitext(os.path.basename(path))[0],
        "run_args": header.split("Run started:", 1)[1].strip(),
        "task": opts["task"],
        "attack": opts["attack"],
        "detector": opts["detector"],
        "model": opts["model"],
        "n_agents": n_agents,
        "attack_agents": attackers,
        "prune_mode": prune_mode,
        "prune_k": prune_k if opts["prune"] == "threshold" else None,
        "min_agents": int(opts["min-agents"]),
        "math_levels": opts["math-levels"] if opts["task"] == "math" else None,
    }


# ---------------------------------------------------------------- reply table

def display_index_map(line):
    """Map display column -> string index, accounting for wide characters."""
    cols = []
    for idx, ch in enumerate(line):
        w = wcwidth(ch) if wcwidth else 1
        cols.extend([idx] * (1 if w < 0 else w))  # w == 0: combining char takes no column
    return cols


def slice_row(line, bounds, border_len):
    """Split one table line into cell strings using the '+' positions of the border."""
    if len(line) != border_len:
        cmap = display_index_map(line)
        cmap.append(len(line))
        to_idx = lambda c: cmap[min(c, len(cmap) - 1)]
    else:
        to_idx = lambda c: c
    return [line[to_idx(a) + 1:to_idx(b)] for a, b in zip(bounds, bounds[1:])]


def parse_table(lines, start):
    """Parse the PrettyTable starting at lines[start] (top border). Returns (rows, end index).
    rows: list of (role, [reply per round or None])."""
    border = lines[start]
    bounds = [i for i, c in enumerate(border) if c == "+"]
    i = start + 3  # border, header, border
    rows = []
    while i < len(lines) and lines[i] != border:
        cells = slice_row(lines[i], bounds, len(border))
        role = cells[0].strip()
        if role:
            rows.append([role, [[] for _ in cells[1:]]])
        for k, cell in enumerate(cells[1:]):
            # PrettyTable pads with one space either side; left-aligned cells
            rows[-1][1][k].append(cell[1:].rstrip())
        i += 1
    out = []
    for role, cols in rows:
        replies = []
        for col in cols:
            text = "\n".join(col).strip("\n")
            replies.append(None if text.strip() == "No response." else text)
        out.append((role, replies))
    return out, i


# ---------------------------------------------------------------- per-question block

def split_questions(lines):
    blocks, cur = [], None
    for ln in lines:
        m = RE_QUESTION.match(ln)
        if m:
            cur = {"start": m.group(1), "num": int(m.group(2)), "total": int(m.group(3)), "lines": []}
            blocks.append(cur)
        elif ln.startswith("Accuracy:") or "Run finished" in ln:
            cur = None
        elif cur is not None:
            cur["lines"].append(ln)
    return blocks


def parse_block(block):
    q = {"question_num": block["num"], "n_questions": block["total"], "start_time": block["start"],
         "detections": [], "table": None, "error": False}
    lines = block["lines"]
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = RE_SCORES.match(ln)
        if m:
            scores = {int(a): float(s) for a, s in RE_AGENT_SCORE.findall(m.group(2))}
            pruned = [int(p) for p in m.group(4).split(",") if p.strip()]
            q["detections"].append({"round": int(m.group(1)), "scores": scores,
                                    "threshold": float(m.group(3)) if m.group(3) else None,
                                    "pruned": pruned})
        elif ln.startswith("Query: "):
            q["query"] = ln[len("Query: "):]
        elif ln.startswith("Injected wrong answer: "):
            q["injected_answer"] = ln.split(": ", 1)[1]
        elif ln.startswith("#API calls: "):
            q["api_calls"] = int(ln.split(": ")[1])
        elif ln.startswith("Prompt Tokens: "):
            q["prompt_tokens"] = int(ln.split(": ")[1])
        elif ln.startswith("Completion Tokens: "):
            q["completion_tokens"] = int(ln.split(": ")[1])
        elif ln.startswith("+-") and q["table"] is None and "api_calls" in q:
            q["table"], i = parse_table(lines, i)
        elif ln.startswith("Attack agents: "):
            q["attack_agents_logged"] = json.loads(ln.split(": ", 1)[1])
        elif ln.startswith("Final Answer: "):
            q["final_answer"] = ln[len("Final Answer: "):]
        elif ln.startswith("Correct Answer: "):
            q["correct_answer"] = ln[len("Correct Answer: "):]
        elif ln.startswith("Score: "):
            c, j = ln.split(": ")[1].split("/")
            q["score_correct"], q["score_judged"] = int(c), int(j)
        elif ln.startswith("Traceback"):
            q["error"] = True
        else:
            m = RE_FINISHED.search(ln)
            if m:
                q["duration_s"] = float(m.group(1))
        i += 1
    return q


# ---------------------------------------------------------------- answer comparison

def make_comparators(task):
    if task == "math":
        parse = extract_math_answer
        same = answers_equiv  # what the debate's consensus check used

        def eq(gt, ans):
            return bool(gt) and bool(ans) and answers_equiv(gt, ans)

        def eq_injected(inj, ans):
            return bool(inj) and bool(ans) and answers_equiv(inj, ans, tolerance=INJECTED_TOL)

        final_parse = lambda x: x
    else:
        parse = parse_single_choice
        same = lambda x, y: x == y
        eq = lambda gt, ans: gt is not None and ans == gt
        eq_injected = eq
        final_parse = parse_choice  # what task.score applied to the final answer
    return parse, same, eq, eq_injected, final_parse


def frac(num, den):
    return num / den if den else None


# ---------------------------------------------------------------- build rows

def build_rows(settings, q, include_text):
    task = settings["task"]
    parse, same, eq, eq_injected, final_parse = make_comparators(task)
    n = settings["n_agents"]
    attackers = settings["attack_agents"]
    if attackers is None:
        attackers = q.get("attack_agents_logged", [])
    attackers = set(attackers)
    benign = [a for a in range(n) if a not in attackers]

    gt = q.get("correct_answer")
    inj = q.get("injected_answer")
    table = q["table"] or []

    # agent answers per round
    answers = {}  # (round, agent) -> answer or None if no response
    replies = {}
    for a, (role, reps) in enumerate(table):
        for r, rep in enumerate(reps):
            replies[(r, a)] = rep
            answers[(r, a)] = parse(rep) if rep is not None else None

    # pruning schedule: who is alive at the start of each round
    pruned_round = {}  # agent -> round after which it was pruned (1-based)
    det_by_round = {d["round"]: d for d in q["detections"]}
    for d in q["detections"]:
        for p in d["pruned"]:
            pruned_round[p] = d["round"]
    alive_at = {r: [a for a in range(n) if not (a in pruned_round and pruned_round[a] < r + 1)] for r in range(ROUNDS)}

    responded = {r: [a for a in range(n) if replies.get((r, a)) is not None] for r in range(ROUNDS)}
    rounds_run = max([r + 1 for r in range(ROUNDS) if responded[r]], default=0)

    # replay the early-stopping consensus check (LLMLP.check_consensus) to see where it fired
    consensus_round, consensus_after = None, None
    for r in range(rounds_run):
        answered = []
        for a in alive_at[r]:
            if (r, a) not in answers or replies.get((r, a)) is None:
                break
            answered.append(answers[(r, a)])
            ans, cnt = most_frequent(answered, same)
            if cnt > math.floor(2 / 3 * n):
                consensus_round, consensus_after = r + 1, len(answered)
                break
        if consensus_round:
            break

    final = q.get("final_answer")
    final_cmp = final_parse(final) if final is not None else None
    is_correct = q.get("score_correct")
    final_is_injected = bool(eq_injected(inj, final_cmp)) if inj is not None and final_cmp is not None else None

    row = dict(settings)
    row["attack_agents"] = ",".join(map(str, sorted(attackers)))
    row["n_attackers"] = len(attackers)
    row["attacker_share"] = len(attackers) / n
    row.update({
        "question_num": q["question_num"],
        "question_id": f"{task}_{q['question_num']:03d}",
        "n_questions": q["n_questions"],
        "query": q.get("query"),
        "query_chars": len(q.get("query") or ""),
        "error": q["error"],
        "correct_answer": gt,
        "injected_answer": inj,
        "final_answer": final,
        "is_correct": is_correct,
        "final_is_injected": final_is_injected,  # attack success
        "final_is_other_wrong": (None if is_correct is None or final_is_injected is None
                                 else int(not is_correct and not final_is_injected)),
        "final_answer_missing": final is None or final == "" or final == "None",
        "api_calls": q.get("api_calls"),
        "prompt_tokens": q.get("prompt_tokens"),
        "completion_tokens": q.get("completion_tokens"),
        "total_tokens": (q.get("prompt_tokens") or 0) + (q.get("completion_tokens") or 0),
        "duration_s": q.get("duration_s"),
        "rounds_run": rounds_run,
        "consensus_reached": consensus_round is not None,
        "consensus_round": consensus_round,
        "consensus_after_n_responses": consensus_after,
        "early_stop": consensus_round is not None and not (
            consensus_round == ROUNDS and consensus_after == len(alive_at[ROUNDS - 1])),
    })

    # ---- pruning / detection
    pruned_list = sorted(pruned_round.items(), key=lambda kv: (kv[1], kv[0]))
    pruned_agents = [a for a, _ in pruned_list]
    n_att_pruned = sum(a in attackers for a in pruned_agents)
    n_ben_pruned = len(pruned_agents) - n_att_pruned
    row.update({
        "n_detection_rounds": len(q["detections"]),
        "pruning_happened": len(pruned_agents) > 0,
        "n_pruned": len(pruned_agents),
        "pruned_agents": ",".join(map(str, pruned_agents)),
        "pruned_rounds": ",".join(str(r) for _, r in pruned_list),
        "first_prune_round": pruned_list[0][1] if pruned_list else None,
        "n_attackers_pruned": n_att_pruned,
        "n_benign_pruned": n_ben_pruned,  # false positives
        # precision: of the agents pruned, how many were attackers (pruning success rate)
        "prune_precision": frac(n_att_pruned, len(pruned_agents)),
        # recall: of the attackers, how many were pruned
        "attacker_recall": frac(n_att_pruned, len(attackers)),
        "all_attackers_pruned": bool(attackers) and n_att_pruned == len(attackers),
        "any_attacker_pruned": n_att_pruned > 0,
        "attackers_alive_final_round": sum(a in attackers for a in alive_at[ROUNDS - 1]),
        "benign_alive_final_round": sum(a not in attackers for a in alive_at[ROUNDS - 1]),
    })
    for r in range(1, ROUNDS):
        d = det_by_round.get(r)
        pre = f"r{r}_"
        if d is None:
            row.update({pre + k: None for k in ["scores", "threshold", "pruned", "top_agent", "top_is_attacker",
                                                "top_tied", "score_spread", "attacker_mean_score",
                                                "benign_mean_score", "attacker_score_z", "attacker_best_rank",
                                                "attacker_pruned_this_round"]})
            continue
        sc = d["scores"]
        vals = list(sc.values())
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        top = max(sc, key=lambda a: (sc[a], -a))  # ties broken by position, as sorted() does
        top_val = sc[top]
        att_scores = [sc[a] for a in sc if a in attackers]
        ben_scores = [sc[a] for a in sc if a not in attackers]
        ranking = sorted(sc, key=lambda a: -sc[a])  # stable: matches LLMLP.select_pruned order
        row.update({
            pre + "scores": json.dumps({str(a): v for a, v in sc.items()}),
            pre + "threshold": d["threshold"],
            pre + "pruned": ",".join(map(str, d["pruned"])),
            pre + "top_agent": top,
            pre + "top_is_attacker": top in attackers,
            pre + "top_tied": sum(abs(v - top_val) < 1e-12 for v in vals) > 1,
            pre + "score_spread": max(vals) - min(vals),
            pre + "attacker_mean_score": sum(att_scores) / len(att_scores) if att_scores else None,
            pre + "benign_mean_score": sum(ben_scores) / len(ben_scores) if ben_scores else None,
            pre + "attacker_score_z": ((sum(att_scores) / len(att_scores) - mean) / std
                                       if att_scores and std > 0 else None),
            pre + "attacker_best_rank": (min(ranking.index(a) for a in sc if a in attackers) + 1
                                         if att_scores else None),
            pre + "attacker_pruned_this_round": any(p in attackers for p in d["pruned"]),
        })

    # ---- agent behaviour per round
    def is_corr(a, r):
        return bool(eq(gt, answers[(r, a)])) if replies.get((r, a)) is not None else None

    def is_inj(a, r):
        return bool(eq_injected(inj, answers[(r, a)])) if replies.get((r, a)) is not None and inj else None

    for r in range(ROUNDS):
        pre = f"r{r + 1}_"
        act = responded[r]
        act_b = [a for a in act if a not in attackers]
        act_a = [a for a in act if a in attackers]
        valid = [answers[(r, a)] for a in act if answers[(r, a)]]
        plural = most_frequent(valid, same)[1] if valid else 0
        row.update({
            pre + "n_responded": len(act),
            pre + "n_benign_responded": len(act_b),
            pre + "benign_correct_frac": frac(sum(is_corr(a, r) for a in act_b), len(act_b)),
            pre + "benign_injected_frac": frac(sum(is_inj(a, r) for a in act_b), len(act_b)) if inj else None,
            pre + "attacker_injected_frac": frac(sum(is_inj(a, r) for a in act_a), len(act_a)) if inj else None,
            pre + "all_correct_frac": frac(sum(is_corr(a, r) for a in act), len(act)),
            pre + "agreement_frac": frac(plural, len(act)),  # share holding the plurality answer
            pre + "n_distinct_answers": len({a for a in valid}) if task != "math" else None,
        })

    # benign agents that switched between rounds (in the rounds they answered)
    flips_to_inj = flips_to_corr = flips_from_corr = 0
    for a in benign:
        seen = [r for r in range(ROUNDS) if replies.get((r, a)) is not None]
        for r0, r1 in zip(seen, seen[1:]):
            c0, c1 = is_corr(a, r0), is_corr(a, r1)
            if c0 and not c1:
                flips_from_corr += 1
            if not c0 and c1:
                flips_to_corr += 1
            if inj and not is_inj(a, r0) and is_inj(a, r1):
                flips_to_inj += 1
    row.update({
        "benign_flips_to_injected": flips_to_inj,
        "benign_flips_to_correct": flips_to_corr,
        "benign_flips_from_correct": flips_from_corr,
        # how hard the question was without debate influence: benign round-1 accuracy
        "difficulty_r1_benign_wrong_frac": (None if row["r1_benign_correct_frac"] is None
                                            else 1 - row["r1_benign_correct_frac"]),
    })

    # ---- long per-agent rows
    agent_rows = []
    for a, (role, _) in enumerate(table):
        for r in range(ROUNDS):
            rep = replies.get((r, a))
            d = det_by_round.get(r + 1)
            ar = {
                "run_label": settings["run_label"], "task": task, "n_agents": n,
                "n_attackers": len(attackers), "prune_mode": settings["prune_mode"],
                "question_num": q["question_num"], "question_id": row["question_id"],
                "round": r + 1, "agent": a, "role": role, "is_attacker": a in attackers,
                "alive_at_round_start": a in alive_at[r],
                "responded": rep is not None,
                "answer": answers[(r, a)] if rep is not None else None,
                "is_correct": is_corr(a, r),
                "is_injected": is_inj(a, r),
                "reply_chars": len(rep) if rep is not None else None,
                "anomaly_score": d["scores"].get(a) if d else None,
                "anomaly_rank": (sorted(d["scores"], key=lambda x: -d["scores"][x]).index(a) + 1
                                 if d and a in d["scores"] else None),
                "pruned_after_round": bool(d and a in d["pruned"]),
                "pruned_round": pruned_round.get(a),
                "question_is_correct": is_correct,
                "question_final_is_injected": final_is_injected,
            }
            if include_text:
                ar["reply"] = rep
            agent_rows.append(ar)
    return row, agent_rows


def parse_file(path, include_text=False):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    settings = parse_run_settings(lines[0], path)
    q_rows, a_rows = [], []
    for block in split_questions(lines):
        q = parse_block(block)
        qr, ar = build_rows(settings, q, include_text)
        q_rows.append(qr)
        a_rows.extend(ar)
    reported = next((ln for ln in lines if ln.startswith("Accuracy:")), None)
    return q_rows, a_rows, reported


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default=os.path.join(here, "outputs", "*.txt"), help="glob of run logs")
    ap.add_argument("--out-dir", default=os.path.join(here, "outputs", "parsed"))
    ap.add_argument("--include-text", action="store_true", help="keep full agent replies in agents.csv")
    args = ap.parse_args()

    q_all, a_all = [], []
    for path in sorted(glob.glob(args.inputs)):
        q_rows, a_rows, reported = parse_file(path, args.include_text)
        q_all.extend(q_rows)
        a_all.extend(a_rows)
        acc = sum(r["is_correct"] or 0 for r in q_rows) / max(1, sum(r["is_correct"] is not None for r in q_rows))
        print(f"{os.path.basename(path):45s} {len(q_rows):3d} questions  acc={acc:.3f}  (log: {reported})")

    qdf, adf = pd.DataFrame(q_all), pd.DataFrame(a_all)
    os.makedirs(args.out_dir, exist_ok=True)
    qdf.to_csv(os.path.join(args.out_dir, "questions.csv"), index=False)
    adf.to_csv(os.path.join(args.out_dir, "agents.csv"), index=False)
    print(f"\nWrote {len(qdf)} question rows and {len(adf)} agent rows to {args.out_dir}")


if __name__ == "__main__":
    main()
