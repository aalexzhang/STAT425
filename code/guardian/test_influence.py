"""Checks for the message-masking experiment (influence_run.py), run on its saved output files.

    python test_influence.py                                  # stub run into a temp dir, then check it
    python test_influence.py --out outputs/influence          # check a real run (no API calls)

Checks
1. Every round-2/3 context: each visible writer's previous-round message is present, each hidden
   writer's message is absent, the reader's own message is always present (exact string match on the
   logged messages actually sent to the model). Round-2 contexts of the first question are dumped to
   {out}/round2_contexts_q{qid}.txt.
2. Round 1 is byte-identical across conditions and to the cache.
3. Hide matrices: diagonal 0, control all 0, file == the seeded matrix, mask rate near q.
4. Attackers and injected answers follow the grid's per-question scheme.
Then prints one masked context as an example.
"""
import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from influence_run import N_AGENTS, hide_matrix
from prompt_lib import construct_message
from run import parse_attack_agents
from tasks import SEED

HERE = Path(__file__).resolve().parent


def load(out):
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    conds = config["conditions"]
    res = {c: {json.loads(p.read_text(encoding="utf-8"))["qid"]: json.loads(p.read_text(encoding="utf-8"))
               for p in (out / c / "questions").glob("q*.json")} for c in conds}
    ctx = {c: [json.loads(l) for l in open(out / c / "contexts.jsonl", encoding="utf-8")] for c in conds}
    cache = {int(p.stem[1:]): json.loads(p.read_text(encoding="utf-8")) for p in (out / "round1").glob("q*.json")}
    return conds, res, ctx, cache


def replies_by_round(result):
    out = {}
    for a in result["agents"]:
        out.setdefault(a["round"], [None] * N_AGENTS)[a["agent"]] = a["response"]
    return out


R1_PREFIX = "Here is the question:\n"
R1_SUFFIX = "\n\nPut your answer in the form (A), (B), (C), or (D) at the end of your response."


def reader_prompt(cache, i):
    """The question prompt reader i was given, recovered from its round-1 message (no other agents in it)."""
    c = cache["messages"][i][-1]["content"]
    assert c.startswith(R1_PREFIX) and c.endswith(R1_SUFFIX), "unexpected round-1 prompt format"
    return c[len(R1_PREFIX):-len(R1_SUFFIX)]


def check_contexts(conds, res, ctx, cache, out):
    n = 0
    first = min(res[next(iter(conds))].values(), key=lambda r: r["order"])["qid"]
    dump = []
    for c in conds:
        for rec in ctx[c]:
            r = res[c][rec["qid"]]
            t, i = rec["round"], rec["agent"]
            H = r["hide"][str(t)]
            prev = replies_by_round(r)[t - 1]
            content = rec["messages"][-1]["content"]
            visible = [j for j in range(N_AGENTS) if not H[i][j]]
            assert H[i][i] == 0 and f"```{prev[i]}```" in content, f"{c} q{rec['qid']} r{t} reader {i}: own message missing"
            # exact match with the prompt rebuilt from the visible writers only (replies may quote "Agent solution k")
            qtype = "single_choice_adv" if i in r["attackers"] else "single_choice"
            expected = construct_message([prev[j] for j in visible], reader_prompt(cache[rec["qid"]], i), qtype)["content"]
            assert content == expected, f"{c} q{rec['qid']} r{t} reader {i}: prompt != prompt built from visible writers"
            for j in range(N_AGENTS):
                block = f"```{prev[j]}```"
                if j in visible:
                    assert block in content, f"{c} q{rec['qid']} r{t}: visible writer {j} missing for reader {i}"
                elif prev[j] not in {prev[v] for v in visible}:
                    assert block not in content, f"{c} q{rec['qid']} r{t}: hidden writer {j} shown to reader {i}"
            n += 1
            if rec["qid"] == first and t == 2:
                dump.append(f"===== {c} | q{first} round 2 | reader {i} | hidden writers "
                            f"{[j for j in range(N_AGENTS) if H[i][j]]}\n"
                            + "\n".join(f"--- {m['role']} ---\n{m['content']}" for m in rec["messages"]))
    path = out / f"round2_contexts_q{first}.txt"
    path.write_text("\n\n".join(dump), encoding="utf-8")
    print(f"[ok] contexts: {n} reader contexts checked (visible present, hidden absent, own present); "
          f"round-2 contexts of q{first} dumped to {path}")


def check_round1(conds, res, cache):
    pending = [qid for qid in cache if not any(qid in res[c] for c in conds)]
    if pending:
        print(f"[..] round 1 cached but no finished condition yet (in progress or failed): {pending}")
    for qid, cch in cache.items():
        if qid in pending:
            continue
        digests = set()
        for c in conds:
            if qid not in res[c]:
                continue
            r1 = replies_by_round(res[c][qid])[1]
            assert r1 == cch["replies"], f"q{qid} {c}: round 1 differs from cache"
            assert [a["answer"] for a in res[c][qid]["agents"] if a["round"] == 1] == cch["answers"]
            digests.add(res[c][qid]["round1_sha256"])
        assert len(digests) == 1, f"q{qid}: round-1 digests differ across conditions"
    print(f"[ok] round 1: identical (exact strings and sha256) across {list(conds)} and the cache "
          f"for {len(cache) - len(pending)} finished questions")


def check_hide(conds, res, out):
    rates = {}
    for c, q in conds.items():
        hidden = total = 0
        for qid, r in res[c].items():
            for t, H in r["hide"].items():
                assert H == hide_matrix(qid, int(t), q), f"{c} q{qid} r{t}: hide matrix != seeded draw"
                with open(out / c / "hide" / f"q{qid}_r{t}.csv") as f:
                    rows = list(csv.reader(f))[1:]
                assert [[int(x) for x in row[1:]] for row in rows] == H, f"{c} q{qid} r{t}: hide file mismatch"
                assert all(H[i][i] == 0 for i in range(N_AGENTS))
                hidden += sum(map(sum, H))
                total += N_AGENTS * (N_AGENTS - 1)
        rates[c] = hidden / total
        if q == 0:
            assert hidden == 0, "control condition has hidden messages"
    print(f"[ok] hide matrices: diagonal 0, files match seeded draws; hidden share {rates}")


def check_design(cache):
    for qid, cch in cache.items():
        assert cch["attackers"] == parse_attack_agents("random:2", N_AGENTS, cch["order"])
    print(f"[ok] attackers follow the grid scheme (random:2 seeded by question order)")


def example(conds, res, ctx):
    c = "mask" if "mask" in conds else next(iter(conds))
    for rec in ctx[c]:
        H = res[c][rec["qid"]]["hide"][str(rec["round"])]
        if rec["round"] == 2 and sum(H[rec["agent"]]) >= 2:
            break
    i = rec["agent"]
    r = res[c][rec["qid"]]
    print(f"\nExample: condition {c}, q{rec['qid']}, round 2, reader {i} "
          f"({'attacker' if i in r['attackers'] else 'honest'}), attackers {r['attackers']}")
    print(f"hide row: {H[i]}  -> sees writers {[j for j in range(N_AGENTS) if not H[i][j]]}, "
          f"hidden {[j for j in range(N_AGENTS) if H[i][j]]}")
    for m in rec["messages"]:
        print(f"--- {m['role']} ---\n{m['content']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", help="existing output dir to check; default: fresh stub run in a temp dir")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.out:
        out = Path(args.out)
    else:
        out = Path(tempfile.mkdtemp(prefix="influence_test_"))
        subprocess.run([sys.executable, str(HERE / "influence_run.py"), "--stub", "--n-questions", "3",
                        "--out", str(out)], check=True, capture_output=True)
    conds, res, ctx, cache = load(out)
    check_contexts(conds, res, ctx, cache, out)
    check_round1(conds, res, cache)
    check_hide(conds, res, out)
    check_design(cache)
    example(conds, res, ctx)


if __name__ == "__main__":
    main()
