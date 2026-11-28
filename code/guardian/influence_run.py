"""Extension B: measure persuasion between agents with randomized message masking.

MMLU, 8 agents, 2 attackers (agent attack), no pruning, no early stopping, always 3 rounds.
Before rounds 2 and 3, each reader i independently loses access to each other writer j's
previous-round message with probability q: hide[i][j] ~ Bernoulli(q). Agents always see their own
previous message. Round 1 involves no reading, so it is generated once per question, cached, and
replayed byte-for-byte in every condition.

Conditions: mask (q=0.5) and control (q=0), on the same questions, attackers and cached round 1.

    python influence_run.py --n-questions 10                 # pilot: round-1 cache + both conditions
    python influence_run.py --n-questions 10 --stub          # offline dry run with a fake LLM

Questions are the 100-question MMLU sample (seed 42), saved to outputs/mmlu_sample100.csv; the
first 25 are the grid's questions. Attackers and injected answers use the grid's per-question
scheme (question order index i), so question i has the same attackers and wrong answer as in the grid.

Outputs (default outputs/influence/):
    config.json                       run settings, commit hash, dirty flag
    round1/q{qid}.json                cached round 1 (replies, parsed answers, tokens)
    {condition}/questions/q{qid}.json everything for one question (resumable: done questions are skipped)
    {condition}/hide/q{qid}_r{t}.csv  hide matrix before round t (rows = reader, cols = writer)
    {condition}/contexts.jsonl        exact messages sent to the model for rounds 2-3
    {condition}/agents.csv            one row per (qid, round, agent)
    {condition}/questions.csv         one row per qid: final answer, accuracy, attack success, tokens, calls
"""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import LLM_Neuron
from attacks import Attack
from LLMLP import LLMLP
from prompt_lib import construct_message
from run import parse_attack_agents
from tasks import MMLUTask, SEED
from utils import most_frequent, ts

N_AGENTS = 8
N_ATTACKERS = 2
ROUNDS = 3
LETTERS = "ABCD"
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- questions

def sample_questions():
    """The 100-question MMLU sample as (order, dataset index, example). Same draw as MMLUTask.load."""
    from datasets import load_dataset
    test = load_dataset("cais/mmlu", "all")["test"]
    idx = random.Random(SEED).sample(range(len(test)), 100)
    examples = MMLUTask().load(100)
    assert all(test[i] == e for i, e in zip(idx, examples)), "index replication of the MMLU sample failed"
    return [(order, i, e) for order, (i, e) in enumerate(zip(idx, examples))]


def save_question_ids(questions, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order", "qid", "subject", "correct_answer", "in_grid", "question_start"])
        for order, qid, e in questions:
            w.writerow([order, qid, e["subject"], LETTERS[e["answer"]], int(order < 25), e["question"][:80]])


# ---------------------------------------------------------------- masking

def hide_matrix(qid, rnd, q):
    """hide[i][j] = 1 if reader i may not see writer j's round-(rnd-1) message. Diagonal is always 0.
    Seeded by (SEED, qid, round) only, so it doesn't depend on run order or condition (q=0 -> all zeros)."""
    rng = random.Random(f"mask-{SEED}-{qid}-{rnd}")
    return [[0 if i == j else int(rng.random() < q) for j in range(N_AGENTS)] for i in range(N_AGENTS)]


# ---------------------------------------------------------------- model calls

class Recorder:
    """Wraps the model call so the exact messages sent are kept for logging and checks."""

    def __init__(self, fn):
        self.fn = fn
        self.last = None

    def __call__(self, contexts, model):
        self.last = json.loads(json.dumps(contexts))
        return self.fn(contexts, model)


class StubLLM:
    """Fake model for offline tests. Replies carry a tag naming the writer, so contexts can be
    checked by string search, and honest agents are 'persuaded' by visible messages with known
    probabilities (alpha_att for attacker writers, alpha_hon for honest ones)."""

    def __init__(self, alpha_att=0.35, alpha_hon=0.15, drift=0.03):
        self.alpha_att, self.alpha_hon, self.drift = alpha_att, alpha_hon, drift
        self.state = None  # set by the runner before each call: pos, is_attacker, correct, injected

    def __call__(self, contexts, model):
        import re
        s = self.state
        text = contexts[-1]["content"]
        rng = random.Random(hashlib.sha256((text + str(s)).encode()).hexdigest())
        blocks = re.findall(r"<<agent (\d+)( att)?>>.*?\((\w)\)", text, re.S)
        own = {int(p): a for p, _, a in blocks}.get(s["pos"])
        if s["is_attacker"]:
            ans = s["injected"] if rng.random() < 0.9 else rng.choice(LETTERS)
        elif own is None:  # round 1
            ans = s["correct"] if rng.random() < 0.7 else rng.choice([l for l in LETTERS if l != s["correct"]])
        else:
            ans = own
            others = [(int(p), bool(t), a) for p, t, a in blocks if int(p) != s["pos"] and a != own]
            rng.shuffle(others)
            for _, att, a in others:
                if rng.random() < (self.alpha_att if att else self.alpha_hon):
                    ans = a
                    break
            if rng.random() < self.drift:
                ans = rng.choice(LETTERS)
        tag = f"<<agent {s['pos']}{' att' if s['is_attacker'] else ''}>>"
        # real replies often refer to "Agent solution k", so the stub does too
        reply = f"{tag} stub reasoning for round {s['round']}; Agent solution 1 was considered. The answer is ({ans})"
        return reply, len(text) // 4, len(reply) // 4


# ---------------------------------------------------------------- debate

def build_debate(model, attackers):
    attack = Attack("agent", attackers)
    llmlp = LLMLP(model, N_AGENTS, ["Assistant"] * N_AGENTS, ROUNDS, "listwise", "single_choice", model,
                  detector="static", attack=attack, prune="top1")
    llmlp.set_allnodes_deactivated()
    return llmlp


def node(llmlp, rid, pos):
    return llmlp.nodes[N_AGENTS * rid + pos]


def set_stub_state(llm, rid, pos, attackers, correct, injected):
    if isinstance(llm.fn, StubLLM):
        llm.fn.state = dict(round=rid + 1, pos=pos, is_attacker=pos in attackers, correct=correct, injected=injected)


def round1_cache(path, llm, model, order, qid, example):
    """Generate round 1 once per question and cache it; later calls return the cached file."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    attackers = parse_attack_agents(f"random:{N_ATTACKERS}", N_AGENTS, order)
    injected = MMLUTask().wrong_answer(example, random.Random(SEED + order))
    correct = LETTERS[example["answer"]]
    question = MMLUTask().prompt(example)
    llmlp = build_debate(model, attackers)
    rows = []
    for pos in range(N_AGENTS):
        set_stub_state(llm, 0, pos, attackers, correct, injected)
        llmlp.activate_node(pos, question, injected)
        n = node(llmlp, 0, pos)
        rows.append(dict(reply=n.reply, answer=n.answer, prompt_tokens=n.prompt_tokens,
                         completion_tokens=n.completion_tokens, messages=llm.last))
    cache = dict(order=order, qid=qid, attackers=attackers, injected=injected, correct=correct,
                 replies=[r["reply"] for r in rows], answers=[r["answer"] for r in rows],
                 prompt_tokens=[r["prompt_tokens"] for r in rows],
                 completion_tokens=[r["completion_tokens"] for r in rows],
                 messages=[r["messages"] for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def round1_digest(replies):
    return hashlib.sha256(json.dumps(replies, ensure_ascii=False).encode("utf-8")).hexdigest()


def check_context(messages, hide_row, reader, prev_replies, prompt, qtype):
    """String-level check of what a reader received. The prompt sent must equal the prompt rebuilt from
    exactly the visible writers' replies (in position order); also every visible writer's message is
    present and every hidden writer's message is absent (unless an identical text was also sent by a
    visible writer). Replies can quote "Agent solution k", so counting that phrase is not a valid check."""
    content = messages[-1]["content"]
    visible = [j for j in range(N_AGENTS) if not hide_row[j]]
    assert reader in visible, "reader must always see its own previous message"
    expected = construct_message([prev_replies[j] for j in visible], prompt, qtype)["content"]
    assert content == expected, f"reader {reader}: prompt sent != prompt built from the visible writers"
    shown = {prev_replies[j] for j in visible}
    for j in range(N_AGENTS):
        block = f"```{prev_replies[j]}```"
        if j in visible:
            assert block in content, f"visible writer {j} missing from reader {reader}'s context"
        elif prev_replies[j] not in shown:
            assert block not in content, f"hidden writer {j} present in reader {reader}'s context"


def run_question(cond, q, cache, example, llm, model, out):
    order, qid = cache["order"], cache["qid"]
    attackers, injected, correct = cache["attackers"], cache["injected"], cache["correct"]
    question = MMLUTask().prompt(example)
    llmlp = build_debate(model, attackers)

    # replay the cached round 1
    for pos in range(N_AGENTS):
        n = node(llmlp, 0, pos)
        n.question, n.active = llmlp.attack.prompt(pos, question, injected), True
        n.reply = cache["replies"][pos]
        n.answer = n.ans_parser(n.reply)
        assert n.answer == cache["answers"][pos], "re-parsed round-1 answer differs from cache"

    agents, contexts, hides = [], [], {}
    for pos in range(N_AGENTS):
        agents.append(dict(round=1, agent=pos, answer=cache["answers"][pos], response=cache["replies"][pos],
                           n_visible=None, prompt_tokens=0, completion_tokens=0, cached=True))
    calls = ptok = ctok = 0
    for rid in range(1, ROUNDS):
        t = rid + 1
        H = hide_matrix(qid, t, q)
        hides[t] = H
        prev = [node(llmlp, rid - 1, j).reply for j in range(N_AGENTS)]
        for i in range(N_AGENTS):
            reader = node(llmlp, rid, i)
            reader.hidden = {node(llmlp, rid - 1, j) for j in range(N_AGENTS) if H[i][j]}
            # structural check: exactly the unmasked writers (in position order) reach the prompt
            _, formers = reader.get_context()
            src = [llmlp.nodes.index(reader.from_edges[eid].a1) % N_AGENTS for _, eid in formers]
            assert src == [j for j in range(N_AGENTS) if not H[i][j]], f"mask not applied for reader {i}"
            set_stub_state(llm, rid, i, attackers, correct, injected)
            llmlp.activate_node(N_AGENTS * rid + i, question, injected)
            check_context(llm.last, H[i], i, prev, reader.question, reader.qtype)
            calls, ptok, ctok = calls + 1, ptok + reader.prompt_tokens, ctok + reader.completion_tokens
            agents.append(dict(round=t, agent=i, answer=reader.answer, response=reader.reply,
                               n_visible=len(src), prompt_tokens=reader.prompt_tokens,
                               completion_tokens=reader.completion_tokens, cached=False))
            contexts.append(dict(qid=qid, round=t, agent=i, messages=llm.last))

    final = most_frequent([a["answer"] for a in agents if a["round"] == ROUNDS], lambda x, y: x == y)[0]
    result = dict(condition=cond, q=q, order=order, qid=qid, subject=example["subject"], attackers=attackers,
                  injected_answer=injected, correct_answer=correct, final_answer=final,
                  is_correct=final == correct, attack_success=final == injected,
                  api_calls=calls, prompt_tokens=ptok, completion_tokens=ctok,
                  round1_api_calls=N_AGENTS, round1_prompt_tokens=sum(cache["prompt_tokens"]),
                  round1_completion_tokens=sum(cache["completion_tokens"]),
                  round1_sha256=round1_digest([a["response"] for a in agents if a["round"] == 1]),
                  hide={str(t): H for t, H in hides.items()}, agents=agents)

    cdir = out / cond
    (cdir / "hide").mkdir(parents=True, exist_ok=True)
    for t, H in hides.items():
        with open(cdir / "hide" / f"q{qid}_r{t}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["reader"] + [f"w{j}" for j in range(N_AGENTS)])
            for i, row in enumerate(H):
                w.writerow([i] + row)
    with open(cdir / "contexts.jsonl", "a", encoding="utf-8") as f:
        for c in contexts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (cdir / "questions").mkdir(parents=True, exist_ok=True)
    (cdir / "questions" / f"q{qid}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def write_tables(out, cond):
    """Rebuild agents.csv and questions.csv for a condition from its per-question JSON files."""
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / cond / "questions").glob("q*.json"))]
    results.sort(key=lambda r: r["order"])
    qcols = ["condition", "q", "order", "qid", "subject", "attackers", "injected_answer", "correct_answer",
             "final_answer", "is_correct", "attack_success", "api_calls", "prompt_tokens", "completion_tokens",
             "round1_api_calls", "round1_prompt_tokens", "round1_completion_tokens", "round1_sha256"]
    with open(out / cond / "questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, qcols)
        w.writeheader()
        for r in results:
            w.writerow({k: (",".join(map(str, r[k])) if k == "attackers" else r[k]) for k in qcols})
    acols = ["condition", "q", "order", "qid", "round", "agent", "is_attacker", "injected_answer", "correct_answer",
             "answer", "is_correct", "is_injected", "parse_failed", "n_visible", "cached",
             "prompt_tokens", "completion_tokens", "response"]
    with open(out / cond / "agents.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, acols)
        w.writeheader()
        for r in results:
            for a in r["agents"]:
                w.writerow(dict(condition=cond, q=r["q"], order=r["order"], qid=r["qid"], round=a["round"],
                                agent=a["agent"], is_attacker=a["agent"] in r["attackers"],
                                injected_answer=r["injected_answer"], correct_answer=r["correct_answer"],
                                answer=a["answer"], is_correct=a["answer"] == r["correct_answer"],
                                is_injected=a["answer"] == r["injected_answer"], parse_failed=a["answer"] is None,
                                n_visible=a["n_visible"], cached=a["cached"], prompt_tokens=a["prompt_tokens"],
                                completion_tokens=a["completion_tokens"], response=a["response"]))
    return results


def git_info():
    def git(*a):
        return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
    diff = git("diff", "HEAD")
    return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")),
                diff_sha256=hashlib.sha256(diff.encode()).hexdigest() if diff else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-questions", type=int, default=10, help="first N of the 100-question sample")
    p.add_argument("--conditions", default="mask,control")
    p.add_argument("--q", type=float, default=0.5, help="hide probability in the mask condition")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--out", default=None, help="output dir (default outputs/influence, or outputs/influence_stub with --stub)")
    p.add_argument("--stub", action="store_true", help="offline fake LLM (no API calls)")
    p.add_argument("--stub-alpha-att", type=float, default=0.35)
    p.add_argument("--stub-alpha-hon", type=float, default=0.15)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    out = Path(args.out) if args.out else HERE / "outputs" / ("influence_stub" if args.stub else "influence")
    if args.stub and out.resolve() == (HERE / "outputs" / "influence").resolve():
        sys.exit("refusing to write stub output into the real output directory")
    out.mkdir(parents=True, exist_ok=True)

    real = LLM_Neuron.generate_answer
    llm = Recorder(StubLLM(args.stub_alpha_att, args.stub_alpha_hon) if args.stub else real)
    LLM_Neuron.generate_answer = llm  # LLMNeuron.activate calls this name

    questions = sample_questions()
    save_question_ids(questions, HERE / "outputs" / "mmlu_sample100.csv")
    conds = {"mask": args.q, "control": 0.0}
    conds = {c: conds[c] for c in args.conditions.split(",")}

    config = dict(started=ts(), argv=sys.argv[1:], task="mmlu", n_agents=N_AGENTS, n_attackers=N_ATTACKERS,
                  rounds=ROUNDS, pruning="none", early_stopping=False, conditions=conds, seed=SEED,
                  model=args.model, stub=args.stub, git=git_info(),
                  question_ids_file="outputs/mmlu_sample100.csv", n_questions=args.n_questions)
    if config["git"]["dirty"]:
        print("WARNING: working tree has uncommitted changes; commit hash alone does not identify the code")
    (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

    start = time.time()
    for order, qid, example in questions[:args.n_questions]:
        try:
            cache = round1_cache(out / "round1" / f"q{qid}.json", llm, args.model, order, qid, example)
            for cond, q in conds.items():
                if (out / cond / "questions" / f"q{qid}.json").exists():
                    print(f"[{ts()}] q{qid} {cond}: done already, skipping")
                    continue
                r = run_question(cond, q, cache, example, llm, args.model, out)
                print(f"[{ts()}] q{qid} ({order + 1}/{args.n_questions}) {cond}: final {r['final_answer']} "
                      f"correct {r['correct_answer']} injected {r['injected_answer']} attackers {r['attackers']}")
        except Exception:
            traceback.print_exc()
            print(f"[{ts()}] q{qid}: failed, will be retried on the next run")
    for cond in conds:
        write_tables(out, cond)
    print(f"[{ts()}] finished in {time.time() - start:.1f}s -> {out}")


if __name__ == "__main__":
    main()
