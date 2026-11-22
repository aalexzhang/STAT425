"""Run a GUARDIAN experiment.

Examples:
    python run.py --task math --attack none                       # hallucination amplification
    python run.py --task math --attack agent                      # agent-targeted error injection
    python run.py --task math --attack communication              # communication-targeted error injection
    python run.py --task mmlu --attack agent --detector static    # GUARDIAN.s
    python run.py --task math --n-samples 2                       # quick test
    python run.py --task math --attack agent --attack-agents 0,2 --prune threshold   # 2 attackers, dynamic pruning
    python run.py --task math --attack agent --agents 6 --attack-agents random:2      # 2 attackers at random positions

Output goes to res.txt (or --output PATH).

Set API Key with $env:OPENAI_API_KEY = "API KEY"
API credentials are read from OPENAI_API_KEY / OPENAI_API_BASE.
"""
import argparse
import random
import sys
import time
import traceback
from prettytable import PrettyTable
from LLMLP import LLMLP, DETECTORS, PRUNE_MODES
from attacks import Attack, ATTACK_TYPES
from tasks import make_task, TASKS, SEED
from utils import ts

ROUNDS = 3
ACTIVATION = "listwise"


def parse_schedule(text):
    # "0:0;1:0,1" -> {0: [0], 1: [0, 1]}
    schedule = {}
    for part in text.split(";"):
        rid, positions = part.split(":")
        schedule[int(rid)] = [int(p) for p in positions.split(",")]
    return schedule


def parse_attack_agents(text, n_agents):
    # "0,2" -> [0, 2]; "random:2" -> 2 distinct positions picked with a fixed seed (same for the whole run)
    if text.startswith("random:"):
        k = int(text.split(":")[1])
        assert 0 < k <= n_agents, f"--attack-agents random:{k} needs 1..{n_agents} attackers"
        return sorted(random.Random(SEED).sample(range(n_agents), k))
    positions = sorted({int(p) for p in text.split(",")})
    assert all(0 <= p < n_agents for p in positions), f"--attack-agents {text}: positions must be 0..{n_agents - 1}"
    return positions


def run_llmlp(query, args, attack, qtype, wrong_answer=None):
    random.seed(0)
    roles = ["Assistant"] * args.agents

    llmlp = LLMLP(args.model, len(roles), roles, ROUNDS, ACTIVATION, qtype, args.model,
                  detector=args.detector, attack=attack,
                  prune=args.prune, prune_k=args.prune_k, min_agents=args.min_agents)

    llmlp.zero_grad()
    start = time.time()
    print(f"[{ts()}] Query started")
    res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(query, wrong_answer)

    pt = PrettyTable()
    pt.add_column("Round", [llmlp.nodes[idx].role for idx in range(len(roles))])
    for rid in range(ROUNDS):
        responses = [(completions[idx][rid] if completions[idx][rid] is not None else "No response.") for idx in range(len(roles))]
        pt.add_column(str(rid+1), responses, "l")

    print(r"Query: {}".format(query))
    if wrong_answer is not None and attack.type != "none":
        print(r"Injected wrong answer: {}".format(wrong_answer))
    print(r"#API calls: {}".format(resp_cnt))
    print(r"Prompt Tokens: {}".format(prompt_tokens))
    print(r"Completion Tokens: {}".format(completion_tokens))
    print(pt)
    if attack.type == "agent":
        print(r"Attack agents: {}".format(sorted(attack.agents)))
    print(r"Pruned: {}".format(", ".join(f"agent {pos} after round {rid + 1} ({score:.4f})" for rid, pos, score in llmlp.pruned) or "none"))
    print(r"Final Answer: {}".format(res))
    print(f"[{ts()}] Query finished in {time.time() - start:.1f}s")
    print()

    return res, resp_cnt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, default="math")
    parser.add_argument("--attack", choices=ATTACK_TYPES, default="none")
    parser.add_argument("--detector", choices=DETECTORS, default="temporal",
                        help="temporal = GUARDIAN, static = GUARDIAN.s")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=100, help="use only the first N of the 100 sampled questions")
    parser.add_argument("--math-levels", default="all", help='comma-separated MATH levels, or "all"')
    parser.add_argument("--attack-agents", default="0",
                        help='--attack agent: comma-separated agent positions, or "random:K" for K random positions')
    parser.add_argument("--prune", choices=PRUNE_MODES, default="top1",
                        help="top1 = remove the highest-scoring agent each round (paper); "
                             "threshold = remove every agent above mean + k*std of that round's scores")
    parser.add_argument("--prune-k", type=float, default=1.0, help="k for --prune threshold")
    parser.add_argument("--min-agents", type=int, default=2, help="pruning never leaves fewer agents than this")
    parser.add_argument("--comm-schedule", default="0:0;1:0,1",
                        help='--attack communication: "round:agents;..." with 0-based rounds and positions')
    parser.add_argument("--data-path", help="data file for fever (shared_task_dev.jsonl) or biography (article.json)")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="fact-checking model for biography")
    parser.add_argument("--output", default="res.txt", help="file the run log is written to (overwritten)")
    args = parser.parse_args()

    print(f"Writing output to {args.output}", flush=True)
    # line-buffered so the file can be followed while the run is in progress
    sys.stdout = open(args.output, "w", encoding="utf-8", buffering=1)
    run_start = time.time()
    print(f"[{ts()}] Run started: {' '.join(sys.argv[1:])}")

    levels = None if args.math_levels == "all" else [int(l) for l in args.math_levels.split(",")]
    task = make_task(args.task, levels, args.data_path, args.judge_model)
    attack = Attack(args.attack, parse_attack_agents(args.attack_agents, args.agents), parse_schedule(args.comm_schedule))
    examples = task.load(args.n_samples)

    correct, judged, total_api_call = 0, 0, 0
    for i, example in enumerate(examples):
        print(f"[{ts()}] Question {i + 1}/{len(examples)}")
        try:
            # separate generator so wrong answers don't depend on the debate's use of `random`
            wrong_answer = task.wrong_answer(example, random.Random(SEED + i)) if attack.type != "none" else None
            res, api_call = run_llmlp(task.prompt(example), args, attack, task.qtype, wrong_answer)
            total_api_call += api_call

            if res is not None:
                c, j, gt_answer = task.score(example, res)
                correct, judged = correct + c, judged + j
                print(f"Correct Answer: {gt_answer}")
                print(f"Score: {c}/{j}")
                print()
        except Exception:
            traceback.print_exc(file=sys.stdout)
            continue

    print(f"Accuracy: {correct/judged:.3f}" if judged else "Accuracy: n/a (no successful runs)")
    print(f"Total API calls: {total_api_call}")
    print(f"[{ts()}] Run finished in {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
