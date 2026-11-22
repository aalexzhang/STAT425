import math
import random
from LLM_Neuron import LLMNeuron, LLMEdge, listwise_ranker_2
from utils import parse_single_choice, most_frequent, is_equiv, extract_math_answer, generate_answer, ts, answers_equiv
from sacrebleu import sentence_bleu
from prompt_lib import GEN_THRESHOLD
import torch
from torch_geometric.data import Data
from torch.optim import Adam
from model_static import DOMINANTDetector
import model_temporal_gib
from attacks import Attack

ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1}
DETECTORS = ("temporal", "static")
PRUNE_MODES = ("top1", "threshold")

class LLMLP:
    
    def __init__(self, default_model_name, agents=4, agent_roles=[],
                 rounds=2, activation="listwise", qtype="single_choice", mtype="gpt-4o-mini", device="cuda:0",
                 detector="temporal", attack=None, prune="top1", prune_k=1.0, min_agents=2):
        # detector: "temporal" = GUARDIAN, "static" = GUARDIAN.s
        # prune: "top1" removes the highest-scoring agent each round (as in the paper);
        #        "threshold" removes every agent scoring above mean + prune_k * std of that round's scores
        # min_agents: pruning never leaves fewer agents than this
        assert detector in DETECTORS, f"unknown detector: {detector}"
        assert prune in PRUNE_MODES, f"unknown prune mode: {prune}"
        self.detector = detector
        self.prune = prune
        self.prune_k = prune_k
        self.min_agents = min_agents
        self.attack = attack if attack is not None else Attack("none")
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        self.device = device

        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "math_exp":
            self.cmp_res = answers_equiv
            self.ans_parser = extract_math_answer 
        elif qtype == "open-ended":
            self.cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score >= GEN_THRESHOLD * 100
            self.ans_parser = lambda x: x
        else:
            raise NotImplementedError("Error qtype")

        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(self.make_node(idx, agent_roles))
        
        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(self.make_node(idx, agent_roles))
                for a1 in agents_last_round:
                    self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.agents:]

        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")
    
    def make_node(self, pos, agent_roles):
        role = self.attack.role(pos, agent_roles[pos])
        return LLMNeuron(role, self.mtype, self.ans_parser, self.attack.node_qtype(pos, self.qtype))

    def activate_node(self, node_idx, question, wrong_answer):
        pos, rid = node_idx % self.agents, node_idx // self.agents
        self.nodes[node_idx].activate(
            self.attack.prompt(pos, question, wrong_answer),
            self.attack.injected_responses(pos, rid, self.qtype, wrong_answer),
        )

    def detect(self, data_t):
        # returns one anomaly score per node of the latest snapshot.
        # The first round always uses the static detector, as in the original code.
        if self.detector == "temporal" and len(data_t) > 1:
            model = model_temporal_gib.train_model(data_t)
            _, _, scores = model.detect(data_t)
        else:
            static_data = data_t[-1]
            model = DOMINANTDetector(hid_dim=128, num_gnn_layers=2).to(self.device)
            optimizer = Adam(model.parameters(), lr=0.001)
            model.fit(static_data, num_epochs=20, optimizer=optimizer)
            _, _, scores = model.detect(static_data)
        return [float(x) for x in scores]

    def select_pruned(self, alive, scores):
        # returns (agent positions to prune, threshold used or None)
        budget = len(alive) - self.min_agents
        if budget <= 0:
            return [], None
        order = sorted(range(len(alive)), key=lambda i: scores[i], reverse=True)
        if self.prune == "top1":
            return [alive[order[0]]], None
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((x - mean) ** 2 for x in scores) / len(scores))
        threshold = mean + self.prune_k * std
        return [alive[i] for i in order if scores[i] > threshold][:budget], threshold

    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()

    def check_consensus(self, idxs, idx_mask):
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            return True, consensus_answer
        return False, None

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def forward(self, question, wrong_answer=None):
        def get_completions():
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        def generate_fully_connected_directed_graph(n):
            edge_index = [[], []]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_index[0].append(i)
                        edge_index[1].append(j)

            edge_index = torch.tensor(edge_index, dtype=torch.long)

            adj = torch.zeros((n, n), dtype=torch.float32)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        adj[i, j] = 1

            return edge_index, adj

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        assert self.rounds > 2

        alive = list(range(self.agents))  # agent positions still in the debate
        history = []                      # per round: {agent position: answer}
        self.pruned = []                  # (round, agent position, score) for every pruned agent
        for rid in range(self.rounds):
            answered = []
            for pos in alive:
                node_idx = self.agents * rid + pos
                self.activate_node(node_idx, question, wrong_answer)
                resp_cnt += 1
                total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                total_completion_tokens += self.nodes[node_idx].completion_tokens
                answered.append(node_idx)

                # stop early once more than 2/3 of all agents agree
                reached, reply = self.check_consensus(answered, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens
            history.append({idx % self.agents: self.nodes[idx].get_answer() for idx in answered})

            # pruning after the last round would not change the final answer
            if rid == self.rounds - 1:
                break

            # one snapshot per round so far, restricted to the agents still alive
            edge_index, adj = generate_fully_connected_directed_graph(len(alive))
            data_t = [Data(text=[answers[pos] for pos in alive], edge_index=edge_index, adj=adj) for answers in history]
            scores = self.detect(data_t)
            pruned, threshold = self.select_pruned(alive, scores)

            score_str = ", ".join(f"agent {pos}: {score:.4f}" for pos, score in zip(alive, scores))
            threshold_str = f", threshold {threshold:.4f}" if threshold is not None else ""
            print(f"[{ts()}] Round {rid + 1} anomaly scores: {score_str}{threshold_str} -> pruned {pruned}")
            for pos in pruned:
                self.pruned.append((rid, pos, scores[alive.index(pos)]))
            alive = [pos for pos in alive if pos not in pruned]

        completions = get_completions()
        return most_frequent([self.nodes[idx].get_answer() for idx in answered], self.cmp_res)[0], resp_cnt, completions, total_prompt_tokens, total_completion_tokens
