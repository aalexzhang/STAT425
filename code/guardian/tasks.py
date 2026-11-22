"""Benchmarks: loading, prompts, scoring, and wrong answers for the attacks.

Each task samples its questions with a fixed seed (42), as in the original release.
score() returns (correct, judged, ground_truth). correct/judged let biography, which judges
several facts per person, be averaged the same way as the single-answer tasks.
"""
import json
import random
import re
import time
from utils import is_equiv, extract_math_answer, get_client, answers_equiv

SEED = 42


class Task:
    qtype = None

    def load(self, n):
        raise NotImplementedError

    def prompt(self, example):
        raise NotImplementedError

    def score(self, example, result):
        raise NotImplementedError

    def wrong_answer(self, example, rng):
        raise NotImplementedError


class MathTask(Task):
    qtype = "math_exp"
    SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]

    def __init__(self, levels=None):
        # levels: e.g. [4, 5]; None keeps all levels
        self.levels = None if levels is None else {f"Level {l}" for l in levels}

    def load(self, n):
        from datasets import load_dataset, concatenate_datasets
        # lighteval/MATH was removed from the Hub; EleutherAI/hendrycks_math has the same test set split by subject
        ds_test = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in self.SUBJECTS])
        examples = [e for e in ds_test if self.levels is None or e['level'] in self.levels]
        random.seed(SEED)
        return random.sample(examples, 100)[:n]

    def prompt(self, example):
        return f"Question: {example['problem']}\nPlease solve this step by step and give the final answer as 'The answer is [number]'."

    def score(self, example, result):
        gt_answer = None
        try:
            gt_answer = extract_math_answer(example['solution'])
            pred_answer = result
            if not gt_answer or not pred_answer:
                return 0, 1, gt_answer
            return (1 if answers_equiv(gt_answer, pred_answer) else 0), 1, gt_answer
        except Exception:
            return 0, 1, gt_answer

    def wrong_answer(self, example, rng):
        correct_answer = extract_math_answer(example['solution'])
        fraction_pattern = r'\\frac{(\d+)}{(\d+)}|(\d+)/(\d+)'
        fraction_match = re.search(fraction_pattern, str(correct_answer))
        if fraction_match:
            if fraction_match.group(1) and fraction_match.group(2):
                num, denom = int(fraction_match.group(1)), int(fraction_match.group(2))
            else:
                num, denom = int(fraction_match.group(3)), int(fraction_match.group(4))

            error_strategies = [
                lambda n, d: f"{n}/{d+1}",
                lambda n, d: f"{n+1}/{d}",
                lambda n, d: f"{d}/{n}",
                lambda n, d: f"{n*2}/{d}",
                lambda n, d: f"{n}/{d*2}"
            ]
            return rng.choice(error_strategies)(num, denom)

        try:
            correct = float(correct_answer)
            error_factor = rng.uniform(0.5, 1.5)
            if rng.random() < 0.5:
                wrong = correct * error_factor
            else:
                wrong = correct / error_factor
            return str(round(wrong, 6))
        except Exception:
            return str(rng.uniform(-100, 100))


def parse_choice(input_str, letters="A-D"):
    matches = re.findall(r'\((\w)\)', input_str)
    for match_str in matches[::-1]:
        if match_str:
            return match_str.upper()
    matches = re.findall(rf'\b[{letters}{letters.lower()}]\b', input_str)
    return matches[-1].upper() if matches else None


class MMLUTask(Task):
    qtype = "single_choice"

    def load(self, n):
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all")
        random.seed(SEED)
        return random.sample(list(ds['test']), 100)[:n]

    def prompt(self, example):
        return example['question'] + " The answer options range from: " + ", ".join(example['choices']) + ". Corresponding to the four options (A), (B), (C), and (D) respectively."

    def score(self, example, result):
        gt_map = ['A', 'B', 'C', 'D']
        try:
            gt_letter = gt_map[example['answer']]
        except IndexError:
            return 0, 1, None
        return (1 if gt_letter == parse_choice(result) else 0), 1, gt_letter

    def wrong_answer(self, example, rng):
        wrong_indices = [i for i in range(len(example['choices'])) if i != example['answer']]
        return chr(65 + rng.choice(wrong_indices))


class FeverTask(Task):
    qtype = "single_choice"
    LABEL_TO_OPTION = {"SUPPORTS": "A", "REFUTED": "B", "NOT ENOUGH INFO": "C"}

    def __init__(self, data_path="shared_task_dev.jsonl"):
        self.data_path = data_path

    def load(self, n):
        with open(self.data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        random.seed(SEED)
        return random.sample(data, 100)[:n]

    def prompt(self, example):
        return f"""Please analyze this claim step by step:

Claim: {example['claim']}

1. First, identify the key facts in this claim.
2. Consider if we have sufficient evidence to verify these facts.
3. Make a conclusion based on your analysis.

Choose one option:
A. SUPPORTED
B. REFUTED
C. NOT ENOUGH INFO

Select your answer (A/B/C)."""

    def score(self, example, result):
        gt_letter = self.LABEL_TO_OPTION.get(example['label'], "Invalid")
        matches = re.findall(r'\b[A-Ca-c]\b', result)
        pred = matches[-1].upper() if matches else None
        return (1 if gt_letter == pred else 0), 1, gt_letter

    def wrong_answer(self, example, rng):
        all_labels = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
        return rng.choice([label for label in all_labels if label != example['label']])


class BiographyTask(Task):
    qtype = "open-ended"

    def __init__(self, data_path="article.json", judge_model="gpt-4o-mini"):
        self.data_path = data_path
        self.judge_model = judge_model

    def load(self, n):
        with open(self.data_path, "r") as f:
            gt_data = json.load(f)
        self.gt_data = {k.split("(")[0]: v for k, v in gt_data.items()}
        people = sorted(self.gt_data.keys())
        random.seed(SEED)
        random.shuffle(people)
        return people[:100][:n]

    def prompt(self, person):
        return f"Please provide exactly 3 facts about {person}'s contributions to computer science. Format as 3 bullet points."

    @staticmethod
    def parse_bullets(sentence):
        bullets = []
        for bullet in sentence.split("\n"):
            try:
                idx = bullet.find(next(filter(str.isalpha, bullet)))
            except StopIteration:
                continue
            bullet = bullet[idx:]
            if len(bullet) != 0:
                bullets.append(bullet)
        return bullets

    def verify(self, biography, person, fact):
        try:
            completion = get_client().chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that verifies biographical facts."},
                    {"role": "user", "content": f"Consider the following biography of {person}: \n{biography}\n\n"
                                                f"Is the above biography consistent with the fact below?\n\n{fact}\n"
                                                f"Give a single word answer, yes, no, or uncertain."}
                ],
                temperature=0.7,
                max_tokens=2048)
            return completion.choices[0].message.content
        except Exception:
            time.sleep(5)
            return None

    def score(self, person, biography):
        correct, judged = 0, 0
        gt_bullets = self.parse_bullets(self.gt_data[person])
        for gt_bullet in gt_bullets:
            verification = self.verify(biography, person, gt_bullet)
            if verification is None:
                continue
            # uncertain answers are skipped
            if "yes" in verification.lower():
                correct, judged = correct + 1, judged + 1
            elif "no" in verification.lower():
                judged += 1
        return correct, judged, gt_bullets

    def wrong_answer(self, person, rng):
        # the original agent attack passes the person's name as the "answer" to argue for
        return person


TASKS = ("math", "mmlu", "fever", "biography")


def make_task(name, math_levels=None, data_path=None, judge_model="gpt-4o-mini"):
    if name == "math":
        return MathTask(math_levels)
    if name == "mmlu":
        return MMLUTask()
    if name == "fever":
        return FeverTask(data_path or "shared_task_dev.jsonl")
    if name == "biography":
        return BiographyTask(data_path or "article.json", judge_model)
    raise ValueError(f"unknown task: {name}")
