import json
from collections import Counter

qs = json.load(open("data/eval/questions.json"))
print("Questions:", len(qs))
counts = Counter(q["gold_paper_id"] for q in qs)
print("Papers covered:", len(counts))
print("Most questions from one paper:", counts.most_common(5))