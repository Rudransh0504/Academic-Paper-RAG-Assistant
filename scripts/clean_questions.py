import json

REMOVE = {"q027", "q029"}   # q027: ambiguous; q029: gold chunk is copyright boilerplate
qs = json.load(open("data/eval/questions_v1_42.json"))
kept = [q for q in qs if q["qid"] not in REMOVE]
json.dump(kept, open("data/eval/questions.json", "w"), indent=2)
print("Remaining:", len(kept))