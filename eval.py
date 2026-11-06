import json, urllib.parse, urllib.request

URL = "http://localhost:8000/ask"
tests = json.load(open("questions.json"))
REFUSE = ["don't know", "do not know", "not in", "cannot", "no information", "not mention", "unknown"]
hits = correct = total_kw = 0
refused_ok = total_ref = 0

for t in tests:
    qs = urllib.parse.urlencode({"q": t["q"]})
    req = urllib.request.Request(f"{URL}?{qs}", method="POST")
    r = json.load(urllib.request.urlopen(req, timeout=300))
    ans = r["answer"].lower()
    if t["keyword"]:
        kw = t["keyword"].lower()
        total_kw += 1
        hit = any(kw in s.lower() for s in r["sources"])
        ok = kw in ans
        hits += hit
        correct += ok
        print(f"{'OK ' if hit else 'MISS'} retrieval | {'OK ' if ok else 'BAD '} answer | {t['q']}")
    else:
        total_ref += 1
        ok = any(p in ans for p in REFUSE)
        refused_ok += ok
        print(f"{'OK ' if ok else 'BAD '} refusal | {t['q']}")

print(f"\nRetrieval hit rate: {hits}/{total_kw}")
print(f"Answer accuracy:    {correct}/{total_kw}")
print(f"Correct refusals:   {refused_ok}/{total_ref}")
