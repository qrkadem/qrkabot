import os
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(script_dir, "./corpora/irc.log")
out_path = os.path.join(script_dir, "./corpora/corpus.txt")

results = []

url_pattern = re.compile(r"https?://\S+|www\.\S+")

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"\[D\]\s+<qrkadem>\s+(.*)", line)
        if match:
            message = match.group(1).strip()
            message = url_pattern.sub("", message).strip()
            if message:
                results.append(message)

with open(out_path, "w", encoding="utf-8") as out:
    for msg in results:
        out.write(msg + "\n")