#!/home/qrkadem/work/code/qrkabot/.venv/bin/python3
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
import random
import sys
from collections import deque
from difflib import SequenceMatcher

detokenizer = TreebankWordDetokenizer()

RECENT_RESPONSES = deque(maxlen=7)

RULE_STARTING_VERBS = [
    "don't", "always", "never", "take", "get", "use", "walk", "build", "keep",
    "avoid", "remember", "forget", "respect", "ban", "post", "read", "write",
    "share", "stay", "pay", "check", "bring", "leave", "listen", "watch",
    "play", "stop", "start", "report", "ping", "join", "wait", "ask",
    "tell", "think", "sleep", "ship", "fix", "test", "be", "make",
    "learn", "try", "help", "mind", "hold", "move", "clean", "sort",
    "wear", "drink", "eat", "call", "reply", "meet", "mute", "vote",
    "carry", "save", "backup", "update", "rollback", "commit", "push", "pull",
    "review", "deploy", "monitor", "log", "follow", "trust", "verify", "report",
    "practice", "sketch", "organize", "label", "archive", "email", "document",
    "refactor", "benchmark", "profile", "compress", "encrypt", "scan", "lock",
    "unlock", "enable", "disable", "charge", "reboot", "sync"
]



corpus_path = "./corpora/corpus.txt"

if not os.path.exists(corpus_path):
    print("Corpus file not found.")
    sys.exit(1)

with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

def clean_and_tokenize_text(text):
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    # tokenize the text into words and punctuation
    tokens = word_tokenize(text)
    return tokens

# clean and tokenize text
cleaned_text = []
for line in text.splitlines():
    tokens = clean_and_tokenize_text(line.strip())
    tokens.append("\n")  # mark end of message
    cleaned_text.extend(tokens)

print("Number of tokens =", len(cleaned_text))

def make_markov_model(cleaned_text, n_gram=2):
    markov_model = {}
    for i in range(len(cleaned_text) - n_gram):
        curr_state, next_state = "", ""
        for j in range(n_gram):
            curr_index = i + j
            if curr_index < len(cleaned_text):
                curr_state += cleaned_text[curr_index] + " "
            next_index = i + j + n_gram
            if next_index < len(cleaned_text):
                next_state += cleaned_text[next_index] + " "

        curr_state = tuple(cleaned_text[i:i+n_gram])
        next_state = tuple(cleaned_text[i+n_gram:i+2*n_gram])

        if curr_state not in markov_model:
            markov_model[curr_state] = {}
            markov_model[curr_state][next_state] = 1
        else:
            if next_state in markov_model[curr_state]:
                markov_model[curr_state][next_state] += 1
            else:
                markov_model[curr_state][next_state] = 1

    # calculating transition probabilities
    for curr_state, transition in markov_model.items():
        total = sum(transition.values())
        for state, count in transition.items():
            markov_model[curr_state][state] = count / total

    return markov_model

pp_markov_model = make_markov_model(cleaned_text, n_gram=2)
print("number of states = ", len(pp_markov_model.keys()))

def generate(pp_markov_model, limit=100, start=None):
    if start is None or start not in pp_markov_model:
        start = random.choice(list(pp_markov_model.keys()))
    n = 0
    curr_state = start
    next_state = None
    story_tokens = list(curr_state)
    while n < limit:
        try:
            if not pp_markov_model[curr_state]:
                # if no transitions, pick a new random state
                curr_state = random.choice(list(pp_markov_model.keys()))
                continue
        except KeyError:
            # if state not in model, pick a new one
            curr_state = random.choice(list(pp_markov_model.keys()))
            continue
        next_state = random.choices(list(pp_markov_model[curr_state].keys()),
                                    list(pp_markov_model[curr_state].values()))

        curr_state = next_state[0]

        # stop if end-of-message token predicted
        if "\n" in curr_state:
            break

        story_tokens.extend(curr_state)
        n += 1
    story_tokens = [tok for tok in story_tokens if tok != "\n"]
    story = detokenizer.detokenize(story_tokens)
    story = re.sub(r"\s+([.,!?;:])", r"\1", story)
    story = re.sub(r'\s+([\'"])', r'\1', story)
    story = re.sub(r'([\'"])\s+', r'\1', story)
    return story

def _normalize_response(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _is_similar_response(candidate, recent_responses, threshold=0.8):
    candidate_norm = _normalize_response(candidate)
    if not candidate_norm:
        return False
    for prev in recent_responses:
        prev_norm = _normalize_response(prev)
        if not prev_norm:
            continue
        ratio = SequenceMatcher(None, candidate_norm, prev_norm).ratio()
        if ratio >= threshold:
            return True
    return False

def _memres(response):
    if response:
        RECENT_RESPONSES.append(response)

def _generate_unique_response(generator_fn, attempts=8):
    candidate = ""
    for _ in range(attempts):
        candidate = generator_fn()
        if not _is_similar_response(candidate, RECENT_RESPONSES):
            return candidate
    return candidate

def _force_starting_verb(text, verb):
    tokens = word_tokenize(text) if text else []
    if not tokens:
        return verb
    tokens[0] = verb
    forced = detokenizer.detokenize(tokens)
    forced = re.sub(r"\s+([.,!?;:])", r"\1", forced)
    forced = re.sub(r"\s+([\'\"])", r"\1", forced)
    forced = re.sub(r"([\'\"])\s+", r"\1", forced)
    return forced

def generate_rules(count=3, limit_range=(8, 16)):
    rules = []
    for _ in range(count):
        verb = random.choice(RULE_STARTING_VERBS)
        limit = random.randint(*limit_range)
        generated = generate(pp_markov_model, limit=limit)
        rule = _force_starting_verb(generated, verb)
        rules.append(rule)
    headers = ["Forum rules:", "today's rules:", "rules:", "Rules:", "THE RULES:", 
               "Current rules:", "Official rules:", "New rules:", "rules of the day:"]
    lines = [random.choice(headers)]
    for i, rule in enumerate(rules, start=1):
        lines.append(f"{i}. {rule}")
    return "\n".join(lines)

def generate_response(prompt=None, limit=random.randint(8, 18), user=None):
    
    if prompt is None:
        response = _generate_unique_response(
            lambda: generate(pp_markov_model, limit=limit)
        )
        _memres(response)
        return response

    prompt_lower = prompt.lower().strip()

    m = re.match(r"(.+?)\s+more like$", prompt, re.IGNORECASE)
    if m:
        base = m.group(1).strip()
        generated = _generate_unique_response(
            lambda: generate(pp_markov_model, limit=limit, start=base)
        )
        response = f"{base}\nmore like\n{generated}"
        _memres(response)
        return response
    elif prompt_lower == "who are you":
        response = "I'm a Markov-chain bot representing qrkadem. https://raw.githubusercontent.com/qrkadem/qrkabot/master/README.md"
        _memres(response)
        return response
    elif prompt_lower == "help":
        response = "I'm actually stupid, so I can't help you."
        _memres(response)
        return response
    elif prompt_lower == "i hate you":
        response = "that's okay, I hate myself too."
        _memres(response)
        return response
    elif prompt_lower == "bannings":
        banned = ["Merth", "KermM", "TIny_Hacker", "MateoConLechuga", "Sumde", "Adriweb", "DeltaX", user, "qrkadem", "tev", "TIFreak8x", "qrkabot", "sumdebot", "nikkybot"]
        response = "RANDOM MONTHLY BANNINGS\n" + random.choice(banned) + ": You lose"
        _memres(response)
        return response
    elif prompt_lower in ("~botabuse", "botabuse", "bot abuse"):
        response = "STOP ABUSING ME\nSTOP ABUSING ME"
        _memres(response)
        return response
    elif prompt_lower == "rust":
        response = "but it's memory safe! guys, it's memory safe! you can trust it! its memory safe!"
        _memres(response)
        return response
    elif "mimic" in prompt_lower:
        response = "no"
        _memres(response)
        return response
    elif re.search(r"\brules\b", prompt_lower):
        response = _generate_unique_response(generate_rules)
        _memres(response)
        return response
    # convert prompt to tokens
    tokens = clean_and_tokenize_text(prompt)
    
    # take first n_gram tokens as starting state
    n_gram = 2 
    if len(tokens) >= n_gram:
        start = tuple(tokens[:n_gram])
    else:
        # pad or pick random if too short
        start = None

    response = _generate_unique_response(
        lambda: generate(pp_markov_model, limit=limit, start=start)
    )
    _memres(response)
    return response