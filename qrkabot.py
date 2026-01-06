#!/home/qrkadem/work/code/qrkabot/.venv/bin/python3
import numpy as np
import pandas as pd
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
import random
import requests
import sys

detokenizer = TreebankWordDetokenizer()



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

def generate_response(prompt=None, limit=random.randint(8, 18), user=None):
    
    if prompt is None:
        return generate(pp_markov_model, limit=limit)

    prompt_lower = prompt.lower().strip()

    m = re.match(r"(.+?)\s+more like$", prompt, re.IGNORECASE)
    if m:
        base = m.group(1).strip()
        generated = generate(pp_markov_model, limit=limit, start=base)
        return f"{base}\nmore like\n{generated}"
    elif prompt_lower == "who are you":
        return "I'm a Markov-chain bot representing qrkadem. https://raw.githubusercontent.com/qrkadem/qrkabot/master/README.md"
    elif prompt_lower == "help":
        return "I'm actually stupid, so I can't help you."
    elif prompt_lower == "i hate you":
        return "that's okay, I hate myself too."
    elif prompt_lower == "bannings":
        banned = ["Merth", "KermM", "TIny_Hacker", "MateoConLechuga", "Sumde", "Adriweb", "DeltaX", user, "qrkadem", "tev", "TIFreak8x", "qrkabot", "sumdebot", "nikkybot"]
        return "RANDOM MONTHLY BANNINGS\n" + random.choice(banned) + ": You lose"
    elif prompt_lower in ("~botabuse", "botabuse", "bot abuse"):
        return "STOP ABUSING ME\nSTOP ABUSING ME"
    elif prompt_lower == "rust":
        return "rust sucks"
    # convert prompt to tokens
    tokens = clean_and_tokenize_text(prompt)
    
    # take first n_gram tokens as starting state
    n_gram = 2  # must match your Markov model
    if len(tokens) >= n_gram:
        start = tuple(tokens[:n_gram])
    else:
        # pad or pick random if too short
        start = None

    return generate(pp_markov_model, limit=limit, start=start)