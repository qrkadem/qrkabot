#!/home/qrkadem/work/code/qrkabot/.venv/bin/python3
import numpy as np
import pandas as pd
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
import requests
import sys

# URL of the text file
url = "http://www.gutenberg.org/files/1342/1342-0.txt"

# Path to the downloaded file
novel_path = "pride_and_prejudice.txt"

# Get text source
if not sys.stdin.isatty():
    # Read from stdin (piped input)
    text = sys.stdin.read()
else:
    # Download the file if it doesn't exist
    if not os.path.exists(novel_path):
        response = requests.get(url)
        if response.status_code == 200:
            # Save the content to a local file
            with open(novel_path, "w", encoding="utf-8") as f:
                f.write(response.text)
        else:
            print("Failed to download the file.")
            sys.exit(1)
    # Read from file
    with open(novel_path, 'r', encoding='utf-8') as f:
        text = f.read()

def clean_and_tokenize_text(text):
    # Tokenize the text into words and punctuation
    tokens = word_tokenize(text)
    return tokens

# Clean and tokenize text
cleaned_text = clean_and_tokenize_text(text)
print("Number of tokens =", len(cleaned_text))

def make_markov_model(cleaned_text, n_gram=3):
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

        curr_state = curr_state.strip()
        next_state = next_state.strip()

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

pp_markov_model = make_markov_model(cleaned_text)
print("number of states = ", len(pp_markov_model.keys()))

def generate_story(pp_markov_model, limit=100, start=None):
    if start is None or start not in pp_markov_model:
        start = random.choice(list(pp_markov_model.keys()))
    n = 0
    curr_state = start
    next_state = None
    story_tokens = curr_state.split()  # Start with the initial state tokens
    while n < limit:
        try:
            if not pp_markov_model[curr_state]:
                # If no transitions, pick a new random state
                curr_state = random.choice(list(pp_markov_model.keys()))
                continue
        except KeyError:
            # If state not in model, pick a new one
            curr_state = random.choice(list(pp_markov_model.keys()))
            continue
        next_state = random.choices(list(pp_markov_model[curr_state].keys()),
                                    list(pp_markov_model[curr_state].values()))

        curr_state = next_state[0]
        story_tokens.extend(curr_state.split())
        n += 1
    
    # Join tokens with spaces, then fix punctuation spacing
    story = ' '.join(story_tokens)
    story = re.sub(r'\s+([,.!?;:])', r'\1', story)  # Remove space before punctuation
    story = re.sub(r'([,.!?;:])\s+', r'\1 ', story)  # Ensure space after punctuation if needed, but actually for commas etc., keep as is
    # Actually, simpler: just remove spaces before punctuation
    return story


# Generate a story
generated_story = generate_story(pp_markov_model, limit=100)
print("Generated Story: \n", generated_story)