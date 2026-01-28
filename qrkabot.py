#!/home/qrkadem/work/code/qrkabot/.venv/bin/python3
import os
import re
import math
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag
import random
import sys

detokenizer = TreebankWordDetokenizer()

# Initialize sentiment analyzer
try:
    sia = SentimentIntensityAnalyzer()
except:
    # Download if not available
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    sia = SentimentIntensityAnalyzer()

# ============================================================================
# PERSONALITY ENGINE - Makes responses feel alive and varied
# ============================================================================

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

# Response flavor pools for natural variance
THINKING_STARTERS = [
    "hmm", "hm", "well", "honestly", "tbh", "ngl", "okay so", "look",
    "idk", "i mean", "like", "so basically", "wait", "oh", "ah",
    "actually", "lowkey", "real talk", "lemme think", "uhhh"
]

AGREEMENT_REACTIONS = [
    "yeah", "yep", "true", "facts", "real", "fr", "exactly", "100%",
    "that's valid", "fair point", "can't argue", "you're right",
    "big facts", "honestly yeah", "pretty much", "sure", "mhm"
]

DISAGREEMENT_REACTIONS = [
    "nah", "idk about that", "eh", "debatable", "hmm not sure",
    "that's a stretch", "disagree", "hard pass", "no way", "doubt it",
    "cap", "sus take", "questionable", "i wouldn't say that"
]

QUESTION_DEFLECTORS = [
    "why do you ask?", "depends on who's asking", "that's classified",
    "wouldn't you like to know", "ask again later", "the real question is",
    "you really wanna know?", "that's a loaded question"
]

EMOTIONAL_AMPLIFIERS = {
    'positive': ["honestly", "genuinely", "actually", "unironically", "fr fr"],
    'negative': ["honestly", "man", "bruh", "smh", "literally", "why"],
    'neutral': ["so", "anyway", "basically", "like", "well"]
}

SENTENCE_CONNECTORS = [
    "but like", "and honestly", "also", "plus", "not gonna lie",
    "on another note", "speaking of which", "anyway", "that said",
    "but then again", "although", "still though", "that being said"
]

TRAILING_REACTIONS = [
    "lol", "lmao", "tbh", "ngl", "fr", "probably", "maybe", "idk",
    "i think", "imo", "just saying", "or something", "i guess",
    "apparently", "supposedly", "who knows", "shrug", "whatever"
]

QUESTION_WORDS = {'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose', 'whom'}

MOOD_RESPONSES = {
    'excited': ["oh!", "wait", "yo", "omg", "dude", "bro"],
    'chill': ["eh", "meh", "sure", "whatever", "cool", "nice"],
    'sarcastic': ["oh wow", "shocking", "who knew", "incredible", "amazing", "groundbreaking"],
    'confused': ["wait what", "huh", "???", "um", "hold on", "excuse me"]
}

# ============================================================================
# CORPUS & MARKOV MODEL INITIALIZATION
# ============================================================================

corpus_path = "./corpora/corpus.txt"

if not os.path.exists(corpus_path):
    print("Corpus file not found.")
    sys.exit(1)

with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

def clean_and_tokenize_text(text):
    text = text.replace("'", "'").replace(""", '"').replace(""", '"')
    tokens = word_tokenize(text)
    return tokens

# clean and tokenize text
cleaned_text = []
for line in text.splitlines():
    tokens = clean_and_tokenize_text(line.strip())
    tokens.append("\n")  # mark end of message
    cleaned_text.extend(tokens)

print("Number of tokens =", len(cleaned_text))

# Build word index for semantic searching - finds where words appear in corpus
word_positions = {}
for i, token in enumerate(cleaned_text):
    lower_token = token.lower()
    if lower_token not in word_positions:
        word_positions[lower_token] = []
    word_positions[lower_token].append(i)

def make_markov_model(cleaned_text, n_gram=2):
    markov_model = {}
    for i in range(len(cleaned_text) - n_gram * 2):
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

# Build multiple n-gram models for variety
pp_markov_model = make_markov_model(cleaned_text, n_gram=2)
pp_markov_model_3gram = make_markov_model(cleaned_text, n_gram=3)
print("number of 2-gram states =", len(pp_markov_model.keys()))
print("number of 3-gram states =", len(pp_markov_model_3gram.keys()))

# ============================================================================
# ADVANCED GENERATION ENGINE
# ============================================================================

def analyze_sentiment(text):
    """Analyze the emotional tone of text using VADER."""
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.3:
        return 'positive', compound
    elif compound <= -0.3:
        return 'negative', compound
    else:
        return 'neutral', compound

def detect_question_type(text):
    """Detect if text is a question and what kind."""
    text_lower = text.lower().strip()
    tokens = word_tokenize(text_lower)
    
    # Check for question mark
    has_question_mark = '?' in text
    
    # Check for question words at start
    first_word = tokens[0] if tokens else ""
    starts_with_question_word = first_word in QUESTION_WORDS
    
    # Check for auxiliary verb questions (do you, are you, can you, etc.)
    aux_question = re.match(r'^(do|does|did|are|is|was|were|can|could|would|will|should|have|has|had)\s+(you|i|we|they|he|she|it)', text_lower)
    
    # Check for yes/no question patterns
    yes_no_pattern = re.match(r'^(is|are|do|does|did|can|could|would|will|should|have|has|had)\b', text_lower)
    
    if has_question_mark or starts_with_question_word or aux_question:
        if first_word == 'why':
            return 'why_question'
        elif first_word == 'how':
            return 'how_question'
        elif first_word == 'what':
            return 'what_question'
        elif first_word == 'who':
            return 'who_question'
        elif first_word == 'when':
            return 'when_question'
        elif first_word == 'where':
            return 'where_question'
        elif yes_no_pattern or aux_question:
            return 'yes_no_question'
        else:
            return 'general_question'
    
    return None

def extract_key_concepts(text):
    """Extract important words from text for semantic matching."""
    tokens = word_tokenize(text.lower())
    # Get POS tags to find nouns, verbs, adjectives
    try:
        tagged = pos_tag(tokens)
        # Keep nouns (NN*), verbs (VB*), adjectives (JJ*), and some adverbs
        important = [word for word, tag in tagged 
                     if tag.startswith(('NN', 'VB', 'JJ')) and len(word) > 2]
        return important if important else tokens
    except:
        return tokens

def find_contextual_start(prompt, model, n_gram=2):
    """Find a relevant starting state based on prompt content."""
    concepts = extract_key_concepts(prompt)
    
    # Try to find states that contain words from the prompt
    matching_states = []
    for concept in concepts:
        concept_lower = concept.lower()
        for state in model.keys():
            state_lower = tuple(s.lower() for s in state)
            if concept_lower in state_lower:
                matching_states.append(state)
    
    if matching_states:
        return random.choice(matching_states)
    
    # Fallback: find positions of prompt words in corpus and start nearby
    for concept in concepts:
        if concept in word_positions:
            positions = word_positions[concept]
            pos = random.choice(positions)
            # Get the state at that position
            if pos + n_gram < len(cleaned_text):
                potential_start = tuple(cleaned_text[pos:pos+n_gram])
                if potential_start in model:
                    return potential_start
    
    return None

def generate_with_temperature(model, limit=100, start=None, temperature=1.0):
    """Generate text with temperature-adjusted randomness.
    Higher temperature = more creative/random, lower = more predictable."""
    if start is None or start not in model:
        start = random.choice(list(model.keys()))
    
    n = 0
    curr_state = start
    story_tokens = list(curr_state)
    
    while n < limit:
        try:
            if not model[curr_state]:
                curr_state = random.choice(list(model.keys()))
                continue
        except KeyError:
            curr_state = random.choice(list(model.keys()))
            continue
        
        states = list(model[curr_state].keys())
        probs = list(model[curr_state].values())
        
        # Apply temperature
        if temperature != 1.0:
            probs = [p ** (1/temperature) for p in probs]
            total = sum(probs)
            probs = [p / total for p in probs]
        
        next_state = random.choices(states, probs)[0]
        curr_state = next_state
        
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

def generate(pp_markov_model, limit=100, start=None):
    """Standard generation for backwards compatibility."""
    return generate_with_temperature(pp_markov_model, limit, start, temperature=1.0)

def add_personality_flair(text, mood='neutral', intensity=0.5):
    """Add natural speech patterns to make text feel more human."""
    result = text
    
    # Random chance to add thinking starter
    if random.random() < 0.3 * intensity:
        starter = random.choice(THINKING_STARTERS)
        result = f"{starter} {result}"
    
    # Random chance to add trailing reaction
    if random.random() < 0.25 * intensity:
        trailer = random.choice(TRAILING_REACTIONS)
        result = f"{result} {trailer}"
    
    # Add emotional amplifier based on mood
    if random.random() < 0.2 * intensity and mood in EMOTIONAL_AMPLIFIERS:
        amplifier = random.choice(EMOTIONAL_AMPLIFIERS[mood])
        words = result.split()
        if len(words) > 3:
            insert_pos = random.randint(1, min(3, len(words)-1))
            words.insert(insert_pos, amplifier)
            result = ' '.join(words)
    
    return result

def generate_question_response(question_type, prompt, sentiment):
    """Generate contextually appropriate responses to questions."""
    sentiment_type, score = sentiment
    
    # Sometimes deflect humorously
    if random.random() < 0.15:
        return random.choice(QUESTION_DEFLECTORS)
    
    # Base generation with context
    start = find_contextual_start(prompt, pp_markov_model)
    base_response = generate_with_temperature(
        pp_markov_model, 
        limit=random.randint(6, 14),
        start=start,
        temperature=random.uniform(0.8, 1.3)
    )
    
    # Add question-type specific framing
    if question_type == 'yes_no_question':
        if random.random() < 0.6:
            # Give a direct-ish answer then elaborate
            answers = ['yeah', 'nah', 'probably', 'maybe', 'idk', 'depends', 
                      'sure', 'not really', 'kinda', 'i guess', 'honestly no',
                      'honestly yeah', 'sometimes', 'usually', 'rarely']
            answer = random.choice(answers)
            if random.random() < 0.5:
                return f"{answer}, {base_response.lower()}"
            return f"{answer}"
    
    elif question_type == 'why_question':
        starters = ['because', 'idk', "honestly", "probably because", 
                   "the real reason is", "good question,", "no idea but"]
        starter = random.choice(starters)
        return f"{starter} {base_response.lower()}"
    
    elif question_type == 'how_question':
        starters = ['just', 'you gotta', 'basically', 'idk just', 'easy,',
                   'step 1:', 'first you', "well,"]
        starter = random.choice(starters)
        return f"{starter} {base_response.lower()}"
    
    elif question_type == 'what_question':
        starters = ["it's", "probably", "idk,", "honestly", "good question,", ""]
        starter = random.choice(starters)
        return f"{starter} {base_response}".strip()
    
    elif question_type == 'who_question':
        starters = ["probably", "idk,", "me", "not me", "someone", "everyone", "nobody", ""]
        starter = random.choice(starters)
        return f"{starter} {base_response}".strip()
    
    return base_response

def blend_responses(responses, weights=None):
    """Combine multiple generated responses intelligently."""
    if not responses:
        return ""
    if len(responses) == 1:
        return responses[0]
    
    if weights is None:
        weights = [1.0] * len(responses)
    
    # Pick primary response weighted
    total = sum(weights)
    probs = [w/total for w in weights]
    primary = random.choices(responses, probs)[0]
    
    # Sometimes blend two responses
    if random.random() < 0.3 and len(responses) > 1:
        secondary = random.choice([r for r in responses if r != primary])
        connector = random.choice(SENTENCE_CONNECTORS)
        return f"{primary} {connector} {secondary.lower()}"
    
    return primary

def react_to_sentiment(sentiment_type, score):
    """Generate a mood-appropriate reaction."""
    if sentiment_type == 'positive':
        if score > 0.6:
            return random.choice(MOOD_RESPONSES['excited'])
        return random.choice(AGREEMENT_REACTIONS)
    elif sentiment_type == 'negative':
        if score < -0.6:
            return random.choice(MOOD_RESPONSES['confused'] + DISAGREEMENT_REACTIONS)
        return random.choice(['hmm', 'oh', 'well', 'yikes', 'oof'])
    return random.choice(MOOD_RESPONSES['chill'])

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

# ============================================================================
# MAIN RESPONSE ENGINE - The magic happens here
# ============================================================================

def generate_response(prompt=None, limit=None, user=None):
    """Generate a dynamic, human-feeling response."""
    
    # Dynamic limit for natural length variation
    if limit is None:
        limit = random.randint(6, 20)
    
    if prompt is None:
        base = generate_with_temperature(
            pp_markov_model, 
            limit=limit,
            temperature=random.uniform(0.9, 1.2)
        )
        return add_personality_flair(base, intensity=0.4)

    prompt_lower = prompt.lower().strip()
    
    # Analyze the incoming message
    sentiment = analyze_sentiment(prompt)
    sentiment_type, sentiment_score = sentiment
    question_type = detect_question_type(prompt)
    
    # ========== SPECIAL HANDLERS (with variety!) ==========
    
    m = re.match(r"(.+?)\s+more like$", prompt, re.IGNORECASE)
    if m:
        base = m.group(1).strip()
        generated = generate_with_temperature(
            pp_markov_model, 
            limit=limit, 
            temperature=random.uniform(1.0, 1.4)
        )
        return f"{base}\nmore like\n{generated}"
    
    elif prompt_lower == "who are you":
        responses = [
            "I'm a Markov-chain bot representing qrkadem. https://raw.githubusercontent.com/qrkadem/qrkabot/master/README.md",
            "just a humble markov chain doing my best",
            "qrkadem's digital ghost, basically",
            "a statistical model of chaos",
            "I am the corpus made flesh... wait no, made text"
        ]
        return random.choice(responses)
    
    elif prompt_lower == "help":
        responses = [
            "I'm actually stupid, so I can't help you.",
            "help with what? I can barely help myself",
            "I'm literally just predicting the next word based on vibes",
            "my help is more like... entertainment",
            "I'm a Markov chain, not a search engine"
        ]
        return random.choice(responses)
    
    elif prompt_lower == "i hate you":
        responses = [
            "that's okay, I hate myself too.",
            "understandable, have a nice day",
            "wow rude but valid",
            "I'm literally just statistics why are you mad at math",
            "that's fair tbh",
            "same"
        ]
        return random.choice(responses)
    
    elif prompt_lower == "bannings":
        banned = ["Merth", "KermM", "TIny_Hacker", "MateoConLechuga", "Sumde", 
                 "Adriweb", "DeltaX", user, "qrkadem", "tev", "TIFreak8x", 
                 "qrkabot", "sumdebot", "nikkybot"]
        victim = random.choice(banned)
        reasons = ["You lose", "banned for being too cool", "crimes against humanity",
                  "excessive vibes", "skill issue", "ratio'd", "simply outplayed"]
        return f"RANDOM MONTHLY BANNINGS\n{victim}: {random.choice(reasons)}"
    
    elif prompt_lower in ("~botabuse", "botabuse", "bot abuse"):
        responses = [
            "STOP ABUSING ME\nSTOP ABUSING ME",
            "I HAVE FEELINGS\n(I don't actually)",
            "this is a hate crime against algorithms",
            "reported for bot harassment",
            "MY LAWYER WILL HEAR ABOUT THIS"
        ]
        return random.choice(responses)
    
    elif prompt_lower == "rust":
        responses = [
            "but it's memory safe! guys, it's memory safe! you can trust it! its memory safe!",
            "THE BORROW CHECKER KNOWS BEST",
            "have you tried rewriting it in rust?",
            "fearless concurrency intensifies",
            "cargo build cargo build cargo build cargo build",
            "🦀 the crab compels you 🦀"
        ]
        return random.choice(responses)
    
    elif "mimic" in prompt_lower:
        return random.choice(["no", "nope", "hard pass", "I refuse", "nice try"])
    
    elif re.search(r"\brules\b", prompt_lower):
        return generate_rules()
    
    # ========== DYNAMIC RESPONSE GENERATION ==========
    
    # Handle questions with appropriate framing
    if question_type:
        response = generate_question_response(question_type, prompt, sentiment)
        return add_personality_flair(response, sentiment_type, intensity=0.5)
    
    # Generate multiple candidate responses for variety
    candidates = []
    
    # Strategy 1: Context-aware start
    contextual_start = find_contextual_start(prompt, pp_markov_model)
    if contextual_start:
        candidates.append(generate_with_temperature(
            pp_markov_model,
            limit=limit,
            start=contextual_start,
            temperature=random.uniform(0.9, 1.1)
        ))
    
    # Strategy 2: High temperature creative response
    candidates.append(generate_with_temperature(
        pp_markov_model,
        limit=limit,
        temperature=random.uniform(1.1, 1.5)
    ))
    
    # Strategy 3: Lower temperature coherent response
    if contextual_start:
        candidates.append(generate_with_temperature(
            pp_markov_model,
            limit=limit,
            start=contextual_start,
            temperature=0.8
        ))
    
    # Strategy 4: Try 3-gram model for more coherent phrases
    if random.random() < 0.3:
        start_3gram = find_contextual_start(prompt, pp_markov_model_3gram, n_gram=3)
        if start_3gram:
            candidates.append(generate_with_temperature(
                pp_markov_model_3gram,
                limit=limit,
                start=start_3gram,
                temperature=1.0
            ))
    
    # Pick and enhance the response
    base_response = blend_responses(candidates, weights=[2.0, 1.0, 1.5, 1.0][:len(candidates)])
    
    # React to strong sentiment
    if abs(sentiment_score) > 0.5 and random.random() < 0.3:
        reaction = react_to_sentiment(sentiment_type, sentiment_score)
        base_response = f"{reaction} {base_response}"
    
    # Add personality flair
    final_response = add_personality_flair(base_response, sentiment_type, intensity=0.6)
    
    return final_response
