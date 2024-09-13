# KINYARWANDA_TOKENIZER
KINYARWANDA TOKENIZER OF 28,000 VOCAB SIZE USING SPACY AND SENTENCEPIECE LIBRARIES

# FILES
-kinya.py     # include model specifics and logic
-use.py       #  include functionality to use model 
-case.py      #include logic to consider cases of lowercase and uppercases

# Requirements
- SpaCy            # package
- Sentencepiece    # package

# TRAINING COMMANDS
- python kinya.py                    # trains the model on data files
- python case.py                     # turns training data into both cases
# USAGE COMMANDS FOR USE.PY
- python use.py tokenize "Input Text"   # tokenizes input
- python use.py show_vocab              # lists vocabulary
