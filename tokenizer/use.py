import sentencepiece as spm
import sys

def load_tokenizer(model_file='kinya.model'):
    """
    Load the trained SentencePiece tokenizer model.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp

def view_vocabulary(tokenizer):
    """
    Print the entire vocabulary of the tokenizer.
    """
    vocab_size = tokenizer.get_piece_size()
    vocab = [tokenizer.id_to_piece(i) for i in range(vocab_size)]
    print(f"Vocabulary Size: {vocab_size}")
    print("Vocabulary:")
    for i, piece in enumerate(vocab):
        print(f"{i}: {piece}")

def tokenize_text(tokenizer, text):
    """
    Tokenize input text using the loaded tokenizer model and display token indices.
    """
    tokens = tokenizer.encode_as_pieces(text)
    indices = tokenizer.encode(text)
    
    print("Tokenized text with indices:")
    for idx, (token, index) in enumerate(zip(tokens, indices)):
        print(f"Index {idx}: Token '{token}'")

    return tokens

def main():
    if len(sys.argv) < 2:
        print("Usage: python use.py <command> [text]")
        print("Commands:")
        print("  show_vocab    - Display the vocabulary of the tokenizer.")
        print("  tokenize      - Tokenize the provided text and show token indices.")
        sys.exit(1)

    command = sys.argv[1].lower()
    tokenizer = load_tokenizer()

    if command == "show_vocab":
        view_vocabulary(tokenizer)
    elif command == "tokenize":
        if len(sys.argv) > 2:
            text = " ".join(sys.argv[2:])
            tokenize_text(tokenizer, text)
        else:
            print("Error: No text provided for tokenization.")
            sys.exit(1)
    else:
        print(f"Unrecognized command: {command}")
        print("Usage: python use.py <command> [text]")
        sys.exit(1)

if __name__ == "__main__":
    main()
