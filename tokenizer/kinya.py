import os
import sentencepiece as spm
import spacy
import glob

def load_spacy_blank_model():
    """
    Load a blank spaCy model for custom language processing.
    """
    nlp = spacy.blank('xx')  # 'xx' is a generic blank language model
    print("spaCy blank model loaded for custom language.")
    # Increase max_length to handle longer texts if needed
    nlp.max_length = 2000000
    return nlp

def clean_and_normalize_text(text, nlp):
    """
    Use spaCy to clean and normalize text for tokenization purposes.
    """
    doc = nlp(text)
    
    # Extract tokens and preserve punctuation and numbers as separate tokens
    tokens = [token.text for token in doc]
    
    # Join tokens into a single string with spaces for SentencePiece processing
    normalized_text = " ".join(tokens)
    
    return normalized_text, tokens

def chunk_text(text, chunk_size=1000000):
    """
    Split text into chunks of a specified size.
    """
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def load_and_prepare_data(nlp, data_folder='data'):
    """
    Read, clean, and preprocess text data using spaCy for normalization.
    """
    all_text = ""
    
    # Reading and cleaning all text files in the data folder
    for filename in glob.glob(os.path.join(data_folder, '*.txt')):
        with open(filename, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
            # Chunk and process the text to avoid exceeding spaCy limits
            text_chunks = chunk_text(raw_text)
            for chunk in text_chunks:
                cleaned_text, tokens = clean_and_normalize_text(chunk, nlp)
                all_text += cleaned_text + "\n"
    
    # Save the cleaned text into a temporary file
    with open('temp_corpus.txt', 'w', encoding='utf-8') as temp_file:
        temp_file.write(all_text)

    print(f"Data from {data_folder} loaded, cleaned, and saved into 'temp_corpus.txt'.")

def train_tokenizer(vocab_size=28000, model_type='bpe'):
    """
    Train the SentencePiece tokenizer model on the preprocessed data and save the model and vocabulary.
    """
    spm.SentencePieceTrainer.train(input='temp_corpus.txt', 
                                   model_prefix='kinya', 
                                   vocab_size=vocab_size, 
                                   model_type=model_type, 
                                   character_coverage=1.0,  # Ensures full character coverage
                                   max_sentence_length=2048)

    print("Tokenizer model 'tokenizer.model' and vocabulary 'tokenizer.vocab' saved.")

def tokenize_with_bpe(text, sp):
    """
    Tokenize text using the trained SentencePiece BPE model for subword tokenization.
    """
    return sp.encode_as_pieces(text)

# Entry point
if __name__ == "__main__":
    data_folder = 'data'
    
    # Step 1: Load spaCy blank model
    nlp = load_spacy_blank_model()
    
    # Step 2: Load, clean, and prepare the data
    load_and_prepare_data(nlp, data_folder)
    
    # Step 3: Train and save the tokenizer for deep learning use
    train_tokenizer()
