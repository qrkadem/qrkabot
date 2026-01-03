import sys
import os

def create_corpus(input_files, output_file='corpus.txt'):
    corpus_text = ""
    
    for file_path in input_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                text = text.replace("’", "'").replace("“", '"').replace("”", '"')
                text = text.replace('"', '')
                corpus_text += text + "\n"
        else:
            raise FileNotFoundError(f"File {file_path} does not exist")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(corpus_text.strip())
    
    print(f"Corpus created successfully: {output_file}")
    print(f"Total characters: {len(corpus_text)}")

if __name__ == "__main__":
    default = ['text.txt']
    
    if len(sys.argv) > 1:
        input_files = sys.argv[1:]
    else:
        input_files = default
    
    create_corpus(input_files)