import os
import pandas as pd
import re
import time
from deep_translator import GoogleTranslator

def translate_to_english(text: str) -> str:
    """
    Translate text using GoogleTranslator (free, unofficial).
    Handles sentence splitting and translation.
    """
    try:
        if pd.isna(text) or str(text).strip() == "":
            return text
        
        sentences = re.split(r'(?<=[.!?]) +', str(text))
        translated_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                try:
                    translation = GoogleTranslator(source='auto', target='en').translate(sentence)
                except Exception as inner_e:
                    print(f"Translation error (sentence): {inner_e}")
                    translation = sentence
                print(f"Original: {sentence} | Translated: {translation}")
                translated_sentences.append(translation)
                time.sleep(0.5)  # optional: biar aman dari limit
        return ' '.join(translated_sentences)

    except Exception as e:
        print(f"Translation error (full text): {e}")
        return str(text)

def translate_major_final_in_chunks(input_file: str, output_file: str, chunk_size: int = 1000):
    """
    Translate 'text' column from input CSV in chunks using GoogleTranslator.
    Save to output CSV incrementally.
    """
    print(f"Loading file: {input_file}")
    df = pd.read_csv(input_file)

    if 'text' not in df.columns:
        print("No 'text' column found.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        translated_df = pd.read_csv(output_file)
        start_index = len(translated_df)
        print(f"Resuming from row {start_index}...")
    else:
        translated_df = pd.DataFrame(columns=df.columns)
        start_index = 0

    for i in range(start_index, len(df), chunk_size):
        print(f"Translating rows {i} to {min(i+chunk_size, len(df))-1}")
        chunk = df.iloc[i:i+chunk_size].copy()
        chunk['text'] = chunk['text'].apply(translate_to_english)

        translated_df = pd.concat([translated_df, chunk], ignore_index=True)
        translated_df.to_csv(output_file, index=False)
        print(f"Saved up to row {i + chunk_size}")

if __name__ == "__main__":
    input_file = "preprocessed/major_final.csv"
    output_file = "translated/major_final.csv"
    translate_major_final_in_chunks(input_file, output_file)
