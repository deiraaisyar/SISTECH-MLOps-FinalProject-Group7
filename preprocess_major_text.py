import pandas as pd
from app.text_preprocessing import preprocessing

def create_major_text_column():
    """
    Create a 'text' column for major_final.csv using only Prodi (program studi) 
    to avoid hallucination from university names
    """
    # Load the data
    df = pd.read_csv('preprocessed/major_final.csv')
    
    # Handle missing values and create text column using only prodi name
    df['text'] = df['Prodi'].fillna('').apply(lambda x: preprocessing(str(x)) if x else '')
    
    # Remove rows with empty text after preprocessing
    df = df[df['text'].str.strip() != ''].reset_index(drop=True)
    
    # Save back to the same file
    df.to_csv('preprocessed/major_final.csv', index=False)
    print(f"Updated 'text' column in major_final.csv (using only Prodi). Total rows: {len(df)}")
    print("Sample text values:")
    print(df[['Prodi', 'text']].head(10))
    
    # Check for any remaining issues
    empty_text = df[df['text'].str.strip() == '']
    if len(empty_text) > 0:
        print(f"Warning: Found {len(empty_text)} rows with empty text")
    else:
        print("✅ No empty text values found")

if __name__ == "__main__":
    create_major_text_column()
