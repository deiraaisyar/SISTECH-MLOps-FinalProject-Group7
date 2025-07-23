import pandas as pd

# Filepath untuk CSV dan JSON
csv_file = "preprocessed/major_final.csv"  # Ganti dengan path file CSV Anda
json_file = "preprocessed/major_final.json"  # Path untuk file JSON output

# Baca file CSV
df = pd.read_csv(csv_file)

# Konversi ke JSON
df.to_json(json_file, orient="records", lines=True)

print(f"File JSON berhasil dibuat: {json_file}")