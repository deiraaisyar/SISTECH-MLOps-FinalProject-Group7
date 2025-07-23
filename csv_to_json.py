import os
import csv
import json

input_folder = "translated"  
output_folder = "preprocessed" 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def convert_csv_to_json(csv_file_path, json_file_path):
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]
    
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        csv_file_path = os.path.join(input_folder, file_name)
        json_file_name = os.path.splitext(file_name)[0] + '.json'
        json_file_path = os.path.join(output_folder, json_file_name)
        
        convert_csv_to_json(csv_file_path, json_file_path)
        print(f"File {file_name} berhasil dikonversi ke {json_file_name}")
