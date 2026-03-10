import pandas as pd
import re
import os

def extract_metadata(readme_path):
    metadata_list = []
    
    with open(readme_path, 'r') as f:
        lines = f.readlines()
    
    # Match lines start after the introductory text
    # Looking for lines that look like: YYYY-MM-DD - ...
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}')
    
    for line in lines:
        line = line.strip()
        if date_pattern.match(line):
            parts = [p.strip() for p in line.split(' - ')]
            if len(parts) >= 6:
                metadata_list.append({
                    'date': parts[0],
                    'team_type': parts[1],
                    'match_type': parts[2],
                    'gender': parts[3],
                    'match_id': parts[4],
                    'teams': parts[5]
                })
    
    return metadata_list

def main():
    readme_path = './Research/README.txt'
    output_path = './data/match_metadata.csv'
    
    if not os.path.exists(readme_path):
        print(f"Error: {readme_path} not found.")
        return
        
    print(f"Extracting metadata from {readme_path}...")
    metadata = extract_metadata(readme_path)
    
    if not metadata:
        print("No metadata found.")
        return
        
    df = pd.DataFrame(metadata)
    # Ensure ID is treated consistently (often string/category)
    df['match_id'] = df['match_id'].astype(str)
    
    print(f"Found {len(df)} matches.")
    df.to_csv(output_path, index=False)
    print(f"Metadata saved to {output_path}")

if __name__ == "__main__":
    main()
