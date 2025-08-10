# Code to find the best configuration based on text extraction results for Activation Steering
import os
import json
import pandas as pd

output_dir = 'results/hallueval_activation_steering/'

results = []
file_map = []
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith('.json'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                data = json.load(f)
                result = {**data.get('metrics', {})}
                results.append(result)
                file_map.append((result, path, data))

df = pd.DataFrame(results)
best_text_extract = df.loc[df['text_extract'].idxmax()]
best_file = None
best_args = None

# Find the file and arguments for the best config
for result, path, data in file_map:
    if result['text_extract'] == best_text_extract['text_extract']:
        best_file = path
        best_args = data['arguments']
        break

print("Best configuration:")
print(f"Text Extract: {best_text_extract['text_extract']}")
print(f"Best file: {best_file}")
print(f"Best layer: {best_args['layers']}")
print(f"Best multiplier: {best_args['multiplier']}")