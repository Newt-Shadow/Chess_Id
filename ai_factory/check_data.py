from pathlib import Path
from collections import Counter

# Path to your processed data
p = Path("datasets/processed/train/images")

# Count prefixes
counts = Counter()
for f in p.glob("*"):
    # Get the prefix (e.g., 'rf_merged_' from 'rf_merged_001.jpg')
    prefix = f.name.split('_')[0] + "_" + f.name.split('_')[1]
    counts[prefix] += 1

print(f"📊 DATASET INVENTORY (Train Split):")
print(f"-----------------------------------")
for prefix, count in counts.most_common():
    print(f"✅ {prefix:<20} : {count} images")

print(f"-----------------------------------")
print(f"TOTAL TRAIN IMAGES   : {sum(counts.values())}")