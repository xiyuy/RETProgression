#!/usr/bin/env python
"""
Diagnostic script to identify file naming mismatches between CSV and disk.
"""

import os
import pandas as pd
import argparse
from pathlib import Path

def diagnose_file_mismatch(data_dir, annotations_file, img_dir):
    """Diagnose why files aren't being found"""
    
    print("="*60)
    print("FILE MISMATCH DIAGNOSTIC REPORT")
    print("="*60)
    
    # Load CSV
    csv_path = os.path.join(data_dir, annotations_file)
    print(f"\n1. Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   - Total rows in CSV: {len(df)}")
    print(f"   - Columns: {list(df.columns)}")
    
    # Show first few entries from CSV
    print(f"\n2. First 10 entries in CSV (first column):")
    for i in range(min(10, len(df))):
        value = df.iloc[i, 0]
        print(f"   [{i}] Raw value: {repr(value)} (type: {type(value).__name__})")
    
    # Check image directory
    img_path = os.path.join(data_dir, img_dir)
    print(f"\n3. Checking image directory: {img_path}")
    
    if not os.path.exists(img_path):
        print(f"   ERROR: Directory does not exist!")
        return
    
    # List actual files in directory
    all_files = []
    for entry in os.scandir(img_path):
        if entry.is_file():
            all_files.append(entry.name)
    
    print(f"   - Total files in directory: {len(all_files)}")
    
    if len(all_files) == 0:
        print(f"   ERROR: No files found in directory!")
        return
    
    # Show sample of actual files
    print(f"\n4. First 10 actual files in directory:")
    for i, filename in enumerate(sorted(all_files)[:10]):
        print(f"   [{i}] {filename}")
    
    # Analyze filename patterns
    print(f"\n5. File extension analysis:")
    extensions = {}
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    for ext, count in sorted(extensions.items(), key=lambda x: -x[1]):
        print(f"   {ext}: {count} files")
    
    # Create lookup sets for fast matching
    print(f"\n6. Creating filename lookup tables...")
    
    # Set of actual files (with different variations)
    actual_files_set = set(all_files)
    actual_stems = {os.path.splitext(f)[0]: f for f in all_files}
    actual_stems_lower = {os.path.splitext(f)[0].lower(): f for f in all_files}
    
    # Try to match CSV entries
    print(f"\n7. Attempting to match CSV entries to files:")
    
    matched = 0
    unmatched_samples = []
    
    for idx in range(len(df)):
        csv_value = str(df.iloc[idx, 0])
        found = False
        attempted = []
        
        # Try different matching strategies
        strategies = [
            (csv_value, "exact"),
            (f"{csv_value}.jpg", "add .jpg"),
            (f"{csv_value}.JPG", "add .JPG"),
            (f"{csv_value}.png", "add .png"),
            (f"{csv_value}.PNG", "add .PNG"),
        ]
        
        # Handle float conversion (e.g., "49281.0" -> "49281")
        if '.' in csv_value and csv_value.replace('.', '').replace('-', '').isdigit():
            base = csv_value.split('.')[0]
            strategies.extend([
                (base, "remove decimal"),
                (f"{base}.jpg", "remove decimal + .jpg"),
                (f"{base}.JPG", "remove decimal + .JPG"),
            ])
        
        for filename, strategy in strategies:
            attempted.append(f"{filename} ({strategy})")
            if filename in actual_files_set:
                matched += 1
                found = True
                break
            
            # Also try stem matching
            stem = os.path.splitext(filename)[0]
            if stem in actual_stems:
                matched += 1
                found = True
                break
            if stem.lower() in actual_stems_lower:
                matched += 1
                found = True
                break
        
        if not found and len(unmatched_samples) < 10:
            unmatched_samples.append({
                'index': idx,
                'csv_value': csv_value,
                'attempted': attempted
            })
    
    print(f"   - Matched: {matched}/{len(df)} ({100*matched/len(df):.1f}%)")
    print(f"   - Unmatched: {len(df)-matched}/{len(df)} ({100*(len(df)-matched)/len(df):.1f}%)")
    
    # Show unmatched samples
    if unmatched_samples:
        print(f"\n8. Examples of unmatched entries (up to 10):")
        for sample in unmatched_samples:
            print(f"   CSV row {sample['index']}: '{sample['csv_value']}'")
            print(f"      Attempted matches:")
            for attempt in sample['attempted'][:5]:
                print(f"        - {attempt}")
    
    # Suggest possible issues
    print(f"\n9. DIAGNOSTIC SUMMARY:")
    if matched < len(df) * 0.1:
        print("   ⚠ CRITICAL: Less than 10% of files matched!")
        print("   Possible issues:")
        print("   - Wrong image directory specified")
        print("   - CSV contains IDs that need transformation to match filenames")
        print("   - Files have different naming convention than expected")
        print("   - Files might be in subdirectories")
        
        # Try to find pattern
        print(f"\n10. Attempting pattern detection:")
        
        # Check if files have common prefix/suffix
        if all_files:
            # Check for common prefixes
            prefixes = {}
            for f in all_files[:100]:  # Sample first 100
                if '_' in f:
                    prefix = f.split('_')[0]
                    prefixes[prefix] = prefixes.get(prefix, 0) + 1
            
            if prefixes:
                print("   Common prefixes found:")
                for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:5]:
                    print(f"     - '{prefix}_': {count} occurrences")
        
        # Check if CSV values might be IDs that need prefix
        print("\n   Suggested fixes:")
        print("   1. Check if files are in a different directory")
        print("   2. Check if filenames have a prefix/suffix not in CSV")
        print("   3. Check if CSV contains patient IDs that map to different filenames")
        print("   4. Verify the image directory path is correct")
    else:
        print(f"   ✓ Successfully matched {matched}/{len(df)} files")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Diagnose file mismatch issues')
    parser.add_argument('--data_dir', required=True, help='Root data directory')
    parser.add_argument('--annotations_file', required=True, help='CSV file with annotations')
    parser.add_argument('--img_dir', required=True, help='Image directory name')
    
    args = parser.parse_args()
    
    diagnose_file_mismatch(args.data_dir, args.annotations_file, args.img_dir)

if __name__ == '__main__':
    main()