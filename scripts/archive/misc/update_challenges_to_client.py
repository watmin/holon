#!/usr/bin/env python3
"""
Script to help update challenge solutions to use HolonClient instead of CPUStore directly.
"""

import os
import re
import glob

def update_file(filepath):
    """Update a single file to use HolonClient."""
    print(f"Processing {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # Add HolonClient import if CPUStore import exists
    if 'from holon import CPUStore' in content:
        content = content.replace(
            'from holon import CPUStore',
            'from holon import CPUStore, HolonClient'
        )

    # Find store initialization patterns
    store_init_patterns = [
        r'store = CPUStore\(\)',
        r'store = CPUStore\(dimensions=\d+\)',
        r'store = CPUStore\(backend=.*\)',
    ]

    for pattern in store_init_patterns:
        if re.search(pattern, content):
            # Add client initialization after store
            content = re.sub(
                r'(store = CPUStore\([^)]*\))\n',
                r'\1\n    client = HolonClient(store)\n',
                content
            )
            break

    # Update function parameters from store to client
    content = re.sub(r'def (\w+)\(store,', r'def \1(client,', content)

    # Update function calls from store to client
    content = re.sub(r'(\w+)\(store,', r'\1(client,', content)

    # Update store.insert to client.insert_json for JSON data
    # This is more complex, so we'll do basic patterns
    content = re.sub(r'store\.insert\(json\.dumps\(([^)]+)\)\)', r'client.insert_json(\1)', content)

    # Update store.query to client.search_json
    # This needs more careful handling
    content = re.sub(r'store\.query\(([^,]+),\s*([^)]*)\)', r'client.search_json(\1, \2)', content)

    # Update result iteration patterns
    # store.query returns (id, score, data), client.search_json returns {"id": id, "score": score, "data": data}
    content = re.sub(
        r'for (.*) in results:.*?\n.*?data = \1\[2\]',
        r'for result in results:\n            data = result["data"]',
        content,
        flags=re.DOTALL
    )

    # Write back if changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Updated {filepath}")
        return True
    else:
        print(f"  - No changes needed for {filepath}")
        return False

def main():
    """Update all challenge solution files."""
    challenge_dirs = [
        'scripts/challenges/001-batch',
        'scripts/challenges/002-batch',
        'scripts/challenges/003-batch',
        'scripts/challenges/004-batch',
        'scripts/challenges/005-batch',
        'scripts/challenges/006-batch',
    ]

    total_updated = 0

    for challenge_dir in challenge_dirs:
        if os.path.exists(challenge_dir):
            # Find all Python files in the directory
            pattern = os.path.join(challenge_dir, '*.py')
            files = glob.glob(pattern)

            print(f"\n📁 Processing {challenge_dir} ({len(files)} files)...")

            for filepath in files:
                if update_file(filepath):
                    total_updated += 1

    print(f"\n🎉 Updated {total_updated} files total")
    print("Note: Manual review may be needed for complex query patterns")

if __name__ == "__main__":
    main()
