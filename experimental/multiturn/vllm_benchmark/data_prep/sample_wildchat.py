#!/usr/bin/env python3
"""
Sample conversations from WildChat dataset based on criteria.

Usage:
    python sample_wildchat.py --num-convs 50 --turns 5 --output sample.json
    python sample_wildchat.py --num-convs 100 --min-turns 3 --max-turns 10 --output sample.json
    python sample_wildchat.py --num-convs 50 --turns 5 --min-tokens 100 --max-tokens 1000 --output sample.json
"""

import argparse
import json
import os
import random
from datetime import datetime

from datasets import load_dataset

NUM_PROC = os.cpu_count()


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description="Sample conversations from WildChat")
    parser.add_argument(
        "--num-convs",
        type=int,
        required=True,
        help="Number of conversations to sample",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Exact number of turns (mutually exclusive with min/max-turns)",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=None,
        help="Minimum number of turns",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Minimum user token count",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum user token count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    # Validate args
    if args.turns is not None and (args.min_turns is not None or args.max_turns is not None):
        parser.error("--turns is mutually exclusive with --min-turns/--max-turns")

    print("Loading dataset...")
    ds = load_dataset("inferencemax/WildChat-4.8M-4o-tokcount", split="train")
    print(f"Loaded {len(ds):,} conversations")

    # Show turn count distribution
    from collections import Counter
    dist = Counter(ds["turn"])
    print(f"\nTurn count distribution (top 10):")
    for count, freq in sorted(dist.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count} turns: {freq:,} conversations")

    # Build filter criteria
    turns = args.turns
    min_turns = args.min_turns
    max_turns = args.max_turns
    min_tokens = args.min_tokens
    max_tokens = args.max_tokens

    def matches_criteria(example):
        turn_count = example["turn"]

        if turns is not None and turn_count != turns:
            return False
        if min_turns is not None and turn_count < min_turns:
            return False
        if max_turns is not None and turn_count > max_turns:
            return False
        if min_tokens is not None and example["user_token_count"] < min_tokens:
            return False
        if max_tokens is not None and example["user_token_count"] > max_tokens:
            return False
        return True

    print(f"Filtering for turns={turns}...")
    filtered = ds.filter(matches_criteria, num_proc=NUM_PROC, desc="Filtering")
    print(f"After filtering: {len(filtered):,} conversations")

    if len(filtered) < args.num_convs:
        print(f"Warning: Only {len(filtered)} conversations match criteria, requested {args.num_convs}")
        sample_size = len(filtered)
    else:
        sample_size = args.num_convs

    # Sample
    random.seed(args.seed)
    indices = random.sample(range(len(filtered)), sample_size)
    sampled = filtered.select(indices)

    # Convert to list of dicts for JSON output
    output = []
    for item in sampled:
        output.append({
            "conversation_hash": item["conversation_hash"],
            "model": item["model"],
            "user_token_count": item["user_token_count"],
            "turn_count": item["turn"],
            "conversation": item["conversation"],
        })

    # Write JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

    print(f"Saved {len(output)} conversations to {args.output}")


if __name__ == "__main__":
    main()
