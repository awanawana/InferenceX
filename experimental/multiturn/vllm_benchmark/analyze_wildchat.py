#!/usr/bin/env python3
"""
Analyze the filtered WildChat dataset.

Usage:
    python analyze_wildchat.py
"""

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset


def main():
    print("Loading dataset...")
    ds = load_dataset("inferencemax/WildChat-4.8M-4o-tokcount", split="train")
    print(f"Loaded {len(ds):,} conversations")

    token_counts = ds["user_token_count"]
    turn_counts = ds["turn"]

    # Basic stats
    print(f"\nUser token count stats:")
    print(f"  Min: {min(token_counts):,}")
    print(f"  Max: {max(token_counts):,}")
    print(f"  Mean: {sum(token_counts) / len(token_counts):,.0f}")

    print(f"\nTurn count stats:")
    print(f"  Min: {min(turn_counts):,}")
    print(f"  Max: {max(turn_counts):,}")
    print(f"  Mean: {sum(turn_counts) / len(turn_counts):,.1f}")

    # Histogram with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Full range (0-13k)
    ax1.hist(token_counts, bins=500, edgecolor="black", alpha=0.7, range=(0, 13000))
    ax1.set_xlabel("User Token Count")
    ax1.set_ylabel("Frequency")
    ax1.set_xlim(0, 13000)
    ax1.set_title("Full Range (0-13k)")

    # Zoomed in (0-3k)
    ax2.hist(token_counts, bins=300, edgecolor="black", alpha=0.7, range=(0, 3000))
    ax2.set_xlabel("User Token Count")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(0, 3000)
    ax2.set_title("Zoomed (0-3k)")

    fig.suptitle("Distribution of User Token Counts (WildChat-4.8M-4o)")
    plt.tight_layout()
    plt.savefig("user_token_histogram.png", dpi=150)
    print("\nSaved histogram to user_token_histogram.png")
    plt.show()

    # Turn count histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(turn_counts, bins=100, edgecolor="black", alpha=0.7, range=(0, 100))
    ax1.set_xlabel("Number of Turns")
    ax1.set_ylabel("Frequency")
    ax1.set_xlim(0, 100)
    ax1.set_title("Full Range (0-100 turns)")

    ax2.hist(turn_counts, bins=50, edgecolor="black", alpha=0.7, range=(0, 20))
    ax2.set_xlabel("Number of Turns")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(0, 20)
    ax2.set_title("Zoomed (0-20 turns)")

    fig.suptitle("Distribution of Turn Counts (WildChat-4.8M-4o)")
    plt.tight_layout()
    plt.savefig("turn_count_histogram.png", dpi=150)
    print("Saved histogram to turn_count_histogram.png")
    plt.show()

    # Hexbin: Token count vs Turn count
    fig, ax = plt.subplots(figsize=(10, 8))
    hb = ax.hexbin(
        turn_counts,
        token_counts,
        gridsize=50,
        cmap="YlOrRd",
        mincnt=1,
        extent=[0, 50, 0, 10000],
    )
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("User Token Count")
    ax.set_title("Token Count vs Turn Count (WildChat-4.8M-4o)")
    cb = fig.colorbar(hb, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig("token_vs_turn_hexbin.png", dpi=150)
    print("Saved hexbin to token_vs_turn_hexbin.png")
    plt.show()


if __name__ == "__main__":
    main()
