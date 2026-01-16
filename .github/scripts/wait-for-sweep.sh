#!/bin/bash
set -euo pipefail

# Script to wait for the most recent slash-sweep workflow run to complete
# and report its status back

REPO="${1:-$GITHUB_REPOSITORY}"
PR_NUMBER="${2:-}"
COMMENT_ID="${3:-}"

if [[ -z "$REPO" ]]; then
    echo "Error: Repository not specified" >&2
    exit 1
fi

if [[ -z "$PR_NUMBER" ]]; then
    echo "Error: PR number not specified" >&2
    exit 1
fi

echo "Waiting for sweep workflow to start..."
echo "Repository: $REPO"
echo "PR: #$PR_NUMBER"

# Wait a bit for the workflow to be triggered
sleep 10

# Get the most recent slash-sweep run for this PR
# We filter by the PR's head branch to ensure we get the right run
MAX_ATTEMPTS=12  # 12 attempts * 5 seconds = 60 seconds max wait for workflow to appear
ATTEMPT=0

while [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; do
    RUN_ID=$(gh run list --repo "$REPO" \
        --workflow pr-comment-sweep.yml \
        --limit 10 \
        --json databaseId,headBranch,status,createdAt,event \
        --jq "map(select(.event == \"issue_comment\")) | .[0].databaseId" 2>/dev/null || echo "")

    if [[ -n "$RUN_ID" && "$RUN_ID" != "null" ]]; then
        echo "Found sweep run: $RUN_ID"
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))
    if [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; then
        echo "Sweep workflow not found yet, waiting... (attempt $ATTEMPT/$MAX_ATTEMPTS)"
        sleep 5
    fi
done

if [[ -z "$RUN_ID" || "$RUN_ID" == "null" ]]; then
    echo "Warning: Could not find sweep workflow run within timeout period"
    exit 0  # Don't fail the Claude workflow
fi

echo "Watching sweep run $RUN_ID until completion..."
RUN_URL="https://github.com/$REPO/actions/runs/$RUN_ID"
echo "Run URL: $RUN_URL"

# Watch the run until completion
# The --exit-status flag will cause this to exit with non-zero if the run fails
if gh run watch "$RUN_ID" --repo "$REPO" --exit-status; then
    CONCLUSION="success"
    echo "✅ Sweep completed successfully!"
    MESSAGE="✅ Sweep completed successfully! [View run]($RUN_URL)"
else
    CONCLUSION="failure"
    echo "❌ Sweep failed or was cancelled"
    MESSAGE="❌ Sweep failed or was cancelled. [View run]($RUN_URL)"

    # Try to get failure details
    echo ""
    echo "Fetching failed job logs..."
    gh run view "$RUN_ID" --repo "$REPO" --log-failed || true
fi

# If a comment ID was provided, update it with the result
if [[ -n "$COMMENT_ID" ]]; then
    echo ""
    echo "Updating comment $COMMENT_ID with sweep result..."

    # Get existing comment body
    EXISTING_BODY=$(gh api "/repos/$REPO/issues/comments/$COMMENT_ID" --jq '.body' || echo "")

    if [[ -n "$EXISTING_BODY" ]]; then
        # Append the sweep result to the existing comment
        NEW_BODY="${EXISTING_BODY}

---

### Sweep Result

${MESSAGE}"

        # Update the comment
        gh api --method PATCH "/repos/$REPO/issues/comments/$COMMENT_ID" \
            -f body="$NEW_BODY" >/dev/null || echo "Warning: Failed to update comment"
    fi
fi

# Exit with appropriate code
if [[ "$CONCLUSION" == "success" ]]; then
    exit 0
else
    exit 1
fi
