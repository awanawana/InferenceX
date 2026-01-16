# Implementation Guide: Wait for Sweep Completion in Claude Workflow

## Problem
Currently, the Claude Code action in `.github/workflows/claude.yml` ends immediately after triggering a sweep, without waiting for it to complete or reporting the results.

## Solution Overview
Add a new step to the Claude workflow that:
1. Detects if a sweep was triggered (by checking if `/sweep` was posted)
2. Waits for the sweep workflow to complete
3. Reports the success/failure status back

## Implementation Steps

### Step 1: Manual Workflow Edit Required
⚠️ **Note**: The Claude Code GitHub App cannot modify workflow files due to permission restrictions. These changes must be made manually by a repository maintainer.

### Step 2: Modify `.github/workflows/claude.yml`

Add the following new step after the "Run Claude Code" step (after line 37):

```yaml
      - name: Run Claude Code
        id: claude
        uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          trigger_phrase: "@claude"
          track_progress: true
          allowed_bots: ''

          claude_args: |
            --allowedTools "Write,Edit,mcp__github_inline_comment__create_inline_comment,Bash(*),Read,Glob,Grep,mcp__github__*"
          prompt: |
            # ... existing prompt content ...

      # NEW STEP: Wait for sweep completion
      - name: Wait for sweep completion
        if: always()
        shell: bash
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          REPO: ${{ github.repository }}
          PR_NUMBER: ${{ github.event.issue.number }}
        run: |
          # Check if Claude triggered a sweep by looking for /sweep in recent comments
          LATEST_COMMENT=$(gh api "/repos/$REPO/issues/$PR_NUMBER/comments" \
            --jq 'map(select(.user.login == "github-actions[bot]" or .user.type == "Bot")) | .[-1].body' || echo "")

          if echo "$LATEST_COMMENT" | grep -q "^/sweep"; then
            echo "Detected sweep trigger, waiting for completion..."

            # Wait a moment for the workflow to start
            sleep 15

            # Get the most recent pr-comment-sweep run
            RUN_ID=$(gh run list --repo "$REPO" \
              --workflow pr-comment-sweep.yml \
              --limit 5 \
              --json databaseId,createdAt \
              --jq '.[0].databaseId')

            if [[ -n "$RUN_ID" && "$RUN_ID" != "null" ]]; then
              echo "Found sweep run: $RUN_ID"
              RUN_URL="https://github.com/$REPO/actions/runs/$RUN_ID"
              echo "Watching run: $RUN_URL"

              # Watch until completion
              if gh run watch "$RUN_ID" --repo "$REPO" --exit-status; then
                echo "✅ Sweep completed successfully!"
                STATUS="success"
                MESSAGE="✅ Sweep completed successfully!"
              else
                echo "❌ Sweep failed"
                STATUS="failure"
                MESSAGE="❌ Sweep failed or was cancelled"
              fi

              # Post result as a comment
              gh issue comment "$PR_NUMBER" --repo "$REPO" --body "### Sweep Result

          $MESSAGE

          [View sweep run]($RUN_URL)"
            else
              echo "Could not find sweep run, skipping wait"
            fi
          else
            echo "No sweep detected, skipping wait"
          fi
```

### Alternative: Using the Helper Script

If you prefer a cleaner approach, you can use the helper script that was created:

```yaml
      - name: Wait for sweep completion
        if: always()
        shell: bash
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # Only run if a sweep was triggered
          LATEST_COMMENT=$(gh api "/repos/${{ github.repository }}/issues/${{ github.event.issue.number }}/comments" \
            --jq 'map(select(.user.login == "github-actions[bot]")) | .[-1].body' || echo "")

          if echo "$LATEST_COMMENT" | grep -q "^/sweep"; then
            .github/scripts/wait-for-sweep.sh \
              "${{ github.repository }}" \
              "${{ github.event.issue.number }}" \
              ""
          else
            echo "No sweep triggered, skipping wait"
          fi
```

### Step 3: Test the Changes

1. Create a test PR
2. Comment with `@claude` and ask it to trigger a sweep
3. Verify that the Claude workflow now waits for the sweep to complete
4. Verify that a follow-up comment is posted with the sweep result

## How It Works

1. **Detection**: After Claude runs, the workflow checks the most recent comments to see if Claude posted a `/sweep` command
2. **Waiting**: If a sweep was triggered, it queries the GitHub API to find the most recent `pr-comment-sweep.yml` run
3. **Monitoring**: Uses `gh run watch` with `--exit-status` to wait for completion and capture the result
4. **Reporting**: Posts a comment with the sweep outcome (success/failure) and a link to the run

## Benefits

- Claude workflows won't complete until sweeps finish
- Immediate feedback on sweep success/failure
- Better integration between Claude and sweep workflows
- Clearer workflow execution tracking

## Limitations

- Requires manual workflow file edit (cannot be automated by Claude)
- Adds wait time to Claude workflow runs (but this is the desired behavior)
- Depends on GitHub CLI and API availability

## Files Created

- `.github/scripts/wait-for-sweep.sh` - Helper script for waiting and reporting (optional)
- `.github/SWEEP_WAIT_IMPLEMENTATION.md` - This implementation guide

## Additional Notes

The script includes:
- Timeout handling (waits up to 60 seconds for workflow to appear)
- Error handling (doesn't fail if sweep isn't found)
- Detailed logging for debugging
- Optional comment update functionality
