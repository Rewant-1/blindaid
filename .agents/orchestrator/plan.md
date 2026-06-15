# Project Plan - Manuscript Revision

This plan outlines the steps required to revise the manuscript to meet requirements R1-R4 and satisfy all acceptance criteria.

## Steps

### Step 1: Initial Exploration and Reference Gathering
- **Task**: Spawn 3 Explorer subagents to run the verification scripts, analyze the current LaTeX file, extract existing numbers, and propose precise changes for main.tex and references.bib.
- **Verification**: Receive Explorer reports containing the precise locations of target text and the correct numbers.

### Step 2: Implementation of Manuscript Changes
- **Task**: Spawn a Worker subagent to modify `paper/main.tex` and `paper/references.bib` based on the Explorers' recommendations.
- **Verification**: Worker runs build/compilation checks on LaTeX (if tools exist) and runs `verify_numbers.py` to check that the implemented numbers match.

### Step 3: Independent Review
- **Task**: Spawn 2 Reviewer subagents to independently review the modified files.
- **Verification**: Reviewers check correctness of text, metric representations, safety arguments, and bibliography format.

### Step 4: Empirical Challenger Verification
- **Task**: Spawn 2 Challenger subagents to verify numerical consistency and check for any remaining gaps.
- **Verification**: Challengers run verification scripts and review output logs.

### Step 5: Integrity Auditing
- **Task**: Spawn a Forensic Auditor subagent to perform integrity forensics.
- **Verification**: Auditor checks that no numbers were hardcoded in verification scripts or falsified.

### Step 6: Synthesis and Completion
- **Task**: Aggregate all subagent reports, update progress, write handoff, and report completion to the Sentinel.
