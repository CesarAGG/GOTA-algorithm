# GLOTA: Globally-Optimal Log Truncation Algorithm

GLOTA is a high-performance Python implementation designed to truncate log files by identifying and summarizing repetitive sequences. It aims for optimal compression in its structured output and enhanced human readability by prioritizing globally significant patterns and offering flexible display options for summarized contexts.

The algorithm is particularly well-suited for analyzing large log files where identifying recurring patterns can significantly reduce log size and aid in diagnostics by highlighting anomalies rather than redundant information.

## Features

*   **Optimal Summarization:** Designed to minimize the number of output items (literals and summary messages).
*   **Hierarchical Summarization:** Accurately represents and summarizes patterns that are nested within other patterns.
*   **Advanced Normalization (Phase 0):**
    *   Supports customizable regex-based substitution to normalize log lines (e.g., remove timestamps, anonymize IPs/UserIDs) before pattern matching.
    *   Efficiently interns unique normalized lines to integer IDs for fast processing.
    *   Preserves original log lines for non-lossy final output.
*   **Efficient Run Finding (Phase 1):** Implements a sophisticated O(N²) algorithm (where N is the number of processed log lines) to find all summarizable maximal primitive runs, with practical performance enhancements if a maximum primitive block length (`n_cap`) is specified.
*   **Optimized Summarization Pass (Phase 2):** Utilizes an interval tree for efficient O(N log N) processing of candidate runs to determine the final set of committed summaries, resolving overlaps and hierarchical relationships.
*   **Recursive Summary Representation (Phase 3 & 4):**
    *   Summary messages can recursively detail the structure of the pattern (`U`) they represent.
    *   Flexible display of the `t` literal instances: users can choose to see a configurable number of lines from the beginning and end of the literal block, with the summary message covering the middle.
    *   Top-level literal lines in the output are always the original (pre-normalization) lines, while the `U` part of summary messages uses normalized lines.
*   **Performance Optimizations:** Leverages NumPy for efficient array operations and Numba (if available) for JIT compilation of performance-critical sections.
*   **Debuggability:** Optional detailed debug log outputs for each phase.

## Algorithm Overview

GLOTA processes logs in four main phases:

1.  **Phase 0: Log Preprocessing:**
    *   Reads the raw log file line by line.
    *   Applies user-defined regex substitutions to normalize each line (e.g., remove timestamps, mask sensitive data).
    *   Optionally filters out lines that become empty after normalization.
    *   Converts unique normalized log lines into compact `uint16` integer IDs. This step will error if more than 65,536 unique normalized lines are found (due to `uint16` limit).
    *   Outputs a sequence of these integer IDs (`L_hashed`), a mapping from IDs back to normalized strings, and a list of the original (pre-regex-normalization) strings corresponding to each position in `L_hashed`.

2.  **Phase 1: Candidate Generation & Sorting:**
    *   Identifies all "summarizable maximal primitive runs" `(U, i, p, k)` from `L_hashed`. `U` is the primitive sequence of IDs, `i` its start index, `p` its length, and `k` its number of maximal repetitions (where `k > t_threshold`).
    *   An optional `n_cap` parameter can limit the maximum length `p` of primitives considered.
    *   The identified runs are sorted primarily by `p` (descending) and secondarily by `i` (ascending) to create `SortedMasterRuns`.
    *   This phase is implemented with an O(N²) algorithm leveraging LPS and Z-arrays.

3.  **Phase 2: Single-Pass Iterative Summarization:**
    *   Processes `SortedMasterRuns` using an interval tree to efficiently manage and query already "committed" summaries.
    *   Applies GLOTA's rules to resolve overlaps, handle truncations, and respect hierarchical summarization (i.e., summaries within the `t_threshold` literal repetitions of a parent summary). It prunes redundant phase-shifted patterns and patterns that straddle parent primitive blocks.
    *   Outputs a list of `CommittedSummaries`.

4.  **Phase 3: Hierarchical Output Generation (Hashed):**
    *   Takes `L_hashed` and `CommittedSummaries` to produce a structured, summarized representation of the log, still using integer IDs.
    *   It recursively renders segments. When a committed summary `S_next=(U,k,...)` is processed:
        *   It renders the first instance of `U` (let this be `Rendered_U0`). This `Rendered_U0` itself can contain literals and nested summaries.
        *   It then assembles the output for `S_next`: a configurable number of leading literal instances of `U` (derived by "shifting" `Rendered_U0`), followed by a summary object for `S_next` (whose `U` part *is* `Rendered_U0`), followed by a configurable number of trailing literal instances of `U` (also derived by shifting `Rendered_U0`).
    *   The output is a list of `('LITERAL', original_log_idx)` or `('SUMMARY_R', Rendered_U0, k-t, ...)` items.

5.  **Phase 4: De-Hashing and Final Output Formatting:**
    *   Takes the structured hashed output from Phase 3.
    *   De-hashes integer IDs to strings:
        *   Top-level `LITERAL` items are converted to their *original* (pre-regex-normalization) string form.
        *   The `U` part of `SUMMARY_R` messages (which is `Rendered_U0`) is de-hashed such that its literal components become *normalized* strings, and its nested summaries are recursively formatted with user-friendly, chevron-based nesting and dynamic separators.
    *   Prints the final human-readable, truncated log to console or a specified file.

## Conjectures and Corollaries

The design and implementation of GLOTA are based on the following key understandings and conjectures, which contribute to its efficiency and the quality of its output:

1.  **Optimality Conjecture:** The GLOTA algorithm, through its phased approach (sorting in Phase 1 and rule-based commitments in Phase 2), produces a summary that is optimal in terms of minimizing the total count of output items (top-level literals or `SUMMARY_R` objects) in its Phase 3 structured representation. Pruning rules are designed to remove only redundant representations.
2.  **Phase 1 Performance Corollary (with `n_cap`):** The O(N²) complexity of Phase 1 can see significant practical speedups if `n_cap` (maximum primitive length) is small relative to N, potentially making the oracle component behave closer to O(M) (linear in sequence length scanned) rather than O(M log M).
3.  **Phase 1 Output Size Corollary:** The number of distinct maximal primitive runs (`M_runs` in `SortedMasterRuns`, with `k >= 2`) generated by Phase 1 is O(N) (linear in the length of the processed log), a known result from combinatorics on words.
4.  **Phase 2 Performance Corollary:** Given `M_runs` is O(N), Phase 2 using an interval tree achieves an O(N log N) complexity. The optimized Rule B (point query for suffix truncation) further enhances practical speed.
5.  **Readability Corollary (Recursive Summaries & Start/End Display):** The "Render First (Rep 0), Shift Others" strategy in Phase 3, configurable `t_start_display`/`t_end_display`, and recursive formatting of `U` strings in Phase 4 significantly enhance readability for logs with complex nested or long repetitions by presenting summaries in their full structural context.
6.  **Sufficiency of Phase 2 Hierarchical Commit Rule:** Committing child summaries in Phase 2 only if they reside in the *first* conceptual block (`rep_idx=0`) of their parent (and pass straddling/validity checks) is sufficient when Phase 3 employs the "Render First (Rep 0), Shift Others" strategy. This simplifies Phase 2 without loss of fidelity.

## Prerequisites

*   Python 3.8+
*   NumPy (`pip install numpy`)
*   tqdm (`pip install tqdm`)
*   Numba (`pip install numba`)
*   IntervalTree (`pip install intervaltree`)

## Execution

To run the GLOTA algorithm:

```bash
python glota.py --log_file <path_to_your_log_file> [options]
```

**Command-Line Arguments:**

*   `--log_file PATH`: (Required) Path to the raw input log file.
*   `--regex_file PATH`: (Optional) Path to a file containing regex patterns for normalization.
    *   Format per line: `pattern_to_find|||replacement_string|||pattern_name_for_debug`
    *   Example: `\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*|||""|||timestamp_removal`
*   `--skip_blank_normalized` / `--no-skip-blank-normalized`: (Optional) Skip lines that become blank after all normalizations. (Default: True, skip blanks)
*   `--debug_output_file PREFIX`: (Optional) Base prefix for debug output files. Generates:
    *   `<PREFIX>.phase0.csv`
    *   `<PREFIX>.phase1_runs.log`
    *   `<PREFIX>.phase2_committed.log`
    *   `<PREFIX>.phase3_hashed_output.log`
*   `--encoding ENCODING`: (Optional) Character encoding for input and debug files. (Default: `utf-8`)
*   `--t_threshold T_INT`: (Optional) Summarization threshold `t`. A run `U^k` is summarized if `k > t`. Must be `>= 1`. (Default: `1`)
*   `--n_cap N_CAP_INT`: (Optional) Maximum primitive block length `p` considered by Phase 1. `0` means no cap. (Default: `0`)
*   `--t_start_display T_START_INT`: (Optional) For `t_threshold >= 2`, number of literal repetitions to show at the start of a `t`-block. (Default: `floor(t_threshold/2)`)
*   `--t_end_display T_END_INT`: (Optional) For `t_threshold >= 2`, number of literal repetitions to show at the end of a `t`-block. (Default: `ceil(t_threshold/2)`)
*   `--output_file PATH`: (Optional) Path to write the final de-hashed summarized log output. If not provided, output is printed to the console.

**Example Usage:**

```bash
python glota.py --log_file application.log --regex_file my_regexes.txt --t_threshold 2 --n_cap 30 --debug_output_file debug/app_summary --output_file summarized_app.log
```

This command would:
*   Process `application.log`.
*   Use regexes from `my_regexes.txt` for normalization.
*   Use `t=2` for summarization decisions.
*   Limit Phase 1 to primitives of length up to 30.
*   Create debug files like `debug/app_summary.phase0.csv`, etc.
*   Write the final summarized log to `summarized_app.log`.
*   Use the default `t_start_display=1`, `t_end_display=1` for the literal parts of summaries.

## Contributing

Contributions are welcome! If you have suggestions for improvements, optimizations, or bug fixes, please consider the following:

1.  **Open an Issue:** Discuss the change you wish to make.
2.  **Fork the Repository:** Create your own fork to work on.
3.  **Create a Branch:** Make your changes in a dedicated branch.
4.  **Test Thoroughly:** Ensure your changes pass existing tests (if any) and add new tests for new functionality.
5.  **Follow Code Style:** Maintain a consistent code style (e.g., PEP 8 for Python).
6.  **Submit a Pull Request:** Clearly describe your changes and their purpose.

When proposing algorithmic changes, please provide clear reasoning and, if possible, test cases or examples that demonstrate the impact of the change.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
