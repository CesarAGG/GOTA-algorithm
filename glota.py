# main_glota.py

import re
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional, TextIO, Set

from numba import jit, types as nb_types
from intervaltree import Interval, IntervalTree

print("Numba JIT compiler and Intervaltree library are available and will be used.")


# --- Phase 0 Functions ---
def load_regex_patterns_from_file(
    regex_file_path: str, encoding: str
) -> List[Tuple[str, str, str]]:
    patterns: List[Tuple[str, str, str]] = []
    try:
        with open(regex_file_path, "r", encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|||")
                if len(parts) == 3:
                    patterns.append(
                        (parts[0].strip(), parts[1].strip(), parts[2].strip())
                    )
                else:
                    print(
                        f"Warning: Malformed regex line {line_num} in {regex_file_path}. Expected 3 parts separated by '|||'. Got {len(parts)} parts. Skipping: '{line}'"
                    )
    except FileNotFoundError:
        print(
            f"Warning: Regex file not found at {regex_file_path}. No regex patterns will be loaded."
        )
    except IOError as e:
        print(
            f"Warning: Error reading regex file {regex_file_path}: {e}. No regex patterns will be loaded."
        )
    return patterns


def Phase0_Preprocess(
    raw_log_file_path: str,
    regex_sub_pattern_list: List[Tuple[str, str, str]],
    skip_blank_normalized_lines: bool = True,
    debug_output_path: Optional[str] = None,
    character_encoding: str = "utf-8",
) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[str]]]:
    normalized_string_to_id: Dict[str, int] = {}
    id_counter: int = 0
    id_to_normalized_string: List[str] = []
    L_processed_log_data: List[Tuple[int, str]] = []

    debug_file_handle: Optional[TextIO] = None
    compiled_regexes: List[Tuple[re.Pattern, str, str]] = []

    for pattern_str, repl_str, name_str in regex_sub_pattern_list:
        try:
            compiled_regexes.append((re.compile(pattern_str), repl_str, name_str))
        except re.error as e:
            print(f"Error: Invalid regex pattern '{name_str}' ('{pattern_str}'): {e}")
            return None, None, None

    if debug_output_path:
        try:
            debug_file_handle = open(
                debug_output_path, "w", encoding=character_encoding
            )
            debug_file_handle.write(
                "Assigned_ID,Original_Raw_Line,Matched_Regex_Names,Normalized_Line\n"
            )
        except IOError as e:
            print(
                f"Warning: Could not open debug file {debug_output_path}: {e}. Debug output disabled."
            )
            debug_file_handle = None

    line_count_for_pbar = 0
    actual_processed_lines = 0
    log_file_path_obj = Path(raw_log_file_path)

    if not log_file_path_obj.is_file():
        print(f"Error: Log file not found at {raw_log_file_path}")
        if debug_file_handle:
            debug_file_handle.close()
        return None, None, None

    try:
        with open(
            raw_log_file_path, "r", encoding=character_encoding, errors="ignore"
        ) as f_count:
            line_count_for_pbar = sum(1 for _ in f_count)
    except Exception as e:
        print(
            f"Warning: Could not count lines in {raw_log_file_path} for progress bar: {e}"
        )
        line_count_for_pbar = 0

    pbar_instance: Optional[tqdm] = None
    try:
        with open(raw_log_file_path, "r", encoding=character_encoding) as f_in:
            pbar_instance = tqdm(
                total=line_count_for_pbar, unit="line", desc="Phase 0 Preprocessing"
            )
            for original_line_raw in f_in:
                actual_processed_lines += 1
                original_line_for_processing = original_line_raw.strip()
                current_line_for_normalization = original_line_for_processing
                matched_patterns_for_this_line: List[str] = []

                for compiled_pattern, replacement, pattern_name in compiled_regexes:
                    prev_line_state = current_line_for_normalization
                    current_line_for_normalization = compiled_pattern.sub(
                        replacement, current_line_for_normalization
                    )
                    if (
                        debug_file_handle
                        and current_line_for_normalization != prev_line_state
                    ):
                        matched_patterns_for_this_line.append(pattern_name)

                normalized_entry_string = " ".join(
                    current_line_for_normalization.split()
                )

                if skip_blank_normalized_lines and not normalized_entry_string:
                    if debug_file_handle:
                        orig_escaped = original_line_raw.rstrip("\n").replace('"', '""')
                        norm_escaped = normalized_entry_string.replace('"', '""')
                        matched_escaped = "|".join(
                            matched_patterns_for_this_line
                        ).replace('"', '""')
                        debug_file_handle.write(
                            f'SKIPPED,"{orig_escaped}","{matched_escaped}","{norm_escaped}"\n'
                        )
                    if pbar_instance:
                        pbar_instance.update(1)
                    continue

                current_numeric_id: int
                if normalized_entry_string not in normalized_string_to_id:
                    if id_counter > 65535:
                        print(
                            f"Error: Exceeded maximum unique log entries (65535) for uint16. Stopping."
                        )
                        if pbar_instance:
                            pbar_instance.total = pbar_instance.n
                        if debug_file_handle:
                            debug_file_handle.close()
                        return None, None, None

                    normalized_string_to_id[normalized_entry_string] = id_counter
                    id_to_normalized_string.append(normalized_entry_string)
                    current_numeric_id = id_counter
                    id_counter += 1
                else:
                    current_numeric_id = normalized_string_to_id[
                        normalized_entry_string
                    ]

                L_processed_log_data.append(
                    (current_numeric_id, original_line_for_processing)
                )

                if debug_file_handle:
                    orig_escaped = original_line_raw.rstrip("\n").replace('"', '""')
                    norm_escaped = normalized_entry_string.replace('"', '""')
                    matched_escaped = "|".join(matched_patterns_for_this_line).replace(
                        '"', '""'
                    )
                    debug_file_handle.write(
                        f'{current_numeric_id},"{orig_escaped}","{matched_escaped}","{norm_escaped}"\n'
                    )
                if pbar_instance:
                    pbar_instance.update(1)
    except FileNotFoundError:
        print(f"Error: Log file not found at {raw_log_file_path}")
        return None, None, None
    except IOError as e:
        print(f"Error reading log file {raw_log_file_path}: {e}")
        return None, None, None
    except UnicodeDecodeError as e:
        print(
            f"Error decoding file {raw_log_file_path} with encoding {character_encoding}: {e}"
        )
        return None, None, None
    finally:
        if pbar_instance:
            pbar_instance.close()
        if debug_file_handle:
            debug_file_handle.close()

    if not L_processed_log_data and id_counter == 0:
        print(
            f"Phase 0: Processed {actual_processed_lines} lines. No processable log entries found."
        )
        return np.array([], dtype=np.uint16), [], []

    L_hashed_numpy_array = np.array(
        [item[0] for item in L_processed_log_data], dtype=np.uint16
    )
    original_lines_by_position_list = [item[1] for item in L_processed_log_data]

    print(
        f"Phase 0: Processed {actual_processed_lines} lines. Found {id_counter} unique normalized entries (max 65536 for uint16)."
    )
    print(
        f"Phase 0: Output log length (hashed): {len(L_hashed_numpy_array)}, dtype: {L_hashed_numpy_array.dtype}"
    )

    return (
        L_hashed_numpy_array,
        id_to_normalized_string,
        original_lines_by_position_list,
    )


# --- Phase 1 Functions ---
LPS_Z_SIGNATURE = nb_types.int32[:](nb_types.uint16[:])


@jit(LPS_Z_SIGNATURE, nopython=True, cache=True)
def _compute_lps_array_phase1(pattern_view: np.ndarray) -> np.ndarray:
    m = len(pattern_view)
    if m == 0:
        return np.empty(0, dtype=np.int32)
    lps = np.zeros(m, dtype=np.int32)
    length: int = 0
    i: int = 1
    while i < m:
        if pattern_view[i] == pattern_view[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


@jit(LPS_Z_SIGNATURE, nopython=True, cache=True)
def _compute_z_array_phase1(s_view: np.ndarray) -> np.ndarray:
    n = len(s_view)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    z = np.zeros(n, dtype=np.int32)
    l: int = 0
    r: int = 0
    for i in range(1, n):
        if i <= r:
            val1: int = r - i + 1
            val2: int = z[i - l]
            z[i] = min(val1, val2)

        idx_zi: int = int(z[i])
        idx_i_plus_zi: int = i + idx_zi
        while idx_i_plus_zi < n and s_view[idx_zi] == s_view[idx_i_plus_zi]:
            z[i] += 1
            idx_zi = int(z[i])
            idx_i_plus_zi = i + idx_zi

        current_z_val_for_r_update: int = int(z[i])
        if i + current_z_val_for_r_update - 1 > r:
            l, r = i, i + current_z_val_for_r_update - 1

    if n > 0:
        z[0] = n
    return z


def Phase1_FindAndSort_SummarizableRuns(
    log_hashed: np.ndarray, t_threshold: int, n_cap: int = 0
) -> List[Tuple[List[int], int, int, int]]:
    N = len(log_hashed)
    if N == 0:
        return []
    all_summarizable_runs: List[Tuple[List[int], int, int, int]] = []

    pbar_phase1: Optional[tqdm] = None
    try:
        pbar_phase1 = tqdm(total=N, unit="pos", desc="Phase 1 Run Finding")
        for i in range(N):
            s_suffix_view = log_hashed[i:]
            M = len(s_suffix_view)
            if M < (t_threshold + 1):
                if pbar_phase1:
                    pbar_phase1.update(1)
                continue

            lps_suffix = _compute_lps_array_phase1(s_suffix_view)
            z_suffix = _compute_z_array_phase1(s_suffix_view)

            yielded_U_prim_hashes_for_suffix: Set[Tuple[int, ...]] = set()

            for k_len in range(t_threshold + 1, M + 1):
                lps_val_for_pk: int = int(lps_suffix[k_len - 1])
                p_val: int = k_len - lps_val_for_pk

                if p_val == 0:
                    continue
                if n_cap > 0 and p_val > n_cap:
                    continue

                if k_len % p_val == 0:
                    num_reps_in_prefix: int = k_len // p_val

                    if num_reps_in_prefix > t_threshold:
                        U_prim_view = s_suffix_view[0:p_val]
                        U_prim_tuple = tuple(int(x) for x in U_prim_view)

                        if U_prim_tuple not in yielded_U_prim_hashes_for_suffix:
                            yielded_U_prim_hashes_for_suffix.add(U_prim_tuple)

                            k_in_suffix: int = 1 + (int(z_suffix[p_val]) // p_val)

                            if k_in_suffix > t_threshold:
                                is_left_maximal = True
                                if i > 0 and p_val <= i:
                                    if np.array_equal(
                                        log_hashed[i - p_val : i], U_prim_view
                                    ):
                                        is_left_maximal = False

                                if is_left_maximal:
                                    all_summarizable_runs.append(
                                        (
                                            [int(x) for x in U_prim_view],
                                            i,
                                            p_val,
                                            k_in_suffix,
                                        )
                                    )
            if pbar_phase1:
                pbar_phase1.update(1)
    finally:
        if pbar_phase1:
            pbar_phase1.close()

    if not all_summarizable_runs:
        return []

    SortedMasterRuns = sorted(all_summarizable_runs, key=lambda r: (-r[2], r[1]))
    return SortedMasterRuns


# --- Phase 2 Functions ---


def Phase2_IterativeSummarization(
    SortedMasterRuns: List[
        Tuple[List[int], int, int, int]
    ],  # U_list, i_star, p_star, k_star_max
    t_threshold: int,
) -> List[Dict[str, any]]:
    """
    Implements GLOTA Phase 2: Single-Pass Iterative Summarization using an Interval Tree
    and optimized logic based on properties of maximal runs.
    """
    CommittedSummariesTree = IntervalTree()
    # Stores dicts: {'s_c_start': ..., 's_c_end': ..., 's_c_details': CommitDetails}
    FinalCommittedSegmentsList: List[Dict[str, any]] = []

    if not SortedMasterRuns:
        print("Phase 2: Received empty SortedMasterRuns. No summarization to perform.")
        return []

    # print(f"Phase 2: Iterative summarization (t={t_threshold}, {len(SortedMasterRuns)} candidate runs)...")
    pbar_phase2: Optional[tqdm] = None
    try:
        pbar_phase2 = tqdm(
            total=len(SortedMasterRuns), unit="run", desc="Phase 2 Summarizing"
        )
        for U_star_list, i_star, p_star, k_star_max in SortedMasterRuns:
            potential_end_R_star_inclusive = i_star + (k_star_max * p_star) - 1

            # --- Rule A: Check for Disqualifying Start Overlap or Invalid Extension ---
            is_contained_in_parent = False
            is_invalidated_by_start_overlap = False
            parent_S_c_for_rule_C: Optional[Dict[str, any]] = None

            overlapping_committed_intervals_at_i_star = CommittedSummariesTree.at(
                i_star
            )

            # 1. Prioritize checking for any invalidating overlap:
            if overlapping_committed_intervals_at_i_star:
                for committed_interval in overlapping_committed_intervals_at_i_star:
                    s_c_end_inclusive = committed_interval.end - 1
                    if potential_end_R_star_inclusive > s_c_end_inclusive:
                        is_invalidated_by_start_overlap = True
                        break

            if is_invalidated_by_start_overlap:
                if pbar_phase2:
                    pbar_phase2.update(1)
                continue

            # 2. If not invalidated, and there were overlaps, R* is contained.
            #    Choose the containing parent S_c with the smallest primitive length 'p'.
            if overlapping_committed_intervals_at_i_star:
                is_contained_in_parent = True
                min_p_val_of_parent = float("inf")

                for committed_interval in overlapping_committed_intervals_at_i_star:
                    s_c_data = committed_interval.data
                    if s_c_data["p"] < min_p_val_of_parent:
                        min_p_val_of_parent = s_c_data["p"]
                        parent_S_c_for_rule_C = s_c_data

                if parent_S_c_for_rule_C is None:
                    # This should not be reached if overlapping_committed_intervals_at_i_star was non-empty.
                    # It implies all 'p' values were float('inf') or 'p' key was missing.
                    print(
                        f"Error: R* at {i_star} was overlapping but no valid parent details selected. Treating as not contained."
                    )
                    is_contained_in_parent = False  # Fallback: treat as not contained

            # --- Rule B: Determine Effective Run (Handle Suffix Truncation using optimized query) ---
            effective_run_start = i_star
            current_max_allowed_end_inclusive = potential_end_R_star_inclusive

            # Query for committed segments S_c that cover R*'s *potential end point*.
            segments_covering_R_star_potential_end = CommittedSummariesTree.at(
                potential_end_R_star_inclusive
            )

            min_s_c_start_that_truncates = float("inf")
            found_truncator_for_R_star = False

            if segments_covering_R_star_potential_end:
                for (
                    committed_interval_covering_end
                ) in segments_covering_R_star_potential_end:
                    s_c_start_of_potential_truncator = (
                        committed_interval_covering_end.begin
                    )
                    # Only care if S_c starts *after* R* starts AND *at or before* R*'s current (potential) end.
                    if (
                        effective_run_start
                        < s_c_start_of_potential_truncator
                        <= current_max_allowed_end_inclusive
                    ):
                        min_s_c_start_that_truncates = min(
                            min_s_c_start_that_truncates,
                            s_c_start_of_potential_truncator,
                        )
                        found_truncator_for_R_star = True

            if found_truncator_for_R_star:
                current_max_allowed_end_inclusive = min_s_c_start_that_truncates - 1

            if current_max_allowed_end_inclusive < effective_run_start:
                if pbar_phase2:
                    pbar_phase2.update(1)
                continue

            length_available = (
                current_max_allowed_end_inclusive - effective_run_start + 1
            )

            if length_available < (p_star * (t_threshold + 1)):
                if pbar_phase2:
                    pbar_phase2.update(1)
                continue

            effective_k = length_available // p_star
            effective_run_end_inclusive = (
                effective_run_start + (effective_k * p_star) - 1
            )
            # effective_k > t_threshold is guaranteed by the length_available check.

            # --- Rule C: Commit Run (hierarchical check and final commit) ---
            skip_R_star_altogether = False
            if is_contained_in_parent and parent_S_c_for_rule_C is not None:
                parent_p = parent_S_c_for_rule_C["p"]
                parent_original_start = parent_S_c_for_rule_C["i"]

                if parent_p == 0:
                    skip_R_star_altogether = True
                else:
                    rep_idx_within_parent = (
                        effective_run_start - parent_original_start
                    ) // parent_p  # This is start_rep_idx

                    if (
                        rep_idx_within_parent >= t_threshold
                    ):  # Standard Hierarchical Check
                        skip_R_star_altogether = True
                    else:
                        # R* is in a literal block of S_P.
                        # Check if R* itself spans across the boundaries of S_P's primitive blocks.
                        end_rep_idx_of_R_star = (
                            effective_run_end_inclusive - parent_original_start
                        ) // parent_p
                        if rep_idx_within_parent != end_rep_idx_of_R_star:
                            skip_R_star_altogether = True

            if skip_R_star_altogether:
                if pbar_phase2:
                    pbar_phase2.update(1)
                continue

            # If we reach here, R* is not skipped. Commit it.
            commit_details = {
                "U": U_star_list,
                "k": effective_k,
                "i": effective_run_start,
                "p": p_star,
                "effective_end": effective_run_end_inclusive,
            }

            CommittedSummariesTree.add(
                Interval(
                    effective_run_start, effective_run_end_inclusive + 1, commit_details
                )
            )

            FinalCommittedSegmentsList.append(
                {
                    "s_c_start": effective_run_start,
                    "s_c_end": effective_run_end_inclusive,
                    "s_c_details": commit_details,
                }
            )
            if pbar_phase2:
                pbar_phase2.update(1)
    finally:
        if pbar_phase2:
            pbar_phase2.close()

    if not FinalCommittedSegmentsList:
        print("Phase 2: No summaries were committed.")
        return []

    FinalCommittedSegmentsList.sort(key=lambda x: (x["s_c_start"], -x["s_c_end"]))

    # print(f"Phase 2: Committed {len(FinalCommittedSegmentsList)} summaries.")
    return FinalCommittedSegmentsList


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Globally-Optimal Log Truncation Algorithm (GLOTA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_file", type=str, required=True, help="Path to the raw input log file."
    )
    parser.add_argument(
        "--regex_file",
        type=str,
        help="Path to a file containing regex patterns for normalization. Each line: pattern|||replacement|||name",
    )
    parser.add_argument(
        "--skip_blank_normalized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip lines that become blank after normalization.",
    )
    parser.add_argument(
        "--debug_output_file",
        type=str,
        help="Base prefix for debug output files (e.g., 'debug_run'). Will create <prefix>.phase0.csv, <prefix>.phase1_runs.log, <prefix>.phase2_committed.log",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Character encoding for input and debug files.",
    )
    parser.add_argument(
        "--t_threshold",
        type=int,
        default=1,
        help="Summarization threshold t (run's k must be > t). Min 1.",
    )
    parser.add_argument(
        "--n_cap",
        type=int,
        default=0,
        help="Max primitive block length for Phase 1 run finding (0 for no cap).",
    )

    args = parser.parse_args()

    print("--- GLOTA Program Start ---")
    print("Parameters:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg:<25}: {value}")
    print("---")

    if args.t_threshold < 1:
        print("Error: --t_threshold must be >= 1. Program will exit.")
        return

    phase0_debug_file_path: Optional[str] = None
    phase1_runs_log_path: Optional[str] = None
    phase2_committed_log_path: Optional[str] = None

    if args.debug_output_file:
        base_debug_path = Path(args.debug_output_file)
        if base_debug_path.parent and not base_debug_path.parent.is_dir():
            try:
                base_debug_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(
                    f"Warning: Could not create debug output directory {base_debug_path.parent}: {e}"
                )

        phase0_debug_file_path = str(base_debug_path.with_suffix(".phase0.csv"))
        phase1_runs_log_path = str(base_debug_path.with_suffix(".phase1_runs.log"))
        phase2_committed_log_path = str(
            base_debug_path.with_suffix(".phase2_committed.log")
        )

    print("\n--- Running Phase 0: Preprocessing ---")
    regex_patterns: List[Tuple[str, str, str]] = []
    if args.regex_file:
        regex_patterns = load_regex_patterns_from_file(args.regex_file, args.encoding)
        print(f"Loaded {len(regex_patterns)} regex patterns.")

    phase0_result = Phase0_Preprocess(
        raw_log_file_path=args.log_file,
        regex_sub_pattern_list=regex_patterns,
        skip_blank_normalized_lines=args.skip_blank_normalized,
        debug_output_path=phase0_debug_file_path,
        character_encoding=args.encoding,
    )

    if phase0_result[0] is None:
        print("Phase 0 failed critically. Exiting.")
        return

    L_hashed: Optional[np.ndarray] = phase0_result[0]
    id_to_normalized_map: Optional[List[str]] = phase0_result[1]
    original_lines_map: Optional[List[str]] = phase0_result[2]

    print(f"Phase 0 completed.")
    if phase0_debug_file_path:
        debug_file_p0 = Path(phase0_debug_file_path)
        if debug_file_p0.exists() and debug_file_p0.stat().st_size > 0:
            print(f"Phase 0 debug output written to: {phase0_debug_file_path}")
        else:
            print(
                f"Phase 0 debug output file was specified ({phase0_debug_file_path}) but possibly not created or is empty."
            )

    SortedMasterRuns: List[Tuple[List[int], int, int, int]] = []
    if L_hashed is None or L_hashed.size == 0:
        print("\nNo data from Phase 0 to process for Phase 1.")
    elif L_hashed is not None:
        print("\n--- Running Phase 1: Finding Summarizable Runs ---")
        SortedMasterRuns = Phase1_FindAndSort_SummarizableRuns(
            L_hashed, args.t_threshold, args.n_cap
        )
        print(
            f"Phase 1 completed. Found {len(SortedMasterRuns)} summarizable master runs."
        )

        if SortedMasterRuns and phase1_runs_log_path:
            try:
                with open(phase1_runs_log_path, "w", encoding=args.encoding) as f_runs:
                    f_runs.write(
                        f"# Phase 1 SortedMasterRuns (Total: {len(SortedMasterRuns)})\n"
                    )
                    f_runs.write(
                        f"# Format: U_list (Python list of int IDs), i_start_index, p_length, k_repetitions\n"
                    )
                    for run_tuple in SortedMasterRuns:
                        f_runs.write(
                            f"{run_tuple[0]},{run_tuple[1]},{run_tuple[2]},{run_tuple[3]}\n"
                        )
                print(f"Phase 1 SortedMasterRuns saved to: {phase1_runs_log_path}")
            except IOError as e:
                print(
                    f"Warning: Could not write Phase 1 runs log to {phase1_runs_log_path}: {e}"
                )

        if SortedMasterRuns:
            print("First few master runs (U limited to 5 elements for display):")
            for r_idx, r_val in enumerate(
                SortedMasterRuns[: min(5, len(SortedMasterRuns))]
            ):
                u_list_display = r_val[0]
                u_str_elements = [str(x) for x in u_list_display[:5]]
                u_str = (
                    "["
                    + ", ".join(u_str_elements)
                    + ("" if len(u_list_display) <= 5 else ", ...")
                    + "]"
                )
                print(
                    f"  {r_idx+1}. U={u_str}, i={r_val[1]}, p={r_val[2]}, k={r_val[3]}"
                )

    CommittedSummaries: List[Dict[str, any]] = []
    if not SortedMasterRuns:
        print("\nNo runs from Phase 1 to process in Phase 2.")
    elif L_hashed is None:
        print("\nError: L_hashed is None from Phase 0, cannot proceed to Phase 2.")
    else:
        print("\n--- Running Phase 2: Iterative Summarization ---")
        CommittedSummaries = Phase2_IterativeSummarization(
            SortedMasterRuns, args.t_threshold
        )
        print(f"Phase 2 completed. Committed {len(CommittedSummaries)} summaries.")

        if CommittedSummaries and phase2_committed_log_path:
            try:
                with open(
                    phase2_committed_log_path, "w", encoding=args.encoding
                ) as f_commit:
                    f_commit.write(
                        f"# Phase 2 CommittedSummaries (Total: {len(CommittedSummaries)})\n"
                    )
                    f_commit.write(
                        f"# Format: s_c_start, s_c_end, U_list (IDs), k_effective, p_length, i_original_U_start\n"
                    )
                    for seg in CommittedSummaries:
                        details = seg["s_c_details"]
                        u_list_str = str(details["U"])
                        f_commit.write(
                            f"{seg['s_c_start']},{seg['s_c_end']},{u_list_str},{details['k']},{details['p']},{details['i']}\n"
                        )
                print(
                    f"Phase 2 CommittedSummaries saved to: {phase2_committed_log_path}"
                )
            except IOError as e:
                print(
                    f"Warning: Could not write Phase 2 committed log to {phase2_committed_log_path}: {e}"
                )

        if CommittedSummaries:
            print("First few committed summaries:")
            for s_idx, s_val in enumerate(
                CommittedSummaries[: min(5, len(CommittedSummaries))]
            ):
                details = s_val["s_c_details"]
                u_list_display = details["U"]
                u_str_elements = [str(x) for x in u_list_display[:5]]
                u_str = (
                    "["
                    + ", ".join(u_str_elements)
                    + ("" if len(u_list_display) <= 5 else ", ...")
                    + "]"
                )
                print(
                    f"  {s_idx+1}. EffectiveRange=[{s_val['s_c_start']}-{s_val['s_c_end']}], Orig_i={details['i']}, U={u_str}, k={details['k']}, p={details['p']}"
                )

    print("\n--- Phase 3: Final Output Generation (Not Implemented) ---")
    # if CommittedSummaries and L_hashed is not None and id_to_normalized_map is not None and original_lines_map is not None:
    # Phase3_Function(...)

    print("\n--- GLOTA Program End ---")


if __name__ == "__main__":
    main()
