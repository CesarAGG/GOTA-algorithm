# main_glota.py

import re
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import List, Set, TextIO, Tuple, Dict, Optional, Union, Any, Literal
from numba import jit, types as nb_types
from intervaltree import Interval, IntervalTree
import sys


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
                            "Error: Exceeded maximum unique log entries (65535) for uint16. Stopping."
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
                                if (
                                    i > 0
                                    and p_val <= i
                                    and np.array_equal(
                                        log_hashed[i - p_val : i], U_prim_view
                                    )
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

                    if rep_idx_within_parent == 0:
                        # R* starts in the VERY FIRST (0-th) block of the parent's primitive U.
                        # Now, check if R* straddles blocks of the parent U.
                        end_rep_idx_of_R_star_in_parent_frame = (
                            effective_run_end_inclusive - parent_original_start
                        ) // parent_p
                        if (
                            rep_idx_within_parent
                            != end_rep_idx_of_R_star_in_parent_frame
                        ):
                            skip_R_star_altogether = True
                    else:
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


# --- Phase 3 Functions ---


def shift_indices_in_rendered_list(
    rendered_list: List[Any], shift_amount: int
) -> List[Any]:
    new_list = []
    if not isinstance(rendered_list, list):
        # Fallback for unexpected input, though ideally rendered_list is always structured
        return (
            [
                item + shift_amount if isinstance(item, int) else item
                for item in rendered_list
            ]
            if isinstance(rendered_list, list)
            else rendered_list
        )

    for item in rendered_list:
        if not isinstance(item, tuple) or not item:
            if (
                isinstance(item, int) and item_type == "LITERAL"
            ):  # Should be ('LITERAL', idx)
                new_list.append(
                    ("LITERAL", item + shift_amount)
                )  # Assume int was an index for a literal
            else:
                new_list.append(item)  # Pass through unknown items
            continue

        item_type = item[0]
        if item_type == "LITERAL" and len(item) == 2:
            tag, original_log_idx = item
            new_list.append((tag, original_log_idx + shift_amount))
        elif item_type == "SUMMARY_R" and len(item) == 5:
            tag, U_rendered_child, k_val, i_val, p_val = item
            shifted_U_rendered_child = shift_indices_in_rendered_list(
                U_rendered_child, shift_amount
            )
            new_list.append(
                (tag, shifted_U_rendered_child, k_val, i_val + shift_amount, p_val)
            )
        else:
            new_list.append(item)
    return new_list


# Helper function to assemble the output for a given S_next summary
def _assemble_output_for_S_next(
    output_list_to_extend: List[Any],
    rendered_U_canonical: List[Any],
    raw_U_s_ids_for_fallback: List[int],
    k_s: int,  # Guaranteed k_s > t_global
    p_s: int,
    original_i_of_S_next: int,  # Start index of S_next in L_hashed
    t_global: int,
    user_t_start_display_count: Optional[int],
    user_t_end_display_count: Optional[int],
):
    if p_s == 0:
        return

    # Determine U representation for the summary message (always from rendered_U_canonical)
    U_for_summary_message = rendered_U_canonical
    if (
        not rendered_U_canonical
    ):  # Fallback if the rendering of the first U was empty (e.g. p_s=0 or content was all filtered)
        U_for_summary_message = [
            ("LITERAL", original_i_of_S_next + i)
            for i in range(len(raw_U_s_ids_for_fallback))
        ]

    # Determine actual t_start and t_end for display
    actual_t_start: int
    actual_t_end: int
    if user_t_start_display_count is not None and user_t_end_display_count is not None:
        if (
            user_t_start_display_count + user_t_end_display_count == t_global
            and user_t_start_display_count >= 0
            and user_t_end_display_count >= 0
        ):
            actual_t_start = user_t_start_display_count
            actual_t_end = user_t_end_display_count
        else:  # Invalid user input, revert to default
            print(
                f"Warning: Custom t_start/t_end invalid for t_global={t_global}. Using default for S_next (orig_i={original_i_of_S_next})."
            )
            actual_t_start = t_global // 2 if t_global >= 2 else t_global
            actual_t_end = t_global - actual_t_start
    elif user_t_start_display_count is not None:
        actual_t_start = max(0, min(user_t_start_display_count, t_global))
        actual_t_end = t_global - actual_t_start
    elif user_t_end_display_count is not None:
        actual_t_end = max(0, min(user_t_end_display_count, t_global))
        actual_t_start = t_global - actual_t_end
    else:  # Default split
        actual_t_start = t_global // 2 if t_global >= 2 else t_global
        actual_t_end = t_global - actual_t_start

    # Ensure consistency after derivations
    actual_t_start = max(0, actual_t_start)
    actual_t_end = max(0, actual_t_end)
    if actual_t_start + actual_t_end > t_global:
        actual_t_end = t_global - actual_t_start
    actual_t_end = max(0, actual_t_end)  # Make sure t_end is not negative

    # 1. Output first actual_t_start literals
    for rep_num in range(actual_t_start):
        if rep_num == 0:
            output_list_to_extend.extend(rendered_U_canonical)
        else:
            shift = rep_num * p_s
            output_list_to_extend.extend(
                shift_indices_in_rendered_list(rendered_U_canonical, shift)
            )

    # 2. Output the summary message (k_s > t_global is guaranteed for S_next)
    num_summarized_reps = k_s - t_global
    summary_msg = (
        "SUMMARY_R",
        U_for_summary_message,
        num_summarized_reps,
        original_i_of_S_next,
        p_s,
    )
    output_list_to_extend.append(summary_msg)

    # 3. Output last actual_t_end literals
    for i in range(actual_t_end):
        rep_num = (t_global - actual_t_end) + i
        shift = rep_num * p_s
        output_list_to_extend.extend(
            shift_indices_in_rendered_list(rendered_U_canonical, shift)
        )


# Recursive rendering function
def _render_hashed_segment_recursive(
    log_L_hashed: np.ndarray,
    current_segment_start: int,
    current_segment_end: int,
    all_committed_summaries: List[Dict[str, any]],
    t_global: int,
    initial_summary_search_idx: int,
    user_t_start_display_count: Optional[int],
    user_t_end_display_count: Optional[int],
) -> List[
    Union[Tuple[str, int], Tuple[str, List[Any], int, int, int]]
]:  # Literal is ('LITERAL', idx)
    output_items_for_this_segment: List[
        Union[Tuple[str, int], Tuple[str, List[Any], int, int, int]]
    ] = []
    current_pos = current_segment_start
    active_summary_candidate_idx = initial_summary_search_idx

    while current_pos <= current_segment_end:
        S_next = None
        s_next_found_at_list_idx = -1

        temp_search_idx = active_summary_candidate_idx
        while temp_search_idx < len(all_committed_summaries):
            s_candidate = all_committed_summaries[temp_search_idx]
            if s_candidate["s_c_start"] < current_pos:
                active_summary_candidate_idx = temp_search_idx + 1
                temp_search_idx += 1
            elif s_candidate["s_c_start"] == current_pos:
                # Phase 2 guarantees s_candidate['s_c_end'] fits within the segment
                # if this is a recursive call for a parent's literal block due to the straddling check.
                S_next = s_candidate
                s_next_found_at_list_idx = temp_search_idx
                break
            else:  # s_candidate['s_c_start'] > current_pos
                break

        if S_next:
            s_details = S_next["s_c_details"]
            U_s_ids_raw: List[int] = s_details["U"]
            k_s: int = s_details["k"]
            p_s: int = s_details["p"]
            original_i_of_S_next: int = s_details["i"]

            rendered_U_s_rep_0: List[Any] = []
            if p_s > 0:
                first_rep_span_start = S_next["s_c_start"]
                first_rep_span_end = first_rep_span_start + p_s - 1

                rendered_U_s_rep_0 = _render_hashed_segment_recursive(
                    log_L_hashed,
                    first_rep_span_start,
                    first_rep_span_end,
                    all_committed_summaries,
                    t_global,
                    s_next_found_at_list_idx + 1,
                    user_t_start_display_count,
                    user_t_end_display_count,
                )

            _assemble_output_for_S_next(
                output_items_for_this_segment,
                rendered_U_s_rep_0,
                U_s_ids_raw,
                k_s,
                p_s,
                original_i_of_S_next,
                t_global,
                user_t_start_display_count,
                user_t_end_display_count,
            )

            current_pos = S_next["s_c_end"] + 1
            active_summary_candidate_idx = s_next_found_at_list_idx + 1
        else:
            output_items_for_this_segment.append(
                ("LITERAL", current_pos)
            )  # Store as ('LITERAL', index)
            current_pos += 1
    return output_items_for_this_segment


# Main entry point for Phase 3
def Phase3_GenerateRecursiveHashedOutput(
    L_hashed: np.ndarray,
    CommittedSummaries: List[Dict[str, any]],
    t_threshold: int,
    user_t_start_display_count: Optional[int] = None,
    user_t_end_display_count: Optional[int] = None,
) -> List[Union[Tuple[str, int], Tuple[str, List[Any], int, int, int]]]:  # Output items
    N = len(L_hashed)
    if N == 0:
        print("Phase 3: Received empty L_hashed. No output.")
        return []

    if not CommittedSummaries:
        print("Phase 3: No committed summaries from Phase 2. Output will be literals.")
        return [("LITERAL", i) for i in range(N)]

    # print(f"Phase 3: Generating recursive hashed output (N={N}, {len(CommittedSummaries)} committed summaries, t={t_threshold}, t_start_user={user_t_start_display_count}, t_end_user={user_t_end_display_count})...")

    result = _render_hashed_segment_recursive(
        log_L_hashed=L_hashed,
        current_segment_start=0,
        current_segment_end=N - 1,
        all_committed_summaries=CommittedSummaries,
        t_global=t_threshold,
        initial_summary_search_idx=0,
        user_t_start_display_count=user_t_start_display_count,
        user_t_end_display_count=user_t_end_display_count,
    )
    # print(f"Phase 3: Recursive rendering complete. Generated {len(result)} items.")
    return result


# --- Phase 4 Functions ---


# This helper function will recursively build the string representation of a U-block
def _format_U_block_to_string(
    U_items_list: List[Any],  # This is a Rendered_U_canonical list
    L_hashed_ref: np.ndarray,
    id_to_norm_map: List[str],
    t_global_val: int,  # args.t_threshold, for total k of nested summaries
    current_nesting_level: int,  # For <, <<, <<<, etc.
    # Global display style parameters (passed down for consistent nested summary phrasing)
    glob_display_t_start: int,
    glob_display_t_end: int,
    base_separator: str = " | ",
) -> str:
    display_parts: List[str] = []
    if not isinstance(U_items_list, list):
        return f"(Malformed U-block: {str(U_items_list)[:50]})"

    # Determine the separator for the current level of nesting
    current_level_separator = " " + ("|" * (current_nesting_level + 1)) + " "

    for sub_item in U_items_list:
        if not isinstance(sub_item, tuple) or not sub_item:
            display_parts.append(f"[corrupted_U_item: {repr(sub_item)}]")
            continue

        sub_item_type = sub_item[0]

        if sub_item_type == "LITERAL" and len(sub_item) == 2:
            _sub_tag, sub_original_log_idx = sub_item
            line_for_U = f"[err_idx:{sub_original_log_idx}]"
            if 0 <= sub_original_log_idx < len(L_hashed_ref):
                sub_norm_id = L_hashed_ref[sub_original_log_idx]
                if 0 <= sub_norm_id < len(id_to_norm_map):
                    line_for_U = id_to_norm_map[sub_norm_id]  # NORMALIZED
                else:
                    line_for_U = f"[err_id:{sub_norm_id}]"
            display_parts.append(line_for_U)

        elif sub_item_type == "SUMMARY_R" and len(sub_item) == 5:
            _s_tag_r, s_Rendered_U_grandchild, s_num_add_reps_r, _s_i_r, _s_p_r = (
                sub_item
            )

            Formatted_nested_U_string = _format_U_block_to_string(
                s_Rendered_U_grandchild,
                L_hashed_ref,
                id_to_norm_map,
                t_global_val,
                current_nesting_level + 1,
                glob_display_t_start,
                glob_display_t_end,  # Pass these down
                base_separator,
            )
            if not Formatted_nested_U_string:
                Formatted_nested_U_string = (
                    f"pattern_of_{_s_p_r}_entries" if _s_p_r > 0 else "empty_pattern"
                )

            opening_chevrons = "<" * (current_nesting_level + 1)
            closing_chevrons = ">" * (current_nesting_level + 1)

            # Phrasing for the nested summary message part, consistent with global style
            nested_summary_text_core: str
            if t_global_val > 0:
                if glob_display_t_start == 0:
                    nested_summary_text_core = f"First {s_num_add_reps_r} reps of '{Formatted_nested_U_string}' summarized; remaining {t_global_val} instances follow"
                elif glob_display_t_end == 0:
                    nested_summary_text_core = f"'{Formatted_nested_U_string}' rep {s_num_add_reps_r} more times"
                else:
                    nested_summary_text_core = f"'{Formatted_nested_U_string}' rep {s_num_add_reps_r} times in between"
            else:  # t_global_val == 0 (defensive, as t_threshold >= 1)
                nested_summary_text_core = (
                    f"'{Formatted_nested_U_string}' rep {s_num_add_reps_r} more times"
                )

            display_parts.append(
                f"{opening_chevrons}{nested_summary_text_core}{closing_chevrons}"
            )

    return current_level_separator.join(display_parts)


# Main recursive printing function for Phase 4
def _dehash_and_print_phase4_items(
    items_list: List[Any],
    id_to_norm_map: List[str],
    orig_lines_list: List[str],
    L_hashed_ref: np.ndarray,
    t_global_val: int,  # This is args.t_threshold
    glob_display_t_start: int,  # Calculated in main from t_global and user overrides
    glob_display_t_end: int,  # Calculated in main
    indent_level: int,
    out_f: TextIO,
):
    current_indent_str = "  " * indent_level
    for item_idx, item in enumerate(items_list):
        if not isinstance(item, tuple) or not item:
            out_f.write(
                f"{current_indent_str}[Warning: Corrupted item at index {item_idx}: {repr(item)}]\n"
            )
            continue

        item_type = item[0]

        if item_type == "LITERAL" and len(item) == 2:
            _tag, original_log_idx = item
            line_to_print = (
                f"[Error: Original log index {original_log_idx} out of bounds]"
            )
            # Top-level literals ALWAYS use original_lines_by_position_list
            if 0 <= original_log_idx < len(orig_lines_list):
                line_to_print = orig_lines_list[original_log_idx]
            out_f.write(f"{current_indent_str}{line_to_print}\n")

        elif item_type == "SUMMARY_R" and len(item) == 5:
            (
                _tag,
                Rendered_U_canonical,
                num_additional_reps,
                _original_i_of_run,
                _p_of_run,
            ) = item

            Formatted_U_string = _format_U_block_to_string(
                Rendered_U_canonical,
                L_hashed_ref,
                id_to_norm_map,
                t_global_val,
                0,  # Initial nesting level for this U string representation
                glob_display_t_start,
                glob_display_t_end,  # Pass global display hints
            )
            if not Formatted_U_string:
                Formatted_U_string = (
                    "an empty pattern"
                    if _p_of_run == 0
                    else f"a pattern of {_p_of_run} entries"
                )

            summary_message: str
            if glob_display_t_start == 0:
                summary_message = (
                    f'(First {num_additional_reps} repetitions of sequence "{Formatted_U_string}" summarized; '
                    f"remaining {t_global_val} instances follow)"
                )
            elif glob_display_t_end == 0:
                summary_message = f'(Sequence "{Formatted_U_string}" repeated {num_additional_reps} more times)'
            else:
                summary_message = f'(Sequence "{Formatted_U_string}" repeated {num_additional_reps} times in between)'

            out_f.write(f"{current_indent_str}{summary_message}\n")
        else:
            out_f.write(
                f"{current_indent_str}[Warning: Unknown item type '{item_type}' at index {item_idx}]\n"
            )


# Main entry for Phase 4
def Phase4_DehashAndPrint(
    SummarizedLog_HashedItems: List[Any],
    id_to_normalized_map: List[str],
    original_lines_by_position_list: List[str],
    L_hashed: np.ndarray,
    t_global_for_display: int,  # This is args.t_threshold
    # User display preferences for t-block, derived in main:
    display_actual_t_start: int,
    display_actual_t_end: int,
    output_file_path: Optional[str],
    character_encoding: str,
):
    output_target_handle: TextIO
    is_stdout = False
    if output_file_path:
        try:
            output_target_handle = open(
                output_file_path, "w", encoding=character_encoding
            )
        except IOError as e:
            print(
                f"Error opening output file {output_file_path}: {e}. Printing to console instead."
            )
            output_target_handle = sys.stdout
            is_stdout = True
    else:
        output_target_handle = sys.stdout
        is_stdout = True
        if SummarizedLog_HashedItems:  # Only print header if there's content
            print("\n--- Phase 4: Final De-hashed Output ---")

    try:
        if not SummarizedLog_HashedItems:
            message_for_empty = "(Log was empty or fully summarized into no items)\n"
            output_target_handle.write(message_for_empty)
            if is_stdout and not output_file_path:
                pass  # Avoid double printing if already on console
            elif is_stdout:
                print(message_for_empty.strip())

        else:
            _dehash_and_print_phase4_items(
                SummarizedLog_HashedItems,
                id_to_normalized_map,
                original_lines_by_position_list,
                L_hashed,
                t_global_for_display,
                display_actual_t_start,  # Pass calculated display counts
                display_actual_t_end,  # Pass calculated display counts
                0,  # initial indent_level
                output_target_handle,
            )
    except Exception as e:
        print(f"Error during Phase 4 output generation: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
    finally:
        if not is_stdout and output_target_handle:
            try:
                output_target_handle.close()
            except Exception as e:
                print(f"Error closing output file: {e}")
        elif is_stdout and SummarizedLog_HashedItems:
            print("--- End of Output ---")


# --- Main Function (Corrected call to Phase4_DehashAndPrint) ---
def main():
    parser = argparse.ArgumentParser(
        description="Globally-Optimal Log Truncation Algorithm (GLOTA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Phase 0 Args
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
        help="Base prefix for debug output files (e.g., 'myrun'). Creates <prefix>.phaseN.ext.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Character encoding for input and debug files.",
    )

    # Phase 1 Args
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

    # Phase 3 Display Args (affecting literal rendering order around summaries)
    parser.add_argument(
        "--t_start_display",
        type=int,
        default=None,
        help="For t>=2, number of literal reps at start of a t-block. Default: floor(t/2).",
    )
    parser.add_argument(
        "--t_end_display",
        type=int,
        default=None,
        help="For t>=2, number of literal reps at end of a t-block. Default: ceil(t/2).",
    )

    # Phase 4 (Output) Args
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to write the final de-hashed summarized log output. If not provided, prints to console.",
    )
    # --show_original_literals flag has been REMOVED.

    args = parser.parse_args()

    print("--- GLOTA Program Start ---")
    print("Parameters:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg:<25}: {value}")
    print("---")

    if args.t_threshold < 1:
        print("Error: --t_threshold must be >= 1. Program will exit.")
        return

    # Validate t_start_display and t_end_display against t_threshold
    if args.t_start_display is not None and args.t_start_display < 0:
        print("Error: --t_start_display cannot be negative. Exiting.")
        return
    if args.t_end_display is not None and args.t_end_display < 0:
        print("Error: --t_end_display cannot be negative. Exiting.")
        return

    if (
        args.t_start_display is not None
        and args.t_end_display is not None
        and args.t_start_display + args.t_end_display != args.t_threshold
    ):
        print(
            f"Error: User-specified --t_start_display ({args.t_start_display}) + --t_end_display ({args.t_end_display}) "
            f"must sum to --t_threshold ({args.t_threshold}). Exiting."
        )
        return
    if args.t_start_display is not None and args.t_start_display > args.t_threshold:
        print(
            f"Error: --t_start_display ({args.t_start_display}) cannot exceed --t_threshold ({args.t_threshold}). Exiting."
        )
        return
    if args.t_end_display is not None and args.t_end_display > args.t_threshold:
        print(
            f"Error: --t_end_display ({args.t_end_display}) cannot exceed --t_threshold ({args.t_threshold}). Exiting."
        )
        return

    # Determine final t_start and t_end for display purposes, to be used by Phase 4
    cfg_t_global = args.t_threshold
    cfg_user_t_start = args.t_start_display
    cfg_user_t_end = args.t_end_display

    display_actual_t_start: int
    display_actual_t_end: int

    if cfg_user_t_start is not None and cfg_user_t_end is not None:
        display_actual_t_start = cfg_user_t_start
        display_actual_t_end = cfg_user_t_end
    elif cfg_user_t_start is not None:
        display_actual_t_start = max(0, min(cfg_user_t_start, cfg_t_global))
        display_actual_t_end = cfg_t_global - display_actual_t_start
    elif cfg_user_t_end is not None:
        display_actual_t_end = max(0, min(cfg_user_t_end, cfg_t_global))
        display_actual_t_start = cfg_t_global - display_actual_t_end
    else:
        display_actual_t_start = (
            cfg_t_global // 2 if cfg_t_global >= 2 else cfg_t_global
        )
        display_actual_t_end = cfg_t_global - display_actual_t_start

    display_actual_t_start = max(0, display_actual_t_start)
    display_actual_t_end = max(0, display_actual_t_end)
    if display_actual_t_start + display_actual_t_end > cfg_t_global:
        if cfg_user_t_start is not None:
            display_actual_t_end = cfg_t_global - display_actual_t_start
        elif cfg_user_t_end is not None:
            display_actual_t_start = cfg_t_global - display_actual_t_end
        else:
            display_actual_t_start = (
                cfg_t_global // 2 if cfg_t_global >= 2 else cfg_t_global
            )
            display_actual_t_end = cfg_t_global - display_actual_t_start
    display_actual_t_end = max(0, display_actual_t_end)
    display_actual_t_start = cfg_t_global - display_actual_t_end

    # Setup Debug File Paths
    (
        phase0_debug_file_path,
        phase1_runs_log_path,
        phase2_committed_log_path,
        phase3_hashed_output_log_path,
    ) = (None, None, None, None)
    if args.debug_output_file:
        base_debug_path = Path(args.debug_output_file)
        if base_debug_path.parent and not base_debug_path.parent.is_dir():
            try:
                base_debug_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(
                    f"Warning: Could not create debug directory {base_debug_path.parent}: {e}"
                )
        phase0_debug_file_path = str(base_debug_path.with_suffix(".phase0.csv"))
        phase1_runs_log_path = str(base_debug_path.with_suffix(".phase1_runs.log"))
        phase2_committed_log_path = str(
            base_debug_path.with_suffix(".phase2_committed.log")
        )
        phase3_hashed_output_log_path = str(
            base_debug_path.with_suffix(".phase3_hashed_output.log")
        )

    # Initialize optional variables that Phases return
    L_hashed: Optional[np.ndarray] = None
    id_to_normalized_map: Optional[List[str]] = None
    original_lines_map: Optional[List[str]] = None
    SortedMasterRuns: List[Tuple[List[int], int, int, int]] = []
    CommittedSummaries: List[Dict[str, any]] = []
    SummarizedLog_Hashed: List[Any] = []

    # --- Phase 0 ---
    print("\n--- Running Phase 0: Preprocessing ---")
    regex_patterns: List[Tuple[str, str, str]] = []
    if args.regex_file:
        regex_patterns = load_regex_patterns_from_file(args.regex_file, args.encoding)

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
    L_hashed, id_to_normalized_map, original_lines_map = phase0_result

    print("Phase 0 completed.")
    if phase0_debug_file_path:
        debug_file_p0 = Path(phase0_debug_file_path)
        if debug_file_p0.exists() and debug_file_p0.stat().st_size > 0:
            print(f"Phase 0 debug output: {phase0_debug_file_path}")
        else:
            print(
                f"Phase 0 debug file specified ({phase0_debug_file_path}) but possibly not created/empty."
            )

    # --- Phase 1 ---
    if L_hashed is None or L_hashed.size == 0:
        print("\nNo data from Phase 0 for Phase 1.")
    else:
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
                        f"# Phase 1 SortedMasterRuns (Total: {len(SortedMasterRuns)})\n# Format: U_list,i,p,k\n"
                    )
                    for r_tup in SortedMasterRuns:
                        f_runs.write(f"{r_tup[0]},{r_tup[1]},{r_tup[2]},{r_tup[3]}\n")
                print(f"Phase 1 SortedMasterRuns saved to: {phase1_runs_log_path}")
            except IOError as e:
                print(f"Warning: Could not write Phase 1 runs log: {e}")
        if SortedMasterRuns:
            print("First few master runs (U limited to 5 elements for display):")
            for r_idx, r_val in enumerate(
                SortedMasterRuns[: min(5, len(SortedMasterRuns))]
            ):
                u_s = str(r_val[0][:5]) + ("..." if len(r_val[0]) > 5 else "")
                print(f"  {r_idx+1}. U={u_s}, i={r_val[1]}, p={r_val[2]}, k={r_val[3]}")

    # --- Phase 2 ---
    if not SortedMasterRuns:
        print("\nNo runs from Phase 1 for Phase 2.")
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
                        f"# Phase 2 CommittedSummaries (Total: {len(CommittedSummaries)})\n# Format: s_start,s_end,U_list,k_eff,p,orig_i_of_U\n"
                    )
                    for seg in CommittedSummaries:
                        d = seg["s_c_details"]
                        f_commit.write(
                            f"{seg['s_c_start']},{seg['s_c_end']},{d['U']},{d['k']},{d['p']},{d['i']}\n"
                        )
                print(
                    f"Phase 2 CommittedSummaries saved to: {phase2_committed_log_path}"
                )
            except IOError as e:
                print(f"Warning: Could not write Phase 2 committed log: {e}")
        if CommittedSummaries:
            print(
                "First few committed summaries (U limited to 5 elements for display):"
            )
            for s_idx, s_val in enumerate(
                CommittedSummaries[: min(5, len(CommittedSummaries))]
            ):
                d = s_val["s_c_details"]
                u_s = str(d["U"][:5]) + ("..." if len(d["U"]) > 5 else "")
                print(
                    f"  {s_idx+1}. EffRange=[{s_val['s_c_start']}-{s_val['s_c_end']}], Orig_i={d['i']}, U={u_s}, k={d['k']}, p={d['p']}"
                )

    # --- Phase 3 ---
    if L_hashed is None:
        print("\nError: L_hashed is None from Phase 0, cannot proceed to Phase 3.")
    elif L_hashed.size == 0 and not CommittedSummaries:
        print("\nNo data or summaries for Phase 3.")
    else:
        print("\n--- Running Phase 3: Generating Recursive Hashed Output ---")
        SummarizedLog_Hashed = Phase3_GenerateRecursiveHashedOutput(
            L_hashed,
            CommittedSummaries,
            args.t_threshold,
            args.t_start_display,
            args.t_end_display,
        )
        print(
            f"Phase 3 completed. Generated {len(SummarizedLog_Hashed)} output items (hashed)."
        )
        if SummarizedLog_Hashed and phase3_hashed_output_log_path:
            try:
                with open(
                    phase3_hashed_output_log_path, "w", encoding=args.encoding
                ) as f_ph3:
                    f_ph3.write(
                        f"# Phase 3 Hashed Output (Total: {len(SummarizedLog_Hashed)} items)\n"
                    )
                    for item in SummarizedLog_Hashed:
                        f_ph3.write(f"{repr(item)}\n")
                print(
                    f"Phase 3 hashed output saved to: {phase3_hashed_output_log_path}"
                )
            except IOError as e:
                print(f"Warning: Could not write Phase 3 hashed log: {e}")
        if SummarizedLog_Hashed:
            print(
                "First few items of Phase 3 hashed output (repr limited to 100 chars):"
            )
            for item_idx, item_val in enumerate(
                SummarizedLog_Hashed[: min(10, len(SummarizedLog_Hashed))]
            ):
                item_repr = repr(item_val)
                print(
                    f"  Item {item_idx+1}: {item_repr[:100]}"
                    + ("..." if len(item_repr) > 100 else "")
                )

    # --- Phase 4: De-Hashing and Final Output ---
    print("\n--- Running Phase 4: De-Hashing and Final Output Generation ---")
    if (
        SummarizedLog_Hashed
        and id_to_normalized_map is not None
        and original_lines_map is not None
        and L_hashed is not None
    ):
        Phase4_DehashAndPrint(
            SummarizedLog_Hashed,
            id_to_normalized_map,
            original_lines_map,
            L_hashed,
            args.t_threshold,  # This is t_global_for_display
            # Pass the calculated display_actual_t_start and display_actual_t_end
            display_actual_t_start,
            display_actual_t_end,
            args.output_file,
            args.encoding,
        )
    else:
        print(
            "Phase 4 skipped due to missing inputs from prior phases (check if L_hashed or maps are None, or SummarizedLog_Hashed is empty)."
        )

    print("\n--- GLOTA Program End ---")


if __name__ == "__main__":
    main()
