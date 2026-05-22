import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

try:
    import RNA
except ImportError:  # pragma: no cover
    RNA = None


logger = logging.getLogger(__name__)


STRUCTURE_VIEW_ORDER = [
    "Struct-PB",
    "Struct-PC",
    "Struct-PE",
    "Struct-PBC",
    "Struct-PBE",
    "Struct-PCE",
    "Struct-PBCE",
]

STRUCTURE_VIEW_INPUT_DIMS = {
    "Struct-PB": 2,
    "Struct-PC": 7,
    "Struct-PE": 65,
    "Struct-PBC": 8,
    "Struct-PBE": 66,
    "Struct-PCE": 71,
    "Struct-PBCE": 72,
}

STRUCTURE_CONTEXT_ORDER = [
    "Stem",
    "Hairpin Loop",
    "Bulge Loop",
    "Internal Loop",
    "Multibranch Loop",
    "Pseudoknot",
]

PRIMARY_OPEN = "("
PRIMARY_CLOSE = ")"
SECONDARY_BRACKET_PAIRS = {
    "[": "]",
    "{": "}",
    "<": ">",
}

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    SECONDARY_BRACKET_PAIRS[letter] = letter.lower()


def sanitize_rna_sequence(seq: str) -> str:
    seq = (seq or "").upper().replace("T", "U")
    seq = re.sub(r"[^ACGUN]", "N", seq)
    return seq or "N"


def truncate_for_structure(seq: str, max_len: int = 3000) -> str:
    seq = sanitize_rna_sequence(seq)
    return seq[:max_len]


def _pair_probability_matrix_from_plist(length: int, plist: Sequence) -> torch.Tensor:
    matrix = np.zeros((length, length), dtype=np.float32)
    for entry in plist:
        try:
            i = int(entry.i) - 1
            j = int(entry.j) - 1
            probability = float(entry.p)
        except Exception:
            continue

        if 0 <= i < length and 0 <= j < length:
            matrix[i, j] = max(matrix[i, j], probability)
            matrix[j, i] = max(matrix[j, i], probability)

    return torch.tensor(matrix, dtype=torch.float32)


def _pair_probability_matrix_from_pair_map(pair_map: Dict[int, int], length: int) -> torch.Tensor:
    matrix = np.zeros((length, length), dtype=np.float32)
    for left, right in pair_map.items():
        if left < right and 0 <= left < length and 0 <= right < length:
            matrix[left, right] = 1.0
            matrix[right, left] = 1.0

    return torch.tensor(matrix, dtype=torch.float32)


def _normalize_dot_bracket_string(dbn_string: Optional[str], expected_length: int) -> str:
    if not dbn_string:
        return ""

    lines = [line.strip() for line in str(dbn_string).splitlines() if line.strip()]
    candidate = lines[-1] if lines else ""

    if lines and lines[0].startswith(">") and len(lines) >= 3:
        candidate = lines[-1]

    candidate = re.sub(r"\s+", "", candidate)
    if expected_length > 0:
        candidate = candidate[:expected_length]
        if len(candidate) < expected_length:
            candidate = candidate.ljust(expected_length, ".")

    return candidate


def run_rnafold(seq_struct: str, probability_cutoff: float = 1e-6) -> Tuple[str, torch.Tensor]:
    """Run ViennaRNA RNAfold and return the MFE dot-bracket string and pair-probability matrix."""
    seq_struct = truncate_for_structure(seq_struct)
    length = len(seq_struct)

    if RNA is None:
        logger.warning("ViennaRNA is unavailable; returning empty RNAfold outputs.")
        return "." * length, torch.zeros((length, length), dtype=torch.float32)

    safe_seq = seq_struct
    try:
        fold_compound = RNA.fold_compound(safe_seq)
        mfe_db, _ = fold_compound.mfe()
        fold_compound.pf()
        plist = fold_compound.plist_from_probs(probability_cutoff)
        probability_matrix = _pair_probability_matrix_from_plist(length, plist)
        return mfe_db, probability_matrix
    except Exception as exc:
        logger.warning("RNAfold failed on the sanitized sequence; falling back to a zero-probability structure. Error: %s", exc)
        return "." * length, torch.zeros((length, length), dtype=torch.float32)


def run_ipknot(seq_struct: str, timeout_seconds: int = 120) -> Optional[str]:
    """Run IPknot if available; otherwise return None."""
    seq_struct = truncate_for_structure(seq_struct)
    ipknot_bin = shutil.which("ipknot")
    if not ipknot_bin:
        return None

    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fasta_file:
        fasta_file.write(">seq\n")
        fasta_file.write(seq_struct + "\n")
        fasta_path = fasta_file.name

    try:
        command_candidates = [
            [ipknot_bin, fasta_path],
            [ipknot_bin],
        ]
        output_text = None
        for command in command_candidates:
            try:
                result = subprocess.run(
                    command,
                    input=(f">seq\n{seq_struct}\n" if len(command) == 1 else None),
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=True,
                )
                output_text = result.stdout
                if output_text:
                    break
            except Exception:
                continue

        if not output_text:
            return None

        bracket_line_pattern = re.compile(r"^[\.\(\)\[\]\{\}\<\>A-Za-z]+$")
        candidates = []
        for line in output_text.splitlines():
            line = line.strip()
            if line and len(line) == len(seq_struct) and bracket_line_pattern.match(line):
                candidates.append(line)

        return candidates[-1] if candidates else None
    finally:
        try:
            os.unlink(fasta_path)
        except OSError:
            pass


class DotBracketParser:
    @staticmethod
    def parse_primary_pairs(dot_bracket: str) -> Tuple[Dict[int, int], List[Tuple[int, int]]]:
        pair_map: Dict[int, int] = {}
        pair_edges: List[Tuple[int, int]] = []
        stack: List[int] = []

        for index, character in enumerate(dot_bracket):
            if character == PRIMARY_OPEN:
                stack.append(index)
            elif character == PRIMARY_CLOSE and stack:
                opener = stack.pop()
                pair_map[opener] = index
                pair_map[index] = opener
                pair_edges.append((opener, index))

        return pair_map, pair_edges

    @staticmethod
    def parse_pseudoknot_pairs(dot_bracket: Optional[str]) -> Tuple[Dict[int, int], List[Tuple[int, int]], Set[int]]:
        if not dot_bracket:
            return {}, [], set()

        pair_map: Dict[int, int] = {}
        pair_edges: List[Tuple[int, int]] = []
        stacks: Dict[str, List[int]] = {opening: [] for opening in SECONDARY_BRACKET_PAIRS}
        closing_to_opening = {closing: opening for opening, closing in SECONDARY_BRACKET_PAIRS.items()}

        for index, character in enumerate(dot_bracket):
            if character in SECONDARY_BRACKET_PAIRS:
                stacks[character].append(index)
            elif character in closing_to_opening:
                opening = closing_to_opening[character]
                if stacks[opening]:
                    opener = stacks[opening].pop()
                    pair_map[opener] = index
                    pair_map[index] = opener
                    if opener < index:
                        pair_edges.append((opener, index))
                    else:
                        pair_edges.append((index, opener))

        paired_positions = set(pair_map.keys())
        return pair_map, pair_edges, paired_positions

    @staticmethod
    def merge_rnafold_ipknot(rnafold_db: str, ipknot_db: Optional[str]) -> Dict[str, object]:
        canonical_pair_map, canonical_edges = DotBracketParser.parse_primary_pairs(rnafold_db)
        pseudoknot_pair_map, pseudoknot_edges, pseudoknot_positions = DotBracketParser.parse_pseudoknot_pairs(ipknot_db)

        if pseudoknot_positions:
            for opener, closer in list(canonical_pair_map.items()):
                if opener < closer and ({opener, closer} & pseudoknot_positions):
                    canonical_pair_map.pop(opener, None)
                    canonical_pair_map.pop(closer, None)

        canonical_positions = set(canonical_pair_map.keys())
        canonical_edges = [
            (left, right)
            for left, right in canonical_edges
            if left in canonical_positions and right in canonical_positions
        ]

        merged_pair_map = dict(canonical_pair_map)
        merged_pair_edges = list(canonical_edges)

        for opener, closer in pseudoknot_pair_map.items():
            if opener < closer:
                merged_pair_map[opener] = closer
                merged_pair_map[closer] = opener
                merged_pair_edges.append((opener, closer))

        paired_positions = set(merged_pair_map.keys())

        return {
            "canonical_db": rnafold_db,
            "merged_db": rnafold_db,
            "canonical_pair_map": canonical_pair_map,
            "merged_pair_map": merged_pair_map,
            "paired_positions": paired_positions,
            "pair_edges": canonical_edges,
            "merged_pair_edges": merged_pair_edges,
            "pseudoknot_positions": pseudoknot_positions,
            "pseudoknot_edges": pseudoknot_edges,
        }


def compute_pairing_indicator(pair_map: Dict[int, int], length: int) -> torch.Tensor:
    indicator = torch.zeros((length, 1), dtype=torch.float32)
    paired_positions = {index for index in pair_map.keys() if 0 <= index < length}
    for index in paired_positions:
        indicator[index, 0] = 1.0
    return indicator


def compute_pairing_probability_feature(probability_matrix: torch.Tensor) -> torch.Tensor:
    if probability_matrix.numel() == 0:
        return torch.zeros((0, 1), dtype=torch.float32)
    return probability_matrix.sum(dim=1, keepdim=True).to(dtype=torch.float32)


def sinusoidal_position_encoding(length: int, d_model: int = 64) -> torch.Tensor:
    if length <= 0:
        return torch.zeros((0, d_model), dtype=torch.float32)

    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
    encoding = torch.zeros((length, d_model), dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding


class StructuralContextAnnotator:
    @staticmethod
    def _build_canonical_tree(dot_bracket: str) -> List[Dict[str, object]]:
        roots: List[Dict[str, object]] = []
        stack: List[Dict[str, object]] = []

        for index, character in enumerate(dot_bracket):
            if character == PRIMARY_OPEN:
                node = {"start": index, "children": []}
                stack.append(node)
            elif character == PRIMARY_CLOSE and stack:
                node = stack.pop()
                node["end"] = index
                if stack:
                    stack[-1]["children"].append(node)
                else:
                    roots.append(node)

        return roots

    @staticmethod
    def _mark_range(labels: torch.Tensor, start: int, end: int, class_index: int, assigned: torch.Tensor) -> None:
        if start > end:
            return
        for position in range(start, end + 1):
            if 0 <= position < labels.size(0) and not assigned[position]:
                labels[position].zero_()
                labels[position, class_index] = 1.0
                assigned[position] = True

    @staticmethod
    def _mark_single(labels: torch.Tensor, position: int, class_index: int, assigned: torch.Tensor) -> None:
        if 0 <= position < labels.size(0) and not assigned[position]:
            labels[position].zero_()
            labels[position, class_index] = 1.0
            assigned[position] = True

    @staticmethod
    def _annotate_node(node: Dict[str, object], labels: torch.Tensor, assigned: torch.Tensor, pseudoknot_positions: Set[int]) -> None:
        start = int(node["start"])
        end = int(node["end"])
        children = sorted(node.get("children", []), key=lambda child: child["start"])

        if start in pseudoknot_positions or end in pseudoknot_positions:
            StructuralContextAnnotator._mark_single(labels, start, 5, assigned)
            StructuralContextAnnotator._mark_single(labels, end, 5, assigned)
        else:
            StructuralContextAnnotator._mark_single(labels, start, 0, assigned)
            StructuralContextAnnotator._mark_single(labels, end, 0, assigned)

        for child in children:
            StructuralContextAnnotator._annotate_node(child, labels, assigned, pseudoknot_positions)

        child_intervals = [(int(child["start"]), int(child["end"])) for child in children]
        cursor = start + 1

        if not child_intervals:
            StructuralContextAnnotator._mark_range(labels, start + 1, end - 1, 1, assigned)
            return

        if len(child_intervals) == 1:
            child_start, child_end = child_intervals[0]
            left_start, left_end = cursor, child_start - 1
            right_start, right_end = child_end + 1, end - 1
            left_len = max(0, left_end - left_start + 1)
            right_len = max(0, right_end - right_start + 1)

            if left_len > 0 and right_len > 0:
                StructuralContextAnnotator._mark_range(labels, left_start, left_end, 3, assigned)
                StructuralContextAnnotator._mark_range(labels, right_start, right_end, 3, assigned)
            else:
                StructuralContextAnnotator._mark_range(labels, left_start, left_end, 2, assigned)
                StructuralContextAnnotator._mark_range(labels, right_start, right_end, 2, assigned)
            return

        # Multiple children imply a multibranch loop between child stems.
        for child_start, child_end in child_intervals:
            StructuralContextAnnotator._mark_range(labels, cursor, child_start - 1, 4, assigned)
            cursor = child_end + 1
        StructuralContextAnnotator._mark_range(labels, cursor, end - 1, 4, assigned)

    @staticmethod
    def annotate_structural_context(canonical_db: str, pair_map: Dict[int, int], pseudoknot_positions: Set[int]) -> torch.Tensor:
        length = len(canonical_db)
        labels = torch.zeros((length, len(STRUCTURE_CONTEXT_ORDER)), dtype=torch.float32)
        assigned = torch.zeros(length, dtype=torch.bool)

        roots = StructuralContextAnnotator._build_canonical_tree(canonical_db)
        if roots:
            roots = sorted(roots, key=lambda node: node["start"])
            cursor = 0
            for root in roots:
                root_start = int(root["start"])
                StructuralContextAnnotator._mark_range(labels, cursor, root_start - 1, 4, assigned)
                StructuralContextAnnotator._annotate_node(root, labels, assigned, pseudoknot_positions)
                cursor = int(root["end"]) + 1
            StructuralContextAnnotator._mark_range(labels, cursor, length - 1, 4, assigned)
        else:
            StructuralContextAnnotator._mark_range(labels, 0, length - 1, 4, assigned)

        for index in range(length):
            if index in pseudoknot_positions:
                labels[index].zero_()
                labels[index, 5] = 1.0
            elif not assigned[index]:
                labels[index].zero_()
                labels[index, 4] = 1.0

        return labels


class StructureGraphBuilder:
    @staticmethod
    def build_edge_index(length: int, pair_edges: Sequence[Tuple[int, int]], pseudoknot_edges: Sequence[Tuple[int, int]] = (), include_backbone: bool = True) -> torch.Tensor:
        directed_edges: List[Tuple[int, int]] = []

        if include_backbone and length > 1:
            for index in range(length - 1):
                directed_edges.append((index, index + 1))
                directed_edges.append((index + 1, index))

        for left, right in list(pair_edges) + list(pseudoknot_edges):
            directed_edges.append((left, right))
            directed_edges.append((right, left))

        if not directed_edges:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(directed_edges, dtype=torch.long).t().contiguous()

    @staticmethod
    def build_view_tensors(pairing_indicator: torch.Tensor, probability_feature: torch.Tensor, structural_context: torch.Tensor, positional_encoding: torch.Tensor) -> Dict[str, torch.Tensor]:
        views = {
            "Struct-PB": torch.cat([pairing_indicator, probability_feature], dim=1),
            "Struct-PC": torch.cat([pairing_indicator, structural_context], dim=1),
            "Struct-PE": torch.cat([pairing_indicator, positional_encoding], dim=1),
            "Struct-PBC": torch.cat([pairing_indicator, probability_feature, structural_context], dim=1),
            "Struct-PBE": torch.cat([pairing_indicator, probability_feature, positional_encoding], dim=1),
            "Struct-PCE": torch.cat([pairing_indicator, structural_context, positional_encoding], dim=1),
            "Struct-PBCE": torch.cat([pairing_indicator, probability_feature, structural_context, positional_encoding], dim=1),
        }
        return {name: tensor.to(dtype=torch.float32) for name, tensor in views.items()}

    @staticmethod
    def build_view_graphs(sample_id: str, truncated_sequence: str, rnafold_db: str, ipknot_db: Optional[str], probability_matrix: torch.Tensor, include_backbone: bool = True) -> Dict[str, Data]:
        merged = DotBracketParser.merge_rnafold_ipknot(rnafold_db, ipknot_db)
        length = len(truncated_sequence)
        pairing_indicator = compute_pairing_indicator(merged["merged_pair_map"], length)
        pairing_probability = compute_pairing_probability_feature(probability_matrix)
        structural_context = StructuralContextAnnotator.annotate_structural_context(
            merged["canonical_db"],
            merged["merged_pair_map"],
            merged["pseudoknot_positions"],
        )
        positional_encoding = sinusoidal_position_encoding(length, 64)
        view_tensors = StructureGraphBuilder.build_view_tensors(pairing_indicator, pairing_probability, structural_context, positional_encoding)
        edge_index = StructureGraphBuilder.build_edge_index(length, merged["pair_edges"], merged["pseudoknot_edges"], include_backbone=include_backbone)

        view_graphs: Dict[str, Data] = {}
        for view_name in STRUCTURE_VIEW_ORDER:
            view_graphs[view_name] = Data(
                x=view_tensors[view_name],
                edge_index=edge_index,
                num_nodes=length,
                sample_id=sample_id,
                raw_sequence=truncated_sequence,
                dbn_string=merged["canonical_db"],
                merged_db=merged["merged_db"],
                original_sequence_length=length,
                structure_sequence_length=length,
                paired_positions=sorted(list(merged["paired_positions"])),
                pseudoknot_positions=sorted(list(merged["pseudoknot_positions"])),
                pair_edges=merged["pair_edges"],
                pseudoknot_edges=merged["pseudoknot_edges"],
            )

        return view_graphs


class StructureCache:
    def __init__(self, cache_dir: Optional[str] = None, version: str = "v2_precomputed_dbn"):
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        self.version = version
        self._memory_cache: Dict[str, Dict[str, object]] = {}
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def make_key(self, sample_id: str, truncated_sequence: str) -> str:
        base = f"{self.version}|{sample_id}|{truncated_sequence}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{key}.pt")

    def get(self, key: str) -> Optional[Dict[str, object]]:
        if key in self._memory_cache:
            return self._memory_cache[key]

        cache_path = self._cache_path(key)
        if cache_path and os.path.exists(cache_path):
            artifact = torch.load(cache_path, map_location="cpu")
            self._memory_cache[key] = artifact
            return artifact
        return None

    def set(self, key: str, artifact: Dict[str, object]) -> None:
        self._memory_cache[key] = artifact
        cache_path = self._cache_path(key)
        if cache_path:
            torch.save(artifact, cache_path)


class StructurePreprocessor:
    def __init__(self, config: Optional[Dict[str, object]] = None, cache_dir: Optional[str] = None):
        config = config or {}
        self.max_len = int(config.get("max_len", 3000))
        self.use_rnafold = bool(config.get("use_rnafold", True))
        self.use_ipknot = bool(config.get("use_ipknot", True))
        self.include_backbone = bool(config.get("include_backbone", True))
        cache_version = str(config.get("cache_version", "v2_precomputed_dbn"))
        self.cache = StructureCache(cache_dir=cache_dir or config.get("cache_dir"), version=cache_version)

    def process(self, sample_id: str, raw_sequence: str, dbn_string: Optional[str] = None) -> Dict[str, object]:
        raw_sequence = raw_sequence or ""
        sanitized = sanitize_rna_sequence(raw_sequence)
        truncated = truncate_for_structure(sanitized, self.max_len)
        sample_key = self.cache.make_key(sample_id or "unknown", truncated)

        cached = self.cache.get(sample_key)
        if cached is not None:
            return cached

        normalized_dbn = _normalize_dot_bracket_string(dbn_string, len(truncated))

        if normalized_dbn:
            merged = DotBracketParser.merge_rnafold_ipknot(normalized_dbn, normalized_dbn)
            probability_matrix = _pair_probability_matrix_from_pair_map(merged["merged_pair_map"], len(truncated))
            view_graphs = StructureGraphBuilder.build_view_graphs(
                sample_id=str(sample_id or "unknown"),
                truncated_sequence=truncated,
                rnafold_db=merged["canonical_db"],
                ipknot_db=normalized_dbn,
                probability_matrix=probability_matrix,
                include_backbone=self.include_backbone,
            )

            artifact: Dict[str, object] = {
                "sample_id": str(sample_id or "unknown"),
                "original_sequence_length": len(sanitized),
                "structure_sequence_length": len(truncated),
                "truncated_sequence": truncated,
                "rnafold_db": merged["canonical_db"],
                "ipknot_db": normalized_dbn,
                "merged_db": merged["merged_db"],
                "paired_positions": merged["paired_positions"],
                "pair_edges": merged["pair_edges"],
                "pseudoknot_positions": merged["pseudoknot_positions"],
                "pseudoknot_edges": merged["pseudoknot_edges"],
                "P_bp": probability_matrix,
                "P": compute_pairing_indicator(merged["merged_pair_map"], len(truncated)),
                "B": compute_pairing_probability_feature(probability_matrix),
                "C": StructuralContextAnnotator.annotate_structural_context(
                    merged["canonical_db"],
                    merged["merged_pair_map"],
                    merged["pseudoknot_positions"],
                ),
                "E": sinusoidal_position_encoding(len(truncated), 64),
                "view_graphs": view_graphs,
            }

            self.cache.set(sample_key, artifact)
            return artifact

        if self.use_rnafold:
            rnafold_db, probability_matrix = run_rnafold(truncated)
        else:
            rnafold_db = "." * len(truncated)
            probability_matrix = torch.zeros((len(truncated), len(truncated)), dtype=torch.float32)

        if self.use_ipknot:
            ipknot_db = run_ipknot(truncated)
        else:
            ipknot_db = None

        view_graphs = StructureGraphBuilder.build_view_graphs(
            sample_id=str(sample_id or "unknown"),
            truncated_sequence=truncated,
            rnafold_db=rnafold_db,
            ipknot_db=ipknot_db,
            probability_matrix=probability_matrix,
            include_backbone=self.include_backbone,
        )

        merged = DotBracketParser.merge_rnafold_ipknot(rnafold_db, ipknot_db)
        artifact: Dict[str, object] = {
            "sample_id": str(sample_id or "unknown"),
            "original_sequence_length": len(sanitized),
            "structure_sequence_length": len(truncated),
            "truncated_sequence": truncated,
            "rnafold_db": rnafold_db,
            "ipknot_db": ipknot_db,
            "merged_db": merged["merged_db"],
            "paired_positions": merged["paired_positions"],
            "pair_edges": merged["pair_edges"],
            "pseudoknot_positions": merged["pseudoknot_positions"],
            "pseudoknot_edges": merged["pseudoknot_edges"],
            "P_bp": probability_matrix,
            "P": compute_pairing_indicator(merged["merged_pair_map"], len(truncated)),
            "B": compute_pairing_probability_feature(probability_matrix),
            "C": StructuralContextAnnotator.annotate_structural_context(
                rnafold_db,
                merged["merged_pair_map"],
                merged["pseudoknot_positions"],
            ),
            "E": sinusoidal_position_encoding(len(truncated), 64),
            "view_graphs": view_graphs,
        }

        self.cache.set(sample_key, artifact)
        return artifact
