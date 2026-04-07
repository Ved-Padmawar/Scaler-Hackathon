import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

from env.models import ClusterAction, Message, Reward

MIN_CLUSTERS = 3
MAX_CLUSTERS = 6

# Generic/stopword labels that give no information about topic
GENERIC_LABELS: Set[str] = {
    "cluster", "group", "other", "misc", "miscellaneous", "unknown",
    "n/a", "messages", "chat", "conversation", "general", "various",
    "mixed", "discussion", "topic",
}

# Common stopwords to exclude from keyword extraction
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "this", "that", "was", "are", "be", "have",
    "i", "you", "we", "they", "he", "she", "me", "my", "your", "our",
    "hi", "hello", "ok", "okay", "yes", "no", "please", "thanks", "thank",
    "ji", "ha", "nahi", "hai", "hain", "bhi", "kya", "aap", "hum",
}


def _extract_keywords(text: str, top_n: int = 8) -> Set[str]:
    """Extract meaningful keywords from text (simple but effective)."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def _cluster_coherence(msg_ids: List[str], msg_text: Dict[str, str]) -> float:
    """
    Internal coherence: fraction of messages in cluster that share at least
    one keyword with the cluster's most common keyword set.
    Scores 0.0-1.0 without needing ground-truth labels.
    """
    if len(msg_ids) <= 1:
        return 0.5  # singleton: neutral score (not rewarded, not punished)

    keyword_sets = [_extract_keywords(msg_text.get(mid, "")) for mid in msg_ids]

    # Count keyword frequency across cluster
    freq: Dict[str, int] = defaultdict(int)
    for kws in keyword_sets:
        for kw in kws:
            freq[kw] = freq[kw] + 1

    # Top keywords that appear in >1 message
    common = {kw for kw, cnt in freq.items() if cnt >= 2}
    if not common:
        return 0.3  # no shared keywords at all

    # Fraction of messages that overlap with common keywords
    shared = sum(1 for kws in keyword_sets if kws & common)
    return shared / len(msg_ids)


def _label_quality_score(label: str) -> float:
    """Score a cluster label 0.0-1.0 based on specificity and descriptiveness."""
    label_lower = label.strip().lower()

    if not label_lower:
        return 0.0

    words = label_lower.split()

    # Generic single-word or blacklisted labels
    if label_lower in GENERIC_LABELS or (len(words) == 1 and label_lower in GENERIC_LABELS):
        return 0.0

    # Any generic word present in a short label is suspicious
    if len(words) <= 2 and any(w in GENERIC_LABELS for w in words):
        return 0.3

    # Must be at least 2 words to be descriptive
    if len(words) < 2:
        return 0.5

    # 2+ words, no generic filler words
    if len(words) >= 3:
        return 1.0

    return 0.85


class ClusterGrader:
    """
    Medium task grader — harder than classify because no labels are provided.
    Scores cluster COHERENCE (internal keyword overlap), not label-recovery purity.
    This means the agent cannot game by copying ground-truth labels.

    Dimensions:
      - coherence (0.45): internal topical consistency, measured by shared keywords
      - coverage  (0.25): fraction of messages assigned (exactly once)
      - label_quality (0.15): specificity and descriptiveness of cluster names
      - structure  (0.15): penalty for going outside 3-6 cluster range
      - duplicate_penalty: deducted from final score
    """

    def grade(
        self,
        action: Optional[ClusterAction],
        messages: List[Message],
    ) -> Reward:
        if action is None:
            return Reward(
                score=0.0,
                breakdown={"coherence": 0.0, "label_quality": 0.0, "coverage": 0.0, "structure": 0.0},
                feedback="No cluster action provided.",
            )

        all_msg_ids = {msg.id for msg in messages}
        msg_text = {msg.id: msg.text for msg in messages}
        total_messages = len(all_msg_ids)

        if total_messages == 0:
            return Reward(
                score=0.0,
                breakdown={"coherence": 0.0, "label_quality": 0.0, "coverage": 0.0, "structure": 0.0},
                feedback="No messages to cluster.",
            )

        clusters = action.clusters
        cluster_labels = action.cluster_labels
        num_clusters = len(clusters)

        # --- Structure: enforce 3-6 clusters ---
        if MIN_CLUSTERS <= num_clusters <= MAX_CLUSTERS:
            structure_score = 1.0
        else:
            deviation = min(abs(num_clusters - MIN_CLUSTERS), abs(num_clusters - MAX_CLUSTERS))
            structure_score = max(0.0, 1.0 - 0.2 * deviation)

        # --- Duplicate/missing assignment validation ---
        all_assigned: List[str] = [mid for ids in clusters.values() for mid in ids]
        unique_assigned = set(all_assigned)
        duplicate_count = len(all_assigned) - len(unique_assigned)
        valid_assigned = unique_assigned & all_msg_ids

        duplicate_penalty = min(0.5, duplicate_count * 0.05)
        coverage = len(valid_assigned) / total_messages if total_messages > 0 else 0.0

        # --- Internal coherence (replaces purity — no ground truth needed) ---
        coherence_scores = []
        for cluster_id, msg_ids in clusters.items():
            valid_ids = [mid for mid in set(msg_ids) if mid in all_msg_ids]
            if not valid_ids:
                continue
            c = _cluster_coherence(valid_ids, msg_text)
            # Weight by cluster size
            coherence_scores.append((c, len(valid_ids)))

        if coherence_scores:
            total_weight = sum(w for _, w in coherence_scores)
            coherence = sum(c * w for c, w in coherence_scores) / total_weight
        else:
            coherence = 0.0

        # --- Label quality ---
        label_quality_scores = [
            _label_quality_score(cluster_labels.get(cid, ""))
            for cid in clusters
        ]
        label_quality = (
            sum(label_quality_scores) / len(label_quality_scores)
            if label_quality_scores
            else 0.0
        )

        # Final score
        raw_score = (
            0.45 * coherence
            + 0.25 * coverage
            + 0.15 * label_quality
            + 0.15 * structure_score
            - duplicate_penalty
        )
        score = round(max(0.0, min(1.0, raw_score)), 4)

        feedback_parts = []
        if coherence >= 0.75:
            feedback_parts.append("Clusters are topically coherent.")
        elif coherence >= 0.5:
            feedback_parts.append("Clusters have moderate coherence; some messages seem off-topic.")
        else:
            feedback_parts.append("Clusters mix unrelated messages — group by topic more carefully.")

        if coverage < 0.9:
            feedback_parts.append(f"Only {coverage:.0%} of messages assigned.")

        if duplicate_count > 0:
            feedback_parts.append(f"{duplicate_count} duplicate message assignment(s) penalized.")

        if not (MIN_CLUSTERS <= num_clusters <= MAX_CLUSTERS):
            feedback_parts.append(f"Expected {MIN_CLUSTERS}-{MAX_CLUSTERS} clusters, got {num_clusters}.")

        if label_quality < 0.7:
            feedback_parts.append("Cluster labels are too generic — use specific 2+ word descriptors.")

        return Reward(
            score=score,
            breakdown={
                "coherence": round(coherence, 4),
                "coverage": round(coverage, 4),
                "label_quality": round(label_quality, 4),
                "structure": round(structure_score, 4),
                "duplicate_penalty": round(duplicate_penalty, 4),
            },
            feedback=" ".join(feedback_parts),
        )
