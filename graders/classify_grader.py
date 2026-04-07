from typing import List, Optional

from env.models import ClassifyAction, Message, Reward


class ClassifyGrader:
    """
    Easy task grader.
    Scores based on exact-match accuracy (correct/total). Coverage is a gating penalty, not a bonus.
    Fix #3: coverage no longer adds positive reward for wrong-but-complete answers.
    """

    def grade(
        self,
        action: Optional[ClassifyAction],
        messages: List[Message],
    ) -> Reward:
        if action is None:
            return Reward(
                score=0.0,
                breakdown={"accuracy": 0.0, "coverage": 0.0, "penalty": 0.0},
                feedback="No classify action provided.",
            )

        ground_truth = {
            msg.id: msg.ground_truth_label
            for msg in messages
            if msg.ground_truth_label is not None
        }

        total = len(ground_truth)
        if total == 0:
            return Reward(
                score=0.0,
                breakdown={"accuracy": 0.0, "coverage": 0.0, "penalty": 0.0},
                feedback="No ground truth labels available.",
            )

        classified = action.classifications
        covered = sum(1 for mid in ground_truth if mid in classified)
        correct = sum(
            1
            for mid, label in ground_truth.items()
            if classified.get(mid) == label
        )

        coverage = covered / total
        accuracy = correct / total

        # Coverage penalty: missing messages hurt the score
        coverage_penalty = (1.0 - coverage) * 0.3
        score = round(max(0.0, accuracy - coverage_penalty), 4)

        feedback_parts = []
        if accuracy >= 0.9:
            feedback_parts.append("Excellent classification accuracy.")
        elif accuracy >= 0.7:
            feedback_parts.append("Good classification accuracy.")
        elif accuracy >= 0.5:
            feedback_parts.append("Moderate accuracy — some labels are off.")
        else:
            feedback_parts.append("Low accuracy — review the label definitions.")

        if coverage < 1.0:
            missed = total - covered
            feedback_parts.append(f"{missed} message(s) were not classified.")

        return Reward(
            score=score,
            breakdown={
                "accuracy": round(accuracy, 4),
                "coverage": round(coverage, 4),
                "penalty": round(coverage_penalty, 4),
            },
            feedback=" ".join(feedback_parts),
        )
