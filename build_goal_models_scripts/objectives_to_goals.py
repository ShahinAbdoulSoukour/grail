"""
Convert an ISO/IEC 27001 Annex A "objective" into a goal statement.
goal statement (the subgoal G1, placed under the clause-title root goal G0).

ISO objectives are always written as  "To <verb> ...".
Converting one into a goal means: drop the leading "To", capitalize, and
remove the trailing period -> a directive goal statement.

"To provide management direction and support for information security..."
    -> "Provide management direction and support for information security..."

One main function. A second optional helper splits an objective that states
two intentions joined by "and to".
"""

import re


def convert_objective_to_goal(objective):
    """
    Convert one ISO 27001 objective into a goal statement.

    Example:
    "To ensure authorized user access and to prevent unauthorized access
     to systems and services."
        -> "Ensure authorized user access and prevent unauthorized access
            to systems and services"

    :param objective: an ISO 27001 objective
    """
    # Normalize whitespace and remove trailing punctuation
    text = re.sub(r"\s+", " ", objective.strip()).rstrip(".")

    # Remove the first "To "
    text = re.sub(r"^[Tt]o\s+", "", text)

    # Remove "to" after "and" or "or"
    text = re.sub(r"\b(and|or)\s+to\s+", r"\1 ", text, flags=re.IGNORECASE)

    # Capitalize first letter
    return text[0].upper() + text[1:] if text else text