# TODO: LLM-powered router
#
# Will take a plain-English business problem and return the right model to use.
#
# Example:
#   route("Which customers are about to cancel?")
#   -> { "model": "ChurnModel", "reason": "...", "data_needed": [...] }
#
# Will be built with the Claude API once churn and sentiment are complete.
