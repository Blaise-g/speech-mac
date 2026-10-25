"""Trained Mimi local-attention predicate from Pocket TTS Python reference semantics."""

def allowed_positions(query_position: int, key_positions: range, context: int = 250) -> tuple[int, ...]:
    """Return causal local keys: delta < context, yielding at most `context` keys."""
    if query_position < 0 or context < 1:
        raise ValueError('query_position must be non-negative and context positive')
    return tuple(k for k in key_positions if 0 <= query_position - k < context)
