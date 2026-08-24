def pad_sequences(seqs: list, pad_value: int = 0,
                  max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
        N = len(seqs)
        L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)

    # Decide the target width. max(..., default=0) safely handles an empty list.
    if max_len is None:
        L = max((len(s) for s in seqs), default=0)
    else:
        L = max_len

    # Pre-fill the whole (N, L) matrix with the pad value.
    out = np.full((N, L), pad_value, dtype=int)

    # Copy each real sequence into its row, truncating if it is too long.
    for i, s in enumerate(seqs):
        n = min(len(s), L)        # how many real tokens actually fit
        out[i, :n] = s[:n]

    return out
