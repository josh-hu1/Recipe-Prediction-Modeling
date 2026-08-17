import numpy as np


def perm_test_diff_means(df, miss_col, feature_col, B=2000, seed=0):
    """
    Permutation test for whether missingness in miss_col
    is associated with feature_col.

    Test statistic:
    absolute difference in feature means between missing
    and non-missing groups.
    """
    rng = np.random.default_rng(seed)

    sub = df[[miss_col, feature_col]].dropna()
    miss = sub[miss_col].to_numpy()
    x = sub[feature_col].to_numpy()

    obs = abs(x[miss].mean() - x[~miss].mean())

    stats = np.empty(B)

    for b in range(B):
        perm = rng.permutation(miss)
        stats[b] = abs(
            x[perm].mean() - x[~perm].mean()
        )

    pval = np.mean(stats >= obs)

    return float(obs), float(pval)


def paired_swap_perm_test(e1, e2, B=5000, seed=0):
    """
    Paired permutation test comparing two sets of errors.

    Randomly swaps paired errors with probability 0.5.

    Returns:
        T_obs: mean(e1) - mean(e2)
        pval: one-sided p-value for T_obs being smaller
    """
    rng = np.random.default_rng(seed)

    T_obs = e1.mean() - e2.mean()
    Ts = np.empty(B)

    for b in range(B):
        swap = rng.random(len(e1)) < 0.5

        a = e1.copy()
        b_err = e2.copy()

        a[swap], b_err[swap] = b_err[swap], a[swap]

        Ts[b] = a.mean() - b_err.mean()

    pval = np.mean(Ts <= T_obs)

    return float(T_obs), float(pval)