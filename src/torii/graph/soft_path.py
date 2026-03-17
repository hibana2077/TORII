import torch


def compute_soft_path_matrix(cost_matrix: torch.Tensor, tau: float = 1.0, k_hops: int = 3) -> torch.Tensor:
    """Compute soft path geometry matrix from an edge-cost matrix.

    D_tau(i,j) = -tau * log(sum_paths exp(-c(path) / tau)).
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    if k_hops < 1:
        raise ValueError("k_hops must be >= 1")

    c = cost_matrix.clone()
    inf_mask = torch.isinf(c)
    c[inf_mask] = 1e6

    a = torch.exp(-c / tau)
    a[inf_mask] = 0.0
    a = a - torch.diag(torch.diag(a))

    a_k = a
    sum_a = a.clone()
    for _ in range(1, k_hops):
        a_k = torch.matmul(a_k, a)
        sum_a = sum_a + a_k

    d_tau = -tau * torch.log(sum_a + 1e-8)
    d_tau.fill_diagonal_(0.0)
    return d_tau
