import numpy as np
import torch
import torch.optim as optim

from src.torii.graph.soft_path import compute_soft_path_matrix

HARD_PRESET = {
    "n": 60,
    "d": 8,
    "feat_noise": 0.35,
    "edge_noise": 0.35,
}


def run_synthetic_matching(seed, config):
    """Run one synthetic matching experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = config.get("n", 20)
    d = config.get("d", 16)
    feat_noise = config.get("feat_noise", 0.1)
    edge_noise = config.get("edge_noise", 0.1)

    x_a = torch.randn(n, d)
    w_a = torch.rand(n, n)
    w_a = (w_a + w_a.T) / 2
    w_a.fill_diagonal_(0)

    perm = torch.randperm(n)
    p_true = torch.zeros(n, n)
    p_true[torch.arange(n), perm] = 1.0

    x_b = x_a[perm] + feat_noise * torch.randn(n, d)
    w_b = p_true.T @ w_a @ p_true
    noise = edge_noise * torch.rand(n, n)
    noise = (noise + noise.T) / 2
    w_b = w_b + noise
    w_b.fill_diagonal_(0)

    tau = config.get("tau", 0.5)
    k_hops = config.get("k_hops", 3)
    d_a = compute_soft_path_matrix(w_a, tau=tau, k_hops=k_hops)
    d_b = compute_soft_path_matrix(w_b, tau=tau, k_hops=k_hops)

    if config.get("shuffle_path", False):
        shuffle_idx = torch.randperm(n)
        d_b = d_b[shuffle_idx][:, shuffle_idx]

    m = torch.randn(n, n, requires_grad=True)
    optimizer = optim.Adam([m], lr=0.1)

    use_edge = config.get("use_edge", False)
    use_path = config.get("use_path", False)
    mu = config.get("mu", 0.5)
    lam = config.get("lambda", 0.5)
    epochs = 400

    for _ in range(epochs):
        optimizer.zero_grad()
        p = torch.softmax(m, dim=1)

        l_base = torch.norm(x_a - p @ x_b, p="fro") ** 2
        loss = l_base

        if use_edge:
            l_edge = torch.norm(w_a - p @ w_b @ p.T, p="fro") ** 2
            loss = loss + lam * l_edge

        if use_path:
            l_path = torch.norm(d_a - p @ d_b @ p.T, p="fro") ** 2
            loss = loss + mu * l_path

        loss.backward()
        optimizer.step()

    p_learned = torch.softmax(m, dim=1)
    pred_perm = p_learned.argmax(dim=1)
    inv_perm = torch.argsort(perm)
    acc = (pred_perm == inv_perm).float().mean().item()

    path_error = (torch.norm(d_a - p_learned @ d_b @ p_learned.T, p="fro") / (n ** 2)).item()
    return acc, path_error


def run_ablation():
    print("=== 1. Core Ablation Experiment ===")
    seeds = [42, 100, 999]
    configs = {
        "Base": {"use_edge": False, "use_path": False},
        "Base + Edge": {"use_edge": True, "use_path": False},
        "Base + Path": {"use_edge": False, "use_path": True},
        "Base + Edge + Path": {"use_edge": True, "use_path": True},
    }

    print(
        "Hard preset:",
        f"n={HARD_PRESET['n']}, d={HARD_PRESET['d']}, "
        f"feat_noise={HARD_PRESET['feat_noise']}, edge_noise={HARD_PRESET['edge_noise']}",
    )

    results = {}
    for name, cfg in configs.items():
        accs, pes = [], []
        merged_cfg = {**cfg, **HARD_PRESET}
        for seed in seeds:
            acc, pe = run_synthetic_matching(seed, merged_cfg)
            accs.append(acc)
            pes.append(pe)
        results[name] = {
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "pe_mean": np.mean(pes),
            "pe_std": np.std(pes),
        }

    print(f"{'Method':<20} | {'Test Accuracy (%)':<20} | {'Path Consistency Err':<20}")
    print("-" * 65)
    for name, metrics in results.items():
        acc_str = f"{metrics['acc_mean'] * 100:.1f} +- {metrics['acc_std'] * 100:.1f}"
        pe_str = f"{metrics['pe_mean']:.4f} +- {metrics['pe_std']:.4f}"
        print(f"{name:<20} | {acc_str:<20} | {pe_str:<20}")
    print("")


def run_shuffle_control():
    print("=== 2. Shuffled Path Control ===")
    seeds = [42, 100, 999]
    print(
        "Hard preset:",
        f"n={HARD_PRESET['n']}, d={HARD_PRESET['d']}, "
        f"feat_noise={HARD_PRESET['feat_noise']}, edge_noise={HARD_PRESET['edge_noise']}",
    )
    configs = {
        "Base + True Path": {"use_edge": False, "use_path": True, "shuffle_path": False},
        "Base + Shuffled Path": {"use_edge": False, "use_path": True, "shuffle_path": True},
    }

    for name, cfg in configs.items():
        merged_cfg = {**cfg, **HARD_PRESET}
        accs = [run_synthetic_matching(seed, merged_cfg)[0] for seed in seeds]
        print(f"{name:<22} | Acc: {np.mean(accs) * 100:.1f}% +- {np.std(accs) * 100:.1f}%")
    print("")


def run_parameter_sweeps():
    print("=== 3. Parameter Sensitivity (mu and tau) ===")
    seeds = [42, 100, 999]
    print(
        "Hard preset:",
        f"n={HARD_PRESET['n']}, d={HARD_PRESET['d']}, "
        f"feat_noise={HARD_PRESET['feat_noise']}, edge_noise={HARD_PRESET['edge_noise']}",
    )

    mus = [0.01, 0.1, 0.5, 1.0, 5.0]
    mu_results = []
    for mu in mus:
        cfg = {"use_edge": True, "use_path": True, "mu": mu}
        merged_cfg = {**cfg, **HARD_PRESET}
        accs = [run_synthetic_matching(seed, merged_cfg)[0] for seed in seeds]
        mu_results.append((mu, np.mean(accs), np.std(accs)))

    print("mu sweep (Base+Edge+Path):")
    for mu, acc_mean, acc_std in mu_results:
        print(f"  mu={mu:<4} -> Acc: {acc_mean * 100:.1f}% +- {acc_std * 100:.1f}%")

    taus = [0.1, 0.5, 1.0, 2.0, 5.0]
    tau_results = []
    for tau in taus:
        cfg = {"use_edge": True, "use_path": True, "tau": tau}
        merged_cfg = {**cfg, **HARD_PRESET}
        accs = [run_synthetic_matching(seed, merged_cfg)[0] for seed in seeds]
        tau_results.append((tau, np.mean(accs), np.std(accs)))

    print("\ntau sweep (Base+Edge+Path):")
    for tau, acc_mean, acc_std in tau_results:
        print(f"  tau={tau:<4} -> Acc: {acc_mean * 100:.1f}% +- {acc_std * 100:.1f}%")


def run_all():
    run_ablation()
    run_shuffle_control()
    run_parameter_sweeps()
