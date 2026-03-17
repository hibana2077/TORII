import torch
import torch.optim as optim

from src.torii.graph.soft_path import compute_soft_path_matrix


def test_convergence():
    torch.manual_seed(42)
    n = 15

    w_a = torch.rand(n, n)
    w_a = (w_a + w_a.T) / 2
    w_a.fill_diagonal_(0)

    perm = torch.randperm(n)
    p_true = torch.zeros(n, n)
    p_true[torch.arange(n), perm] = 1.0

    w_b = p_true.T @ w_a @ p_true
    noise = 0.05 * torch.rand(n, n)
    noise = (noise + noise.T) / 2
    w_b = w_b + noise
    w_b.fill_diagonal_(0)

    d_a = compute_soft_path_matrix(w_a, tau=0.5, k_hops=3)
    d_b = compute_soft_path_matrix(w_b, tau=0.5, k_hops=3)

    m = torch.randn(n, n, requires_grad=True)
    optimizer = optim.Adam([m], lr=0.1)

    print("Starting optimization...")
    for epoch in range(1001):
        optimizer.zero_grad()
        p = torch.softmax(m, dim=1)

        l_edge = torch.norm(w_a - p @ w_b @ p.T, p="fro") ** 2
        l_path = torch.norm(d_a - p @ d_b @ p.T, p="fro") ** 2
        loss = 0.1 * l_edge + 0.5 * l_path

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(
                f"Epoch {epoch:4d} | Total Loss: {loss.item():.4f} | "
                f"L_edge: {l_edge.item():.4f} | L_path: {l_path.item():.4f}"
            )

    p_learned = torch.softmax(m, dim=1)
    pred_perm = p_learned.argmax(dim=1)
    acc = (pred_perm == perm).float().mean().item()

    print("\nOptimization Finished!")
    print(f"Final Alignment Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    test_convergence()
