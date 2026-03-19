# Method

## 概述

TORII 的核心目標是比較兩張影像在「區域關係結構」上的相似度，而不是只看整張圖的全域特徵。整體流程是先從 Vision Transformer (ViT) 擷取 patch token，將 patch 聚合成較少量的 super-node，再把每張圖表示成一個 super-node graph。之後利用跨圖的雙向 soft transport，把兩個 graph 對齊，並在節點與邊兩個層次上計算 alignment loss。

目前專案中的可執行實作主要在 [exp/torii_core.py](/c:/Users/hiban/Desktop/code%20space/TORII/exp/torii_core.py)。

## 1. 影像到 patch token

給定影像 $I$，先做標準化與 resize，接著送入 timm 的 ViT backbone，取出 `forward_features` 的輸出，移除 prefix token 後保留 patch tokens：

$$
X = [x_1, x_2, \dots, x_N]^\top \in \mathbb{R}^{N \times d},
$$

其中 $N$ 是 patch 數量，$d$ 是 token 維度。實作上假設 patch token 可以還原成正方形 patch grid，對應程式在 [exp/torii_core.py:48](/c:/Users/hiban/Desktop/code%20space/TORII/exp/torii_core.py#L48)。

## 2. Patch graph coarsening: super-node 建構

為了降低 patch graph 的規模，方法中先對 patch token 做 k-means，將 $N$ 個 patch 分成 $K$ 群。每一群的 centroid 作為一個 super-node feature：

$$
Z = [z_1, z_2, \dots, z_K]^\top \in \mathbb{R}^{K \times d}.
$$

實作上使用 torch 版本的 k-means，並以 centroid 間的 cosine similarity 建立 super-node adjacency：

$$
\mathrm{sim}(z_i, z_j) = \frac{z_i^\top z_j}{\|z_i\| \|z_j\|},
$$

$$
W_{ij} = \frac{\mathrm{sim}(z_i, z_j) + 1}{2}, \qquad W_{ii}=0.
$$

因此每張影像最後會被表示成一個 graph：

$$
G = (V, W), \qquad V = \{z_1, \dots, z_K\}.
$$

對應程式可見 [exp/torii_core.py:70](/c:/Users/hiban/Desktop/code%20space/TORII/exp/torii_core.py#L70) 與 [exp/torii_core.py:105](/c:/Users/hiban/Desktop/code%20space/TORII/exp/torii_core.py#L105)。

## 3. 雙向 soft transport 對齊

給定兩張影像的 super-node feature：

$$
Z^A \in \mathbb{R}^{n_A \times d}, \qquad Z^B \in \mathbb{R}^{n_B \times d},
$$

先計算跨圖節點相似度矩陣：

$$
S_{ij} = \mathrm{sim}(z_i^A, z_j^B).
$$

接著用溫度參數 $\tau_p$ 建立雙向 transport matrix：

$$
P^{A \leftarrow B}_{ij}
=
\frac{\exp(S_{ij}/\tau_p)}
{\sum_{k=1}^{n_B}\exp(S_{ik}/\tau_p)},
$$

$$
P^{B \leftarrow A}_{ji}
=
\frac{\exp(S_{ij}/\tau_p)}
{\sum_{k=1}^{n_A}\exp(S_{kj}/\tau_p)}.
$$

其中 $P^{A \leftarrow B}$ 表示把 graph $B$ 的資訊 transport 到 graph $A$ 的座標系；$P^{B \leftarrow A}$ 則是相反方向。這個設計避免只做單向對齊時的偏置。對應程式在 [exp/torii_core.py:128](/c:/Users/hiban/Desktop/code%20space/TORII/exp/torii_core.py#L128) 與 [exp/torii_core.py:134](/c:/Users/hiban/Desktop/code%20space/TORII/exp/torii_core.py#L134)。

## 4. Alignment losses

### 4.1 Node alignment loss

先用 transport 後的 super-node feature 去重建對方圖上的節點表示：

$$
\mathcal{L}_{\mathrm{node}}
=
\left\| Z^A - P^{A \leftarrow B} Z^B \right\|_F^2
+
\left\| Z^B - P^{B \leftarrow A} Z^A \right\|_F^2.
$$

這一項要求對齊後的節點特徵在雙向上都要相近。

### 4.2 Edge consistency loss

再把對方 graph 的 adjacency transport 過來，比較一階結構是否一致：

$$
\mathcal{L}_{\mathrm{edge}}
=
\left\| W^A - P^{A \leftarrow B} W^B (P^{A \leftarrow B})^\top \right\|_F^2
+
\left\| W^B - P^{B \leftarrow A} W^A (P^{B \leftarrow A})^\top \right\|_F^2.
$$

這一項對應 graph 的 local relation consistency。

## 5. 總目標函數

總分數定義為：

$$
\mathcal{L}_{\mathrm{total}}
=
\lambda_n \mathcal{L}_{\mathrm{node}}
+
\lambda_e \mathcal{L}_{\mathrm{edge}}.
$$

其中 $\lambda_n$、$\lambda_e$ 分別控制節點與邊結構項的權重。

## 6. 直觀解釋

這個方法可以分成兩個層次理解：

1. `Node`：比較兩張圖是否有相似的局部語意區塊。
2. `Edge`：比較這些區塊之間的直接關係是否相似。

因此，若兩張圖屬於同一類別，即使局部位置或外觀有變化，只要它們的區域關係結構仍接近，TORII 就應該得到較低的 alignment score。

## 7. 與 `method.tex` 的關係

[docs/method.tex](/c:/Users/hiban/Desktop/code%20space/TORII/docs/method.tex) 與這份說明現在是一致的，重點放在：

1. variable-cardinality 的 bidirectional super-node alignment
2. node alignment loss
3. edge consistency loss
