## Method

### Overview

Given two Vision Transformer (ViT) images, we first construct a relation graph over patch tokens for each image, where nodes represent patch tokens and edges encode pairwise relations. Instead of redesigning node matching itself, we assume a soft node correspondence matrix is available from an existing matching module. Our goal is to regularize this correspondence by enforcing not only local edge consistency, but also **multi-hop path consistency** between the two relation graphs.

The key idea is simple: if two nodes are matched, then the relation geometry induced by paths around them should also agree. To this end, we introduce a **differentiable path-aware relation alignment loss** that compares soft path structures under the current node correspondence.

### Relation Graphs and Soft Node Correspondence

Let \(G^A=(V^A, W^A)\) and \(G^B=(V^B, W^B)\) denote the two patch relation graphs, where \(W^A\) and \(W^B\) are edge-weight matrices. A soft node correspondence matrix \(P \in \mathbb{R}^{n_A \times n_B}\) aligns nodes in the two graphs. Here, \(P\) is a relaxed permutation matrix and can be produced by any existing node matching method.

This design intentionally keeps the node alignment module orthogonal to our contribution. Our method is a plug-in structural objective that improves correspondence quality by constraining graph relations beyond local node similarity.

### Local Relation Consistency

As a basic structural constraint, we first encourage local edge agreement after transporting graph \(B\) into the coordinate system of graph \(A\):

$$
\mathcal{L}_{\text{edge}} = \| W^A - P W^B P^\top \|_F^2.
$$

This term aligns first-order relations, but it only captures local structure. In ViT patch graphs, however, semantically meaningful interactions are often transmitted through multiple hops rather than a single edge.

### Path-Aware Relation Alignment

To capture such higher-order consistency, we define a soft path geometry on each graph. For any node pair \((i,j)\), instead of using only the single shortest path, we aggregate over all valid paths between them with a soft minimum:

$$
D_\tau(i,j) = -\tau \log \sum_{\pi \in \Pi(i,j)} \exp\left(-\frac{c(\pi)}{\tau}\right),
$$

where \(\Pi(i,j)\) is the set of paths from node \(i\) to node \(j\), \(c(\pi)\) is the path cost obtained by summing edge weights along \(\pi\), and \(\tau>0\) is a temperature parameter. When \(\tau\) is small, this quantity approaches the classical shortest-path distance; for larger \(\tau\), it becomes a smooth aggregation over multiple competitive paths.

This construction can be viewed as a graph analogue of the soft-min principle behind soft-DTW: rather than committing to a single discrete alignment or path, we optimize a differentiable objective induced by a distribution over feasible paths.

Let \(D_\tau^A\) and \(D_\tau^B\) be the resulting soft path-geometry matrices for the two graphs. We then enforce consistency under the current node correspondence:

$$
\mathcal{L}_{\text{path}} = \| D_\tau^A - P D_\tau^B P^\top \|_F^2.
$$

Unlike local edge matching, this term explicitly preserves **multi-hop relational structure**. Intuitively, if two regions are matched across images, then their accessibility patterns and soft geodesic relations to the rest of the graph should also be aligned.

### Final Objective

Our final training objective augments any base node-matching loss with local and path-aware structural regularization:

$$
\mathcal{L} = \mathcal{L}_{\text{node}} + \lambda \mathcal{L}_{\text{edge}} + \mu \mathcal{L}_{\text{path}},
$$

where \(\mathcal{L}_{\text{node}}\) denotes the loss used by the underlying correspondence module, and \(\lambda, \mu\) control the strengths of local and multi-hop alignment constraints.

Overall, our method does not replace existing node matching methods. Instead, it provides a clean and differentiable **path-aware relation alignment objective** that can be attached to any soft correspondence framework, enabling the learned alignment to respect both node-level matching and graph-level path geometry.
