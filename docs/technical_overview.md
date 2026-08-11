# Technical Overview: Hyperbolic Asymmetric Attention

## Abstract

Hyperbolic Vision Transformers can represent hierarchy through radial depth, but a distance-based attention compatibility does not explicitly distinguish the forward, general-to-specific direction from its reverse. This work introduces Hyperbolic Asymmetric Attention (HAA), a Lorentz-manifold attention score that combines a stabilized geodesic term with a directional term adapted from hyperbolic entailment cones. A preliminary diagnosis first measures whether a trained HexFormer baseline already provides the radial and angular structure required by the mechanism. It does not: no encoder layer satisfies the joint Phase-0 criteria. The method is therefore studied together with targeted objectives, architectural depth controls, and geometric telemetry. On CIFAR-100, the experimental program comprises one re-trained HexFormer baseline and 20 directional configurations. The best configuration reaches 76.56% top-1 accuracy, 1.07 percentage points above the 75.49% baseline. Separate configurations produce the strongest sustained negative angular signal and the strongest non-collapsed cone occupancy. The results show that targeted geometric properties can be shaped during training, but do not establish that the intended cone regime is the sole source of the classification improvement.

## Contents

1. [Introduction](#1-introduction)
2. [Background and related work](#2-background-and-related-work)
3. [Problem formulation](#3-problem-formulation)
4. [Baseline geometry and Phase-0 diagnosis](#4-baseline-geometry-and-phase-0-diagnosis)
5. [Hyperbolic Asymmetric Attention](#5-hyperbolic-asymmetric-attention)
6. [Training the geometry](#6-training-the-geometry)
7. [Architectural depth controls](#7-architectural-depth-controls)
8. [Geometric telemetry and diagnostics](#8-geometric-telemetry-and-diagnostics)
9. [Experimental setup](#9-experimental-setup)
10. [Experimental progression and results](#10-experimental-progression-and-results)
11. [Analysis and interpretation](#11-analysis-and-interpretation)
12. [Current implementation and reproducibility](#12-current-implementation-and-reproducibility)
13. [Limitations and open directions](#13-limitations-and-open-directions)
14. [Conclusion](#14-conclusion)
15. [References](#references)

## 1. Introduction

Vision Transformers usually represent tokens in Euclidean space and compare them through dot products. Hyperbolic models offer a different inductive bias. Their exponential volume growth can accommodate tree-like structure with relatively low distortion, and their radial coordinate provides a natural way to express a progression from general concepts near the origin to specific concepts at greater depth.

HexFormer supplies the backbone for this study. It maps image tokens to the Lorentz model of hyperbolic space, applies Lorentz-compatible transformer operations, and performs classification through a hyperbolic output head. Its baseline attention uses a compatibility derived from Lorentz distance. That compatibility distinguishes near from far, but distance by itself does not say whether a key continues away from the origin in a general-to-specific direction or instead lies toward the origin.

HAA adds this missing directional quantity. For every query-key pair it measures the angle, at the query, between the direction to the key and the direction back to the Lorentz origin. It combines that signal with a depth-dependent cone aperture adapted from the hyperbolic entailment cones of Ganea et al. The resulting attention score has two branches: a spatial term and an entailment term.

The project asks three connected questions. First, does the trained baseline already contain a layer with sufficient radial variation, forward angular orientation, and cone occupancy? Second, if not, can auxiliary objectives and architectural controls actively produce those properties? Third, how do the induced geometric regimes relate to image-classification accuracy?

The work contributes the HAA compatibility, an intrinsic cone condition that implies the intended depth order for valid pairs, a smooth learnable aperture, training mechanisms that target distinct geometric failure modes, and diagnostics that expose those mechanisms during optimization. Empirically, it organizes 20 directional configurations as a progression rather than as an undifferentiated hyperparameter sweep. The completed evidence is restricted to CIFAR-100 and a terminal-layer emphasis, so the conclusions concern the observed mechanism and operating regimes at that scale.

## 2. Background and related work

### 2.1 Hyperbolic representations and hierarchy

In an $n$-dimensional Euclidean space, the volume of a ball grows polynomially with radius. In hyperbolic space it grows exponentially. This matches the expansion of nodes in a tree as depth increases and motivates the use of hyperbolic embeddings for hierarchical data.

Nickel and Kiela demonstrated this principle with Poincaré embeddings learned from observed hierarchical relations. Their model places general nodes closer to the center of the ball and specific nodes closer to its boundary while using hyperbolic distance to preserve the graph structure. The ordering is not a free consequence of the manifold: it is learned from supervised relations. This distinction matters for images, where a flat classification objective does not directly provide token-level hypernym pairs.

In this overview, radial depth is interpreted as a possible general-to-specific coordinate. It is a design target and an empirical hypothesis, not a semantic law of Lorentz representations. Whether a trained image model actually uses depth in this way must be measured.

### 2.2 Lorentz geometry

For curvature $-1/K$, $K>0$, the upper sheet of the Lorentz hyperboloid is

```math
\mathcal L_K^n
=
\left\{x\in\mathbb R^{n+1}:\langle x,x\rangle_{\mathcal L}=-K,\ x_0>0\right\},
```

with Minkowski bilinear form

```math
\langle x,y\rangle_{\mathcal L}
=-x_0y_0+\sum_{r=1}^{n}x_ry_r.
```

The Lorentz origin is $O=(\sqrt K,0,\ldots,0)$. The geodesic distance between two points on the manifold is

```math
d_{\mathcal L}(x,y)
=\sqrt K\,\mathrm{arcosh}\!\left(-\frac{\langle x,y\rangle_{\mathcal L}}{K}\right).
```

The radial depth used throughout the study is the distance from the origin,

```math
\tilde c(x)=d_{\mathcal L}(O,x)
=\sqrt K\,\mathrm{arcosh}\!\left(\frac{x_0}{\sqrt K}\right).
```

The tangent space at $x$ is $T_x\mathcal L_K^n=\{v:\langle x,v\rangle_{\mathcal L}=0\}$. Tangent vectors have positive Lorentz norm $\|v\|_{\mathcal L}=\sqrt{\langle v,v\rangle_{\mathcal L}}$. The logarithmic map $\log_x(y)$ gives the tangent direction of the geodesic from $x$ to $y$; the exponential map returns a tangent update to the manifold. These maps make it possible to define angles intrinsically at a query rather than by comparing ambient Euclidean coordinates.

For numerical work, the maintained code frequently uses the unnormalized tangent direction

```math
\tilde u_{x\to y}=y+\frac{\langle x,y\rangle_{\mathcal L}}{K}x.
```

It is collinear with $\log_x(y)$, so normalization cancels in a cosine. This identity avoids unnecessary inverse hyperbolic functions in the angular branch.

### 2.3 HexFormer

HexFormer is a Vision Transformer formulated in the Lorentz model. Its relevant inherited components are the hyperbolic patch embedding, Lorentz linear projections, Lorentz transformer blocks, exponential-map aggregation, and Lorentz classification head. The thesis uses the Tiny configuration with nine encoder layers, hidden dimension 192, MLP dimension 384, 12 attention heads, and $4\times4$ image patches. A CIFAR-100 image produces 64 patch tokens plus one CLS token.

In the baseline attention path, separate Lorentz projections construct queries and keys. Pairwise compatibility uses HexFormer's `csqdist`, the Lorentz chordal squared-distance expression, and is then divided by a learned attention temperature before softmax. HAA changes this compatibility calculation at selected layers while preserving the surrounding backbone.

### 2.4 Hyperbolic Entailment Cones

Ganea, Bécigneul, and Hofmann introduced hyperbolic entailment cones as a geometric representation of directed partial orders. Each point acts as the apex of a cone oriented away from the origin. A shallow point has a broad cone that can contain many descendants; a deep point has a narrow cone that admits only closely aligned descendants.

In the Poincaré ball, their half-aperture is

```math
\psi(x)=\arcsin\!\left(\beta\frac{1-\|x\|^2}{\|x\|}\right),
```

with a fixed aperture parameter and a radial domain restriction that keeps the expression valid. If $u$ is a proposed parent and $v$ a proposed child, membership compares the angle from $u$ toward $v$ with the forward radial direction. The construction was developed in a setting with observed hierarchical relations and explicit structural constraints.

The Poincaré and Lorentz models are isometric. If $r=\|x\|$ and the corresponding Lorentz depth for $K=1$ is $\tilde c=2\mathrm{atanh}(r)$, then

```math
\frac{1-r^2}{r}=\frac{2}{\sinh\tilde c}.
```

The Poincaré aperture therefore becomes $\arcsin(2\beta/\sinh\tilde c)$; the factor of two can be absorbed into the aperture constant. This yields the Lorentz functional form used by HAA.

The prior construction is directly relevant, but its assumptions cannot be transferred silently. CIFAR-100 supplies class labels and a coarse-to-fine class grouping, not token-level hypernym edges. It also does not guarantee a radially stratified token manifold. The Phase-0 diagnosis therefore tests the geometric preconditions before HAA is trained.

### 2.5 Positioning of HAA

Three parts must be distinguished. HexFormer provides the Lorentz Vision Transformer and distance-based baseline. Hyperbolic entailment cones provide the pre-existing geometric idea of a directed, depth-dependent cone. This project adapts the cone relation to query-key attention, derives an intrinsic Lorentz membership condition, makes the aperture learnable and numerically smooth, and develops the objectives and telemetry used to investigate whether the geometry can form in image classification.

The project does not claim to introduce entailment cones themselves. Its contribution lies in the attention formulation and in the experimental study of how a Lorentz image model responds when directional hierarchy is added without the original form of pairwise hierarchy supervision.

## 3. Problem formulation

### 3.1 Baseline compatibility

Let $Q_i=W_Q(X_i)$ and $K_j=W_K(X_j)$. The current baseline defines

```math
d^2_{\mathrm{cL}}(Q_i,K_j)
=-2K-2\langle Q_i,K_j\rangle_{\mathcal L},
\qquad
s_{\mathrm{dist}}(Q_i,K_j)=-a\,d^2_{\mathrm{cL}}(Q_i,K_j),
```

where $a$ includes the head scaling and learned attention temperature. This distance-based compatibility is symmetric in its two geometric arguments, so

```math
s_{\mathrm{dist}}(q,k)=s_{\mathrm{dist}}(k,q).
```

This statement does not imply that the complete token-index logit matrix satisfies $L_{ij}=L_{ji}$. Queries and keys come from distinct projections, and exchanging token indices changes which projection is applied to each token. The relevant limitation is narrower: proximity alone does not explicitly encode the direction of a hierarchical relation.

### 3.2 Desired directional relation

The design assigns shallow representations the role of general concepts and deep representations the role of specific concepts. For a query $Q_i$ and candidate key $K_j$, the intended forward relation is

```math
\tilde c(K_j)>\tilde c(Q_i)
```

together with angular alignment away from the origin. Greater depth alone is insufficient because a deep point can lie in an unrelated direction. Angular alignment alone is also insufficient because the permitted angular region should narrow as the query becomes more specific.

These conditions define the behavior HAA seeks to encourage. They are not assumed properties of a trained Lorentz transformer. The baseline diagnosis and training telemetry determine whether they hold.

### 3.3 Desired role of HAA

HAA should preserve a spatial preference for nearby query-key pairs while making the forward relation distinguishable from its reverse. It should expose its aperture and branch weights as trainable quantities, remain differentiable near shallow depths, and produce measurable signals that separate genuine directional change from radial or angular collapse. Auxiliary objectives may shape these prerequisites, but the formal HAA modification remains the compatibility score.

## 4. Baseline geometry and Phase-0 diagnosis

Phase 0 asks whether any layer of the trained HexFormer baseline is already suitable for directional cone attention. All measurements are made layer by layer from the baseline checkpoint, before HAA optimization. Four metrics describe complementary properties.

- **M1, radial-depth variance:** unbiased within-image variance of token depth $\tilde c(Q)$. The compatibility threshold is $\sigma^2_{\tilde c}>0.100$. A nearly constant depth cannot support a useful general-to-specific ordering.

- **M2, angular signal:** the 50-bin KL divergence from a uniform distribution on $[-1,1]$ is reported as a descriptive measure of angular non-uniformity. The M2 decision criterion is distinct: it requires a bootstrap $p<0.05$ together with a negative mean $Z(Q,K)$.

- **M3, cone occupancy:** fraction of valid pairs satisfying $B(Q_i;\beta=1.0)+Z\leq0$ under the neutral diagnostic setting $\beta=1.0$, with a target interval from 0.15 to 0.35. Zero occupancy means the directional region is unused; near-total occupancy would make it non-selective. This diagnostic setting is separate from the $\beta$ initialization used in later directional training runs.

- **M4, spatial variation:** coefficient of variation of the projected spatial norms. It is descriptive and does not enter the mandatory layer decision.

The observed layer-wise values are:

| Layer | M1 radial variance | Angular KL | Mean Z | M3 occupancy | M4 spatial CV |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00259 | 1.36 | +0.806 | 0 | 0.0518 |
| 1 | 0.00132 | 1.49 | +0.768 | 0 | 0.0377 |
| 2 | 0.00218 | 1.99 | +0.886 | 0 | 0.0498 |
| 3 | 0.00750 | 1.30 | +0.757 | 0 | 0.1070 |
| 4 | 0.01450 | 1.63 | +0.838 | 0 | 0.1060 |
| 5 | 0.00500 | 1.33 | +0.744 | $1.97\times10^{-9}$ | 0.0825 |
| 6 | 0.01680 | 1.43 | +0.761 | 0 | 0.1340 |
| 7 | 0.01030 | 1.44 | +0.775 | 0 | 0.1110 |
| 8 | 0.01920 | 2.02 | +0.874 | 0 | 0.1440 |

Every layer has radial variance far below the M1 threshold. The non-uniformity tests give bootstrap $p<0.001$ at every layer, but every mean $Z$ is strongly positive, opposite to the design target. Cone occupancy is effectively zero throughout. Spatial variation increases in later layers but does not repair the mandatory failures.

Define $k^*$ as a layer satisfying M1, M2, and M3 jointly. The result is

```math
k^*=\mathrm{null}.
```

The conclusion is not that hyperbolic hierarchy is impossible, but that its required radial and directional structure does not simply emerge from this baseline and training objective. This motivates terminal placement as the first controlled intervention and motivates auxiliary mechanisms that act separately on orientation, occupancy, and collapse.

## 5. Hyperbolic Asymmetric Attention

### 5.1 Design objective

For each query $Q_i$, HAA should prefer keys that are spatially compatible and lie in a depth-appropriate forward direction. The spatial branch prevents the score from discarding local geometric organization. The entailment branch distinguishes candidate keys by direction and query depth. Their weights remain separate so that the study can observe how training balances them.

### 5.2 Geometric notation

Let

```math
a=\langle Q_i,K_j\rangle_{\mathcal L},\qquad
b=\langle O,Q_i\rangle_{\mathcal L},\qquad
g=\langle O,K_j\rangle_{\mathcal L}.
```

Write $\tilde c_i=d_{\mathcal L}(O,Q_i)$ for query depth. All reported experiments use $K=1$, although the conceptual definitions retain $K$. A pair is geometrically valid only when both tangent directions needed by the angular cosine have non-negligible norm.

### 5.3 Angular entailment signal Z

The angular signal is the tangent-space cosine at $Q_i$:

```math
Z(Q_i,K_j)
=
\frac{\langle \log_{Q_i}(K_j),\log_{Q_i}(O)\rangle_{\mathcal L}}
{\|\log_{Q_i}(K_j)\|_{\mathcal L}\,\|\log_{Q_i}(O)\|_{\mathcal L}}.
```

The direction toward the origin is the reverse of the intended descendant direction. Consequently, $Z<0$ means the key lies on the forward side of the query, while $Z>0$ means it points toward the origin or a shallower direction.

![Angular signal Z in the tangent space at the query](../assets/angular_signal_z.svg)

*The sign of $Z$ distinguishes keys in the forward, deeper direction from keys pointing toward the origin.*

Using unnormalized tangents gives the equivalent algebraic form

```math
Z
=
\frac{g+ab/K}
{\sqrt{a^2/K-K}\,\sqrt{b^2/K-K}}.
```

This form is central to the maintained implementation because it avoids explicit log maps for every pair. It also makes the degeneracies visible: the denominator vanishes when the query coincides with the origin or when query and key coincide in the relevant tangent direction.

### 5.4 Cone aperture B

The conceptual Lorentz half-aperture is

```math
\psi(\tilde c_i)
=\arcsin\!\left(\frac{\beta}{\sinh(\tilde c_i/\sqrt K)}\right),
```

with active domain $\sinh(\tilde c_i/\sqrt K)\geq\beta$. The maintained smooth formulation below extends the aperture into the shallow or degenerate regime instead of leaving it undefined.

The quantity used in the score is its cosine,

```math
B_{\mathrm{cone}}(Q_i)
=\cos\psi(\tilde c_i)
=\sqrt{1-\frac{\beta^2}{\sinh^2(\tilde c_i/\sqrt K)}}.
```

Near the origin, the cone approaches a hemisphere and $B$ approaches zero. With increasing depth, $B$ approaches one, so membership requires nearly perfect forward alignment. In contrast with the fixed aperture parameter in the original entailment-cone construction, HAA parameterizes $\beta$ as a positive learnable quantity.

The raw square-root argument can be negative in the shallow regime. The maintained default at $K=1$ uses

```math
u_B=1-\frac{\beta^2}{\sinh^2\tilde c_i},\qquad
B(Q_i)=\sqrt{\frac{1}{s}\mathrm{softplus}(s u_B)+10^{-8}},
\quad s=4.
```

This smoothly approximates a zero floor while retaining a gradient for $\beta$ where a ReLU floor would be inactive. A legacy `relu` mode remains available. The `--disable_B` ablation fixes $B\equiv1$, which removes the depth-dependent aperture and makes the membership threshold $Z\leq-1$.

### 5.5 Cone membership and intrinsic depth ordering

Membership is defined by

```math
B(Q_i)+Z(Q_i,K_j)\leq0.
```

Because $B\geq0$, membership requires $Z\leq-B\leq0$. For a valid non-degenerate pair, $Z<0$ implies

```math
d_{\mathcal L}(O,K_j)>d_{\mathcal L}(O,Q_i).
```

The result follows from the hyperbolic law of cosines. The numerator sign of $Z$ implies

```math
\cosh\!\left(\frac{d(O,K_j)}{\sqrt K}\right)
>
\cosh\!\left(\frac{d(Q_i,K_j)}{\sqrt K}\right)
\cosh\!\left(\frac{d(O,Q_i)}{\sqrt K}\right),
```

and the first factor on the right is at least one. Since $\cosh$ is strictly increasing on non-negative arguments, the key must be deeper. Thus the membership rule couples aperture, direction, and depth ordering without a separate norm comparison for every pair.

### 5.6 Entailment term Phi

Let $x=B+Z$. The cone boundary is $x=0$. HAA uses a margin-softplus function

```math
\Phi(x)=\mathrm{softplus}(x-m)-\mathrm{softplus}(-m),
\qquad m=0.1.
```

It satisfies $\Phi(0)=0$, is negative for pairs inside the cone, and is positive outside. Since the attention score contains $-\tau\Phi$, valid in-cone pairs receive a bounded reward while violations are reduced smoothly. Its derivative is $\sigma(x-m)$, avoiding a discontinuous hinge at the cone boundary.

### 5.7 Spatial term H

The spatial branch uses

```math
\mathcal H(d)=\log\cosh(d/\delta_0),\qquad \delta_0=15.
```

The function is quadratic near zero and approximately linear at large distance. The implementation evaluates it stably as

```math
t+\mathrm{softplus}(-2t)-\log2,\qquad t=d/\delta_0,
```

rather than forming $\cosh(t)$ directly. Before this transformation, distance is smoothly capped by $d_{\mathrm{soft}}=40\tanh(d/40)$.

### 5.8 Final HAA score

The pre-softmax score is

```math
\mathrm{Score}_{ij}
=
-\lambda\,\mathcal H\!\left(d_{\mathcal L,\mathrm{soft}}(Q_i,K_j)\right)
-\tau\,\Phi\!\left(B(Q_i)+Z(Q_i,K_j)\right).
```

The spatial term retains distance sensitivity. The entailment term introduces direction. The score is then divided by the learned attention temperature and passed to softmax in the same attention pipeline as the baseline.

![HAA pre-softmax score computation](../assets/haa_score.svg)

*The HAA score combines the spatial and entailment branches before the attention softmax.*

### 5.9 Learnable beta, lambda, tau and attention temperature

The aperture and branch weights are kept positive with softplus parameterizations:

```math
\beta=\mathrm{softplus}(\beta_{\mathrm{raw}}),\quad
\lambda=\mathrm{softplus}(\lambda_{\mathrm{raw}}),\quad
\tau=\mathrm{softplus}(\tau_{\mathrm{raw}}).
```

Their requested initial values are converted to raw parameters through the inverse softplus. The default HAA presets initialize $\lambda=1$ and $\tau=1$; the aggressive terminal ablation uses $\lambda=0.3$ and $\tau=3$. The spatial weight can be frozen with `--no-learn_lambda`. The attention temperature is a separate learned scalar initialized to one. It scales the combined logit and should not be conflated with the aperture smoothing factor $s=4$.

### 5.10 Numerical stabilization

The maintained implementation applies safeguards at each sensitive operation:

- Lorentz distance clamps its inverse-hyperbolic-cosine argument to at least $1+10^{-3}$, then uses a differentiable soft cap at 40.

- Angular squared norms are formed as $a^2/K-K$ and $b^2/K-K$, clamped at zero, and receive $10^{-12}$ before their square roots.

- A pair is valid only when both unclamped squared tangent norms exceed $10^{-6}$. Invalid pairs receive $Z=0$ and are excluded from auxiliary reductions.

- The angular cosine is clamped to $[-1,1]$. During training, a NaN element is replaced individually with zero; during evaluation, any NaN angular result raises an error rather than silently altering the metric.

- Depth computations clamp $x_0/\sqrt K$ to at least $1+10^{-3}$. The prototype distance uses a stable closed form for $\mathrm{arcosh}(1+u)$.

These are numerical definitions of the maintained score, not additional modeling claims. The standalone Phase-0 utility deliberately retains the historical diagnostic convention discussed in Section 8.4.

## 6. Training the geometry

### 6.1 Why the HAA score alone is insufficient

C1-C4 show that changing compatibility does not automatically reorganize the token manifold. Applying HAA to all layers reduces accuracy, while terminal score-only HAA returns close to the baseline but leaves the angular signal positive and the cone empty. This is consistent with Phase 0: the score is introduced into a representation that lacks both radial spread and forward orientation. The later objectives therefore target the observable failure modes instead of assuming the classification loss will solve them indirectly.

Auxiliary weights generally follow a warmup, a 25-epoch linear ramp, and a plateau. Warmup start and plateau weight are configurable by mechanism. Scheduling prevents a strong geometric constraint from dominating before the classifier has formed useful features.

### 6.2 Directional objective

The angular objective removes the aperture from its optimization path so that it must change $Z$, not inflate $\beta$. Over valid pairs it computes

```math
s_{\mathrm{ang}}
=\mathrm{mean}\left[\sigma\left(4(-Z-0.05)\right)\right]
```

and applies a squared band hinge,

```math
\mathcal L_{\mathrm{ang}}
=\mathrm{ReLU}(0.55-s_{\mathrm{ang}})^2
+\mathrm{ReLU}(s_{\mathrm{ang}}-0.85)^2.
```

The lower bound encourages a substantial fraction of forward pairs; the upper bound avoids the trivial goal that every pair should occupy the same angular half-space. C11 and C12 show that this signal can drive the final mean $Z$ below zero when combined with prototypes and an appropriate depth control.

### 6.3 Anti-collapse objectives

Directional or occupancy pressure can be satisfied by compressing the representation. Two objectives address that failure.

The radial-variance loss reads the post-$W_Q$ query tensor consumed by HAA. It computes the within-image token-depth population variance for each head, averages over images and heads, and applies

```math
\mathcal L_{\mathrm{radvar}}
=\mathrm{ReLU}(0.10-\sigma^2_{\tilde c})^2.
```

The one-sided form stops pushing once the floor is reached. Supervising post-projection queries is material because earlier representations can retain variation that the query projection removes.

The spread objective combines the same radial floor with a spatial coefficient-of-variation floor:

```math
\mathcal L_{\mathrm{spread}}
=w_r\mathrm{ReLU}(0.10-\sigma^2_{\tilde c})^2
+w_s\mathrm{ReLU}(0.30-\mathrm{CV}_{\mathrm{spatial}})^2.
```

It was introduced after occupancy-focused runs populated the cone while collapsing radial variation. C17 demonstrates the intended anti-collapse role: it sustains both the occupancy target and radial variance near or above its floor.

### 6.4 Cone occupancy objective

The soft occupancy surrogate is

```math
s_{\mathrm{occ}}
=\mathrm{mean}\left[\sigma\left(6(-(\mathrm{sg}(B)+Z)-0.05)\right)\right],
```

where $\mathrm{sg}$ denotes stop-gradient. The loss is

```math
\mathcal L_{\mathrm{occ}}
=\mathrm{ReLU}(0.10-s_{\mathrm{occ}})^2.
```

Detaching $B$ blocks a shortcut in which the optimizer changes the aperture rather than the token geometry. Gradients must flow through $Z$. This protects the intended meaning of occupancy, but does not by itself protect radial spread, which is why the spread objective is needed in C17 and C18.

### 6.5 Frozen hyperbolic prototype metric learning

The prototype objective supplies an external class-conditioned geometric anchor. Superclass directions form a simplex equiangular tight frame. Each fine-class direction is placed within a $\pi/8$ angular cap around its CIFAR-100 superclass direction. Superclass prototypes are shallow and fine prototypes are deeper. The CLS representation is pulled toward the frozen prototype for its fine label:

```math
\mathcal L_{\mathrm{proto}}
=\frac{1}{N}\sum_{n=1}^{N}d_{\mathcal L}^2(h_{\mathrm{CLS}}^{(n)},p_{y_n}).
```

Under MixUp or CutMix, the same mixing coefficient forms a convex combination of the squared Lorentz distances to the two corresponding frozen prototypes. Because the targets do not move, this objective provides an absolute reference that the batch-derived hierarchy loss cannot. It is central to the C8-C12 metric-learning family and is also reused in the C19-C20 aperture study.

At thesis time, superclass depth was $d_s=0.3$, while each fine class received a class-specific depth offset sampled once from $[0.5,1.85]$ and then frozen. The maintained implementation is deterministic: every fine class uses $d_f=1.175$, so every fine prototype has absolute depth 1.475. Angular construction and the superclass-to-fine grouping are preserved, but the maintained C9 and C12 presets are therefore analogues rather than exact restorations.

### 6.6 Other experimental objectives

The hyperbolic hierarchy loss (HHL) uses batch class means and a margin of 0.3 to encourage a superclass mean depth below the corresponding fine-class mean depth. The two quantities come from the same minibatch, so the signal is weaker and noisier than a frozen external anchor. It plays a supporting role in C5 and C6, not the same role as the prototype metric loss.

Later experiments also target the CLS row directly. The direction loss applies a one-sided squared hinge to the valid-key mean $Z$ for each CLS query, with target -0.05. The CLS-variance loss places a floor of 0.05 on the across-key variance of that row. C19 and C20 combine these terms with prototypes and a CLS-depth residual to study the cone aperture.

The code additionally contains a beta-cap objective, an experimental hyperbolic entailment-cone loss with selectable negatives, and a prototype-softmax classifier. These are available research mechanisms, but they are not principal components of the main reported operating regimes or the five curated presets.

## 7. Architectural depth controls

### 7.1 Per-head query-depth scaling

At an HAA layer, the optional query-depth MLP acts after $W_Q$ and before the HAA score. For the CLS query of each head, it predicts

```math
\alpha_q=0.1+1.4\,\sigma(\mathrm{MLP}(q_{\mathrm{spatial}})),
\qquad 0.1<\alpha_q<1.5.
```

If $q=(q_0,q_s)$, it scales $q_s' = \alpha_q q_s$ and reconstructs $q_0'=\sqrt{K+\|q_s'\|^2}$. The result remains on the Lorentz manifold. Bias initialization gives an initial scale of approximately 0.94, which is near identity. Since the operation occurs inside attention, it changes the query depth seen by $B$, $Z$, and the HAA score without directly rescaling the post-encoder CLS used by the classifier.

### 7.2 CLS-depth residual

The CLS-depth residual acts after the encoder and before final normalization and classification. It predicts the same bounded form

```math
\alpha_{\mathrm{CLS}}=0.1+1.4\,\sigma(\mathrm{MLP}(h_{\mathrm{CLS,spatial}}))
```

and reconstructs the Lorentz time coordinate after scaling the CLS spatial component. Its initial value is also approximately 0.94, near identity. It gives prototype and CLS-focused objectives a direct radial degree of freedom at the representation consumed by the classification head.

### 7.3 Architectural roles

The controls act at different locations and solve different problems. Query-depth scaling changes the geometry consumed by HAA, independently for each head. The CLS-depth residual changes the post-encoder representation used by metric learning and classification. The maintained architecture permits both controls to be enabled together.

The historical C9 residual used an earlier $1+0.8\tanh(\cdot)$ parameterization with range $(0.2,1.8)$ and exact unit initialization. Later thesis configurations used the bounded sigmoid form. The maintained C9 and C12 analogues use the bounded implementation, one reason their public presets should be interpreted by role and mechanism combination rather than as byte-for-byte historical runs.

## 8. Geometric telemetry and diagnostics

### 8.1 Ordinary HAA telemetry

Ordinary telemetry is recorded during the same per-epoch evaluation pass used for classification metrics. For each active HAA layer it reports the learned $\beta$, $\lambda$, and $\tau$; mean $Z$; hard cone occupancy; fraction of queries near the origin; invalid-pair and NaN rates; and near-degenerate query-key rates. It also records attention entropy for the CLS row and patch rows, CLS-row angular mean and variance, and CLS-row occupancy.

When a depth control is active, telemetry adds aggregate scale statistics and gradients. Query-depth scaling reports mean, standard deviation, minimum, and maximum $\alpha_q$, plus the associated gradient norm. The CLS-depth residual reports the corresponding $\alpha_{\mathrm{CLS}}$ statistics. Gradient probes for the aperture and depth paths help distinguish a saturated signal from a component that is still trainable. Training also records pre- and post-query-projection radial variance, which is useful for detecting a projection that erases variation present at its input.

These values serve different purposes. Mean $Z$ measures orientation, occupancy applies the full $B+Z\leq0$ criterion, radial variance measures usable depth diversity, and spatial CV measures non-radial spread. None can substitute for the others.

### 8.2 Deep diagnostics

With `--deep_diagnostics`, the training entry point performs an additional full pass over the evaluation loader at epochs 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, and the final epoch. For every active HAA layer it aggregates five quantities: projected-query radial-depth variance, mean angular signal, 50-bin angular KL divergence, spatial-norm coefficient of variation, and attention entropy.

The extra pass is scheduled separately from ordinary evaluation and should be enabled deliberately when longitudinal geometry matters. It is scheduled rather than run every epoch because it adds evaluation and computation cost. The deep metrics are written to TensorBoard and included in the run CSV.

### 8.3 TensorBoard

TensorBoard is the maintained logging workflow. Scalar series include losses, classification accuracy, learning rate, auxiliary schedules, ordinary HAA telemetry, and any enabled deep diagnostics. Logs are stored below the configured `--log_dir` in a timestamped run directory.

```bash
tensorboard --logdir runs
```

### 8.4 Phase-0 utility

`run_phase_0.py` is a standalone CUDA utility for applying the historical layer-wise diagnosis to a CIFAR-100 checkpoint without changing model source. It registers forward hooks, processes the official evaluation split, and writes JSON measurements plus a PNG summary. It requires an explicit checkpoint:

```bash
python run_phase_0.py \
  --checkpoint output/best_cifar100_baseline.pth \
  --data-root data \
  --output-dir output/phase0 \
  --device cuda:0
```

This utility is not an alternative classifier evaluation path. It implements the historical Phase-0 definitions, including their diagnostic epsilon and legacy aperture floor, so its purpose is to reproduce the layer selection analysis. Ordinary HAA telemetry uses the maintained safe-$Z$ and smooth-aperture implementation described in Section 5.10.

### 8.5 Interpreting geometric signals

Accuracy alone cannot diagnose HAA because different degeneracies can yield similar classifier performance. A non-zero occupancy can arise from useful forward ordering, from a broad aperture near the origin, or from collapsed representations. A negative mean $Z$ establishes average forward orientation but does not establish sufficient radial variance. Conversely, a healthy radial variance does not make a positive mean $Z$ directional in the intended sense. The joint reading of $Z$, occupancy, radial variance, spatial CV, near-origin fraction, and attention entropy is therefore part of the method rather than an optional visualization.

## 9. Experimental setup

### 9.1 Dataset and evaluation split

Training uses the 50,000-image CIFAR-100 training split. The official 10,000-image evaluation split is used for per-epoch model selection and final evaluation. Images are RGB with resolution $32\times32$, and the task has 100 fine classes grouped into 20 superclasses.

### 9.2 Model architecture

All completed experiments use Tiny HexFormer: nine transformer layers, hidden dimension 192, MLP dimension 384, 12 heads, patch size 4, and Lorentz curvature parameter $K=1$ for both encoder and decoder. The token sequence contains 64 patch tokens and one CLS token. HAA experiments generally activate the directional score only at layer 8, the terminal zero-indexed layer, after C1 and C2 established that score-only injection across all layers was unfavorable.

### 9.3 Training protocol

Each run is trained for 100 epochs with batch size 128 and evaluation batch size 512. Optimization uses RiemannianAdamW, initial learning rate $4.35\times10^{-3}$, weight decay 0.05, and a single-cycle cosine schedule to a minimum learning rate of $10^{-6}$. There is no optimizer warmup in the curated CIFAR-100 protocol. The classification loss uses label smoothing 0.1. Computation is float32 on CUDA. All thesis runs use seed 1.

Auxiliary losses have their own warmup and ramp schedules. Unless overridden, angular and occupancy ramps begin at epoch 15, prototype, radial, spread, and HHL ramps begin at epoch 5, and each reaches its configured plateau over 25 epochs. These schedules change the loss weights, not the base learning-rate schedule.

### 9.4 Data augmentation

The image pipeline applies random horizontal flip, random crop to $32\times32$ with four pixels of padding, the CIFAR AutoAugment policy, CIFAR-100 normalization, and random erasing with probability 0.25. Repeated augmentation sampling uses a length factor of three.

At batch level, half of minibatches remain unmixed. The other half is divided equally between CutMix and MixUp, giving probabilities 0.25, 0.25, and 0.50 for CutMix, MixUp, and no mixing. Label-smoothed classification and the appropriate mixed-label criterion are used in their respective branches.

### 9.5 Comparison design

The baseline is a re-training of Tiny HexFormer under the common protocol. C1-C4 compare score placement and strength. Later families alter selected auxiliary or architectural components while retaining terminal HAA. Some pairs are controlled single-variable ablations: C11 versus C12 removes radial variance; C17 versus C18 removes query-depth scaling; C19 versus C20 disables the aperture. Other comparisons, such as C15 versus C17 or C9 versus C12, are descriptive because more than one mechanism differs.

Accordingly, the results support direct statements about observed configurations and the controlled pairs. They do not support assigning every accuracy difference to a single mechanism. In particular, C9 combines angular, prototype, radial, and residual components.

### 9.6 Model selection and evaluation

After each epoch, top-1 accuracy on the official evaluation split is measured. The checkpoint with the highest per-epoch top-1 value is retained as the best model. At the end of training, both the final weights and the reloaded best checkpoint are evaluated on that same official split. Reported thesis accuracies are the best top-1 values from this procedure.

No dataset or checkpoint is distributed with the repository. Evaluation-only mode in the maintained entry point requires `--load_checkpoint`; it does not silently evaluate random weights.

### 9.7 Metrics

Classification metrics are top-1 and top-5 accuracy plus cross-entropy loss. Geometric analysis uses radial-depth variance, mean $Z$, angular-distribution KL, cone occupancy, spatial CV, near-origin fraction, attention entropy, CLS-row statistics, depth-scale statistics, and selected gradient norms. The full experimental comparison relies on both metric groups.

## 10. Experimental progression and results

### 10.1 Full baseline + C1-C20 table

The following table reports the thesis experiments. The five maintained public presets discussed in Section 12 are not a replacement for this study.

| Run | Family | Main intervention | Best top-1 |
|---|---|---|---:|
| Baseline | Reference | HexFormer distance compatibility | 75.49% |
| C1 | Score placement | HAA at all nine layers, uniform initialization | 73.46% |
| C2 | Score placement | HAA at all nine layers, depth-scaled aperture initialization | 73.51% |
| C3 | Score placement | Terminal HAA score only | 75.41% |
| C4 | Score strength | Terminal score, $\tau_0=3.0$, $\lambda_0=0.3$ | 74.94% |
| C5 | Initial occupancy | Occupancy + HHL | 75.28% |
| C6 | Initial occupancy | Occupancy + HHL, stronger auxiliary weights | 75.96% |
| C7 | Metric learning | Angular + radial variance + CLS-depth residual | 75.72% |
| C8 | Metric learning | Angular + prototype + CLS-depth residual | 76.34% |
| **C9** | Metric learning | Angular + prototype + radial variance + CLS-depth residual | **76.56%** |
| C10 | Prototype depth | C9 mechanisms with prototypes re-anchored shallow | 75.15% |
| C11 | Angular alignment | Angular + prototype + radial variance + bounded CLS-depth residual | 75.55% |
| C12 | Angular alignment | C11 with radial-variance objective removed | 76.37% |
| C13 | Occupancy | Prototype-softmax head + occupancy | 67.09% |
| C14 | Occupancy | Occupancy + CLS-depth residual | 75.16% |
| C15 | Occupancy | Occupancy without CLS-depth residual | 75.77% |
| C16 | Occupancy | Occupancy with spatial weight frozen near zero | 74.10% |
| C17 | Query depth | Occupancy + spread + per-head query-depth scaling | 75.77% |
| C18 | Query depth | C17 with query-depth scaling removed | 74.82% |
| C19 | Aperture | CLS direction + CLS variance + prototype + CLS-depth residual | 75.43% |
| C20 | Aperture | C19 with $B\equiv1$ | 74.90% |

### 10.2 C1-C4: score and placement

C1 and C2 place HAA in all nine layers and finish about two percentage points below the baseline. Their terminal mean angular signals remain near +0.97 and +0.98, with empty cones. Distributing a directional pressure across layers that do not meet the Phase-0 prerequisites neither forms the geometry nor preserves baseline accuracy.

C3 restricts HAA to the terminal layer and reaches 75.41%, close to 75.49%. The cone remains empty and $Z$ remains positive. C4 increases entailment weight and reduces spatial weight, but reaches 74.94%, with mean $Z$ near +0.96 and peak occupancy below $3\times10^{-5}$. When the entailment contribution is nearly constant across keys, increasing its magnitude largely cancels under softmax. The observation motivates terminal placement plus explicit geometric supervision, not a still larger score coefficient.

### 10.3 C5-C6: first cone occupancy

C5 adds occupancy and HHL and is the first run to sustain non-zero occupancy, approximately 1.4%, while reaching 75.28%. The occupancy appears through a degenerate route: the near-origin query fraction becomes one from epoch 18 and the aperture opens. C6 raises the auxiliary weights and reaches 75.96%. These runs show that the score-only stall can be broken, but also reveal that an occupancy objective needs protection against shortcuts and collapse.

### 10.4 C7-C10: metric/prototype learning

C7, C8, and C9 form an approximate additive sequence. With angular loss and the CLS-depth residual held in common, radial variance alone gives C7 at 75.72%, +0.23 points over baseline. Prototype loss gives C8 at 76.34%, +0.85 points. Combining both gives C9 at **76.56%**, +1.07 points. The two individual gains sum to 1.08 points, close to the combined improvement.

C9 nevertheless finishes with mean $Z=+0.342$, minimum $+0.0585$ at epoch 30, final occupancy $1.19\times10^{-5}$, and peak occupancy $3.15\times10^{-3}$. The near-origin fraction remains zero. Its best accuracy is therefore not accompanied by the intended occupied, negative-mean cone. Radial-variance telemetry was not captured for C9, so its late radial health cannot be inferred.

C10 moves the prototype target to the shallow superclass depth. It reaches 75.15% and drives final mean $Z$ to $-0.054$, but removes the deeper class-specific target. This outcome supports, without proving, the interpretation that the fine-class metric anchor is important to the C9 gain.

### 10.5 C11-C12: angular alignment

C11 and C12 form the principal angular-alignment family under the bounded residual. C11 reaches 75.55%, crosses zero at epoch 20, reaches a minimum of $-0.112$, and finishes at $-0.074$. C12 differs by removing radial-variance loss. It reaches 76.37%, crosses zero at epoch 20, reaches $-0.099$ at epoch 52, and finishes at approximately $-0.080$. C10 also finishes negative after shallow prototype re-anchoring, but at lower accuracy and under a different intervention.

The angular transition is coordinated with collapse. For C12, radial variance falls from 0.266 at epoch 10 to $1.3\times10^{-4}$ at epoch 20, while spatial CV falls from 0.661 to 0.015. Occupancy peaks at 0.0506 at epoch 33. C11 shows the same pattern, with radial variance falling from 0.209 to $2.98\times10^{-5}$, spatial CV from 0.615 to 0.009, and occupancy peaking at 0.0489. The result demonstrates sustained angular control, but not a joint solution to orientation and spread.

### 10.6 C13-C16: occupancy and collapse

C13-C16 directly target occupancy under different surrounding choices. All settle near occupancy 0.07 and all exhibit collapsed depth or spatial structure:

| Run | Best top-1 | Final occupancy | Final mean Z | Late radial variance | Minimum spatial CV |
|---|---:|---:|---:|---:|---:|
| C13 | 67.09% | 0.071 | -0.012 | 0.008 | 0.110 |
| C14 | 75.16% | 0.066 | -0.003 | 0.004 | $1.4\times10^{-3}$ |
| C15 | 75.77% | 0.068 | -0.022 | 0.003 | $1\times10^{-4}$ |
| C16 | 74.10% | 0.056 | -0.067 | 0.011 | 0.104 |

C15 briefly exceeds the 0.10 occupancy target at epoch 34 while its near-origin fraction is one from epochs 27 through 45. C14 instead collapses spatial variation. C16, with the spatial branch frozen near zero, is weak in both accuracy and radial health. C13 changes the classifier to fixed-prototype softmax and is confounded by head and optimization differences; its 67.09% should not be read as an isolated HAA effect. The family establishes that occupancy is trainable, but that occupancy alone is an incomplete objective.

### 10.7 C17-C18: query-depth ablation

C17 adds both spread protection and per-head query-depth scaling. It sustains occupancy 0.117, at or above the target from epoch 48 onward, while radial variance never falls below 0.096 and spatial CV remains between approximately 0.30 and 0.75. Accuracy reaches 75.77%. Its final mean $Z=+0.415$, so it realizes non-collapsed occupancy without realizing negative average orientation.

C18 is the controlled counterpart with query-depth scaling disabled. The spread loss blocks collapse, but occupancy stays near $10^{-6}$, mean $Z$ rises to +0.944, and accuracy is 74.82%. Within this pair, query-depth scaling is decisive for opening a non-collapsed occupancy regime.

C15 and C17 both reach 75.77% through different geometries. C15 has mildly negative mean $Z$ on collapsed depth; C17 has positive mean $Z$ on preserved depth. Because they also differ in spread loss and query scaling, they are descriptive examples of equal accuracy rather than a one-variable comparison.

### 10.8 C19-C20: aperture ablation

C19 combines CLS-row direction and variance objectives, prototypes, and the CLS-depth residual. It reaches 75.43%, radial variance 0.075, and a small occupancy of $1.5\times10^{-4}$. C20 changes only the aperture by forcing $B\equiv1$. It reaches 74.90%, a decrease of 0.53 points; occupancy is zero, the near-origin fraction becomes one after epoch 67, and radial variance falls to 0.0045.

This controlled pair is consistent with the aperture acting as a useful gradient channel for depth. It does not show that discrete in-cone routing caused the accuracy difference, because C19 itself has very small occupancy and the aperture removal coincides with radial collapse.

### 10.9 Joint operating regimes

The study identifies three principal regimes:

- **Classification regime, C9:** highest top-1 accuracy, prototype-anchored, with positive mean $Z$ and an effectively empty cone.

- **Angular-alignment regime, C12:** 76.37%, only 0.19 points below C9, with the strongest sustained negative mean among the main reported configurations, but collapsed radial and spatial variation.

- **Non-collapsed occupancy regime, C17:** occupancy 0.117 with preserved spread, but positive mean $Z$ and baseline-level accuracy.

No reported configuration simultaneously maximizes classification, negative angular orientation, cone occupancy, and radial health. That outcome is the basis for the analysis below.

## 11. Analysis and interpretation

### 11.1 Geometric properties can be actively shaped

The experiments move geometric quantities by large, targeted amounts. C11 and C12 reverse the sign of mean $Z$. C13-C17 raise occupancy from effectively zero to meaningful fractions. C17 preserves radial and spatial spread while doing so. C19-C20 expose an aperture-dependent depth effect. These are active changes induced by specific objectives and controls, not properties already present in Phase 0.

The qualification is joint behavior. Negative orientation can coincide with collapse, and healthy occupancy can coexist with positive mean orientation. The evidence supports controllability of individual properties, not complete realization of every desired property at once.

### 11.2 Classification improvement

The best directional configuration, C9, reaches **76.56% top-1 accuracy**. The re-trained HexFormer baseline reaches **75.49%**, so the observed improvement is **+1.07 percentage points**. This is the strongest classification outcome of the experimental program.

The result belongs to the full C9 configuration: terminal HAA, angular supervision, frozen fine-class prototypes, radial-variance supervision, and the CLS-depth residual. It should not be rewritten as a score-only HAA effect.

### 11.3 Geometry and classification do not necessarily move together

C9, C12, and C17 separate the measured axes. C9 has the best accuracy. C12 has the strongest sustained negative final angular signal among the main reported runs and is only 0.19 points below C9. C17 has the strongest sustained non-collapsed occupancy. Since those outcomes occur in different configurations, a single scalar such as occupancy cannot predict classification quality across the study.

This does not make either accuracy or geometry secondary. It shows that HAA creates several operating regimes and that the relation between its internal geometry and the classifier is not monotonic under the tested interventions.

### 11.4 Interpretation of the C9 gain

The metric-learning family supplies the strongest evidence about C9. C8, which adds prototypes without radial variance, gains 0.85 points; C7, which adds radial variance without prototypes, gains 0.23; C9 combines them for 1.07. C10 loses the gain when the class target is re-anchored shallow. Together with C9's empty cone and positive mean $Z$, these observations make prototype-based representation regularization a plausible main source of the improvement.

That explanation remains an interpretation. The experiments do not isolate the prototype objective outside the HAA configuration with every other factor matched, and they do not exclude the possibility that the same objective changes how the model uses continuous entailment logits without producing hard occupancy.

### 11.5 What remains unresolved

The present comparison does not separate all interactions among HAA, prototype geometry, radial variance, and the residual. It also does not establish whether hard cone membership is the right statistic for how the attention head uses the entailment term. Matched multi-seed studies, complete telemetry for the best run, and interventions that hold prototypes fixed while varying the directional branch would sharpen attribution.

## 12. Current implementation and reproducibility

### 12.1 Maintained public presets

The repository provides five CIFAR-100 entry points:

| Preset | Role | HAA settings | Active additions | Deep diagnostics |
|---|---|---|---|---|
| `classification_vit/config/cifar100_baseline.txt` | Public baseline | `baseline` | None | Off |
| `classification_vit/config/cifar100_terminal_score_only.txt` | Closest current role to C3 | `terminal`, $\beta_0=1$ | None | Off |
| `classification_vit/config/cifar100_c9_current_analogue.txt` | Current C9 analogue | `terminal`, $\beta_0=2$ | Angular 0.3, prototype 0.5, radial variance 0.5, CLS-depth residual | Off |
| `classification_vit/config/cifar100_c12_current_analogue.txt` | Current C12 analogue | `terminal`, $\beta_0=2$ | Angular 0.3, prototype 0.5, CLS-depth residual | On |
| `classification_vit/config/cifar100_c17_equivalent.txt` | Closest maintained C17 equivalent | `terminal`, $\beta_0=2$ | Occupancy 0.5, spread 0.5, query-depth MLP | On |

All four directional presets use initial $\lambda=1$, initial $\tau=1$, and softplus aperture smoothing with factor 4.

The five presets are not the complete thesis study and should not be counted as five of twenty archival run files. The full evidence remains the baseline plus C1-C20 table in Section 10.

### 12.2 Material thesis-time vs maintained differences

The public baseline retains the thesis model and training settings apart from configurable paths, output locations, and other infrastructure changes. The current terminal score-only preset has the closest experimental role to C3, but uses the maintained smooth aperture and safe angular implementation rather than restoring the exact earlier source state.

The C9 and C12 presets preserve their named auxiliary mechanisms, but historical prototypes used a frozen class-specific sampled fine depth. Current prototypes use a common deterministic fine depth. Historical C9 also used the earlier tanh CLS residual and supervised radial variance at an earlier captured representation; the maintained preset uses the bounded residual and post-$W_Q$ within-image radial variance. C17's occupancy, spread, and per-head query-depth combination is logically preserved by its public preset.

These differences affect exact numerical reproduction. The thesis values in Section 10 are historical results, not claimed outputs of rerunning the maintained analogues.

### 12.3 Configuration reference

Configuration files use `configargparse` and can be loaded with `-c` or `--config_file`. Command-line values override file values. The current options most relevant to the documented workflows are:

| Area | Current options |
|---|---|
| Run and paths | `--config_file`, `--exp_name`, `--output_dir`, `--log_dir` |
| Execution | `--device`, `--dtype`, `--seed`, `--load_checkpoint`, `--eval_only`, `--deep_diagnostics` |
| Training | `--num_epochs`, `--batch_size`, `--batch_size_test`, `--lr`, `--weight_decay`, `--optimizer`, `--warmup` |
| Model | `--model_size`, `--patch_size`, `--num_layers`, `--hidden_dim`, `--mlp_dim`, `--num_heads`, `--encoder_k`, `--decoder_k` |
| Data | `--dataset`, `--data_root`, `--tiered_lmdb_root` |
| HAA score | `--haa_mode`, `--haa_tau_init`, `--haa_lambda_init`, `--learn_lambda` / `--no-learn_lambda` |
| Aperture | `--B_smooth`, `--B_softplus_temp`, `--beta_init_override`, `--disable_B` |
| Direction and hierarchy | `--gamma_angular_max`, `--gamma_angular_warmup`, `--eta_max`, `--eta_warmup` |
| Prototype and radial | `--eta_proto_max`, `--zeta_radvar_max`, `--sigma2_target`, `--d_s`, `--d_f_mid`, `--proto_seed` |
| Occupancy and spread | `--phi_occ_max`, `--occ_s_target`, `--occ_kappa`, `--occ_m_smooth`, `--omega_spread_max`, `--spread_sigma2_target`, `--spread_cv_target` |
| Depth controls | `--use_q_depth_mlp`, `--use_cls_depth_residual` |
| CLS objectives | `--chi_cls_dir_max`, `--cls_dir_z_star`, `--psi_cls_var_max`, `--cls_var_sigma2_star` |
| Other experimental paths | `--xi_betacap_max`, `--use_hec_loss`, `--hec_weight`, `--hec_negative_mode`, `--use_proto_softmax`, `--proto_T_init` |

`--haa_mode` accepts `baseline`, `terminal`, `full_uniform`, `continuous`, and `terminal_aggressive`. CUDA device strings may name one device or a comma-separated set. CPU execution is rejected by the maintained training and Phase-0 paths.

### 12.4 CLI overrides

The following command starts from the C17-equivalent preset and overrides portable locations plus the occupancy plateau:

```bash
python classification_vit/train.py \
  --config_file classification_vit/config/cifar100_c17_equivalent.txt \
  --data_root /path/to/cifar100 \
  --output_dir output/c17_variant \
  --log_dir runs/c17_variant \
  --phi_occ_max 0.6 \
  --deep_diagnostics
```

Boolean flags are enabled by presence. The spatial weight is the exception with an explicit negative form: `--no-learn_lambda` freezes it.

### 12.5 Running experiments

Install the pinned requirements and run from the repository root:

```bash
python -m pip install -r requirements.txt
python classification_vit/train.py \
  --config_file classification_vit/config/cifar100_baseline.txt
```

A representative maintained HAA run is:

```bash
python classification_vit/train.py \
  --config_file classification_vit/config/cifar100_c12_current_analogue.txt
```

CIFAR-100 is obtained through `torchvision` at the configured `--data_root`. Outputs include best and final checkpoints, an epoch metrics CSV, and a best-metrics text record when `--output_dir` is set. No checkpoints or datasets are bundled.

### 12.6 Evaluation and checkpoints

Evaluation-only mode requires a checkpoint and uses the same configured model construction:

```bash
python classification_vit/train.py \
  --config_file classification_vit/config/cifar100_c12_current_analogue.txt \
  --eval_only \
  --load_checkpoint /path/to/checkpoint.pth
```

The checkpoint must match the configured model and mechanism combination. A mismatched config can fail strict state loading or produce an invalid comparison.

### 12.7 TensorBoard

Use the configured log root:

```bash
tensorboard --logdir runs
```

For HAA experiments, this exposes classification curves and the geometric signals discussed in Section 8. Scheduled deep diagnostics are absent unless `--deep_diagnostics` is enabled.

## 13. Limitations and open directions

The completed study is limited to CIFAR-100. Its 32-pixel images, 65-token sequence, and two-level class grouping provide a useful controlled setting but do not establish behavior on a larger visual hierarchy. A tieredImageNet experiment and an LMDB-oriented input path were prepared, but the run was not completed because the available shared-storage input pipeline could not sustain the required I/O. No accuracy or geometric result is reported for tieredImageNet.

Every thesis run uses one seed. The reported ordering is therefore descriptive and carries no across-seed variance estimate, especially for the 0.19-point difference between C9 and C12. Multi-seed matched comparisons are needed to establish the stability of the accuracy ordering.

The main auxiliary experiments inject HAA at the terminal layer. All-layer score-only variants were tested, but intermediate or multi-layer injection together with the successful auxiliary objectives was not. Varying this placement would determine whether the observed operating regimes are specific to terminal aggregation.

Telemetry was introduced incrementally. Radial variance is unavailable for C9, C5, and C6; the CLS-row metrics intended for C19 and C20 were not captured in those historical runs; and query-depth scales were logged as aggregate statistics rather than per-head trajectories. These gaps limit particular mechanistic claims and are represented as missing evidence rather than filled by inference.

Finally, the current experiments do not isolate every source of the C9 improvement. Useful next comparisons include a multi-seed factorial study of prototype, radial, residual, and directional components; matched hard-occupancy and continuous-logit probes; a larger hierarchy with completed input infrastructure; and terminal versus intermediate injection under identical auxiliary supervision.

## 14. Conclusion

This project develops a directional Lorentz attention score from spatial distance and entailment geometry, then studies the conditions required for that score to operate in a Vision Transformer. Phase 0 shows that the required structure is absent from the re-trained baseline. The subsequent 20 directional configurations demonstrate that angular orientation, occupancy, and spread can be actively modified, although their strongest forms occur in different regimes.

The best configuration reaches 76.56% top-1 accuracy, improving the 75.49% HexFormer baseline by 1.07 percentage points. C12 reaches 76.37% with sustained negative angular alignment, while C17 sustains 11.7% non-collapsed occupancy. Together, these results establish both a classification improvement and a substantive geometric finding: controllable HAA geometry and maximum classification accuracy do not reduce to one common operating point in the present study. Better causal attribution, repeated runs, varied injection depth, and completed large-scale evaluation remain open.

## References

1. Haya Alyoussef, Ahmad Bdeir, Diego Coello de Portugal Mecke, Tom Hanika, Niels Landwehr, and Lars Schmidt-Thieme. "HexFormer: Hyperbolic Vision Transformer with Exponential Map Aggregation." International Conference on Learning Representations, 2026. arXiv:2601.19849.

2. Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully Hyperbolic Convolutional Neural Networks for Computer Vision." International Conference on Learning Representations, 2024.

3. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." International Conference on Learning Representations, 2021.

4. Octavian-Eugen Ganea, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic Entailment Cones for Learning Hierarchical Embeddings." Proceedings of the 35th International Conference on Machine Learning, Proceedings of Machine Learning Research 80, 2018.

5. Alex Krizhevsky. "Learning Multiple Layers of Features from Tiny Images." Technical Report, University of Toronto, 2009.

6. Marc Law, Renjie Liao, Jake Snell, and Richard Zemel. "Lorentzian Distance Learning for Hyperbolic Representations." Proceedings of the 36th International Conference on Machine Learning, Proceedings of Machine Learning Research 97, pages 3672-3681, 2019.

7. Pascal Mettes, Elise van der Pol, and Cees G. M. Snoek. "Hyperspherical Prototype Networks." Advances in Neural Information Processing Systems 32, 2019.

8. Maximilian Nickel and Douwe Kiela. "Poincaré Embeddings for Learning Hierarchical Representations." Advances in Neural Information Processing Systems 30, 2017.
