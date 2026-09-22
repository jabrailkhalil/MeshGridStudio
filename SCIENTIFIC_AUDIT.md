# Scientific audit: MeshGridStudio

**Baseline:** main `9b18165ba00d325bc98b9c9189cfba4efae2a0c6`, release 1.2.1.
**Scope:** source, manuscript, numerical outputs, tests and supplementary packaging.
Corrections are research-source changes, not a new verified Windows release.

## Confirmed defects and corrections

| Issue | Evidence in baseline | Correction / regression |
|---|---|---|
| False identification and priority claim for the 3D shape energy | `article.tex` called `||A||²/J^(2/3)` a new generalization; Knupp (2001), Table 2, gives inverse mean ratio and distinguishes Winslow | Correct method names and literature; preserve Python `generate_winslow_3d` only as a legacy API. The inverse-harmonic energy `||cof A||²/J` is not implemented in this experiment. |
| 3D volume was an inexact corner trapezoid | Twisted cube, 11³: old 7.538886023361551 versus exact `(8/3)*(2+cos(.6)) = 7.534228306425808` | Exact-in-arithmetic 2×2×2 Gauss integration of each trilinear cell. Analytic tests across resolutions/twists and independent five-point quadrature. Volume-derived metrics and normalization are regenerated. |
| Incorrect cell-shape measurement | Code used one corner in 3D while text said center; global-coordinate derivatives gave AR=1 for a cube with unequal cell widths | Center, cell-local Jacobian and SVD. Cube with 8×6×7 nodes has AR=1.4. Two-dimensional anisotropic-count semantics corrected too. |
| Corner positivity was overinterpreted | A stored hexahedron has minimum corner J=+0.0527771 but a sampled noncorner J=-0.1564756 | Add post-solve 27-point diagnostics and an explicit counterexample. **Not a full validity certificate**; see remaining limitation below. |
| Jacobian threshold had inconsistent units | Nonlinear wrappers compared a whole-domain derivative with a per-cell floor, while the objective used the derivative floor | Consistent derivative-unit threshold at initialization and post-solve diagnostics; normalized-coordinate optimization. |
| Edge compatibility check could miss defects | 33 arbitrary samples can miss a kink in a finer piecewise-linear shared edge | Check the union of both edge knot sets; regression with a 65-knot hidden kink. |
| Nonfinite parameter/gradient handling | NaN tolerance or gradient could evade ordinary comparisons | Reject nonfinite tolerances, weights, barrier parameters and initial/trial gradients in the audited paths. |
| Formula in module docstring had a factor-of-two ambiguity | `-mu ln(det D)` in prose, `-mu ln(J/J*)` in code | Correct documentation, retaining the implemented and manuscript-defined latter barrier. |
| Supplementary archive omitted 3D | Old ZIP contained no 3D kernel, tests or numerical results and used an incompatible split layout | New root-layout packager, complete 3D files, saved grids, SHA-256 manifest and verification script. |
| Artifact / manuscript drift | CSV and TeX runtimes differed; documentation said 43 tests while baseline discovery ran 45; author list was incomplete | Regenerate outputs, generate 3D numerical discussion from CSV, record actual executed test count, restore Tishkin–Yashina–Khalilov author order for coauthor review. |

The new regression module adds 18 tests to the 45 baseline tests. Exact outcomes,
software versions, sampled derivative checks and source hashes are stored in
`output/audit/verification.json` and `output/audit/tests.txt`, not inferred from
README claims. No existing accuracy test was relaxed to hide a failure: the
analytic twisted-volume tolerance was tightened from 0.01 to 2e-12, and the
anisotropic-cube shape assertion was corrected from 1 to 1.4.

## What the fixes change numerically

For the ball's known inverse-mean-ratio run, the old corner-based AR95 was
approximately 2.7523; the corrected center-cell value is approximately 2.1021.
These are **different measurements of shape**, not an improvement of the mesh
caused by a new algorithm. For the ball's adaptive run, CV of volumes changes
from approximately 0.21935 to approximately 0.22117. The actual current values
are in `results_3d.csv`; the manuscript discussion is generated from that file.

The corrected volume of the twisted domain is the same for all three methods
up to rounding, as expected when the boundary faces are fixed. The quadrature
is exact for the discrete trilinear geometry, not for an underlying curved
analytic sphere. The reference volume used for normalization is also identified
as a numerical boundary approximation.

## What is not established

1. **Novelty of a new functional is not established.** Inverse mean ratio is
   prior art, and published variational comparisons in 2D/3D already exist.
   Removing an unsupported claim does not create a new scientific contribution.
   The provenance and relation to prior art of the particular adaptive-tension
   formula must be checked with its originating author and the underlying text.
   A search by its informal name is not proof of absence of earlier work.
2. **Strict positivity throughout each hexahedron is not certified.** The new
   diagnostic samples `{0, .5, 1}³`. Positivity at all samples does not imply
   positivity between them or global injectivity. The objective barrier still
   checks eight corners; the denser diagnostic is post-solve. A Bernstein or
   interval-based validator would be a separate implementation and validation
   project, not a claim supplied by this patch.
3. **No universal ranking or PDE-accuracy advantage follows.** One primary μ,
   limited domains/resolutions, fixed boundary nodes and unequal objective-specific
   gradient tolerances do not establish general superiority. Checking derivatives
   at several μ values is not a mesh-quality sensitivity study.
4. **No new Windows EXE has been tested.** The GUI model dispatch is tested; old
   release executables/screenshots are not updated source binaries. No local
   desktop interaction or packaged Windows self-test was performed by this audit.

## Minimum next work before a journal submission

**Before coauthor review:** review this corrected draft, confirm author order and
contributions, explicitly agree whether the second 3D comparator should be
inverse mean ratio or an implemented inverse-harmonic Winslow energy. Resolve
the bibliographic source and intended interpretation of adaptive tension.
The current material is suitable for that substantive discussion, not a claim
that the novelty requirement has been accepted.

**Before journal submission:** choose a precise research question and claim.
A compact discriminating study should include several regular resolutions for
cube/twisted cube/prism, a small predeclared μ range for the adaptive functional,
and a tighter-stopping control on selected non-affine cases. These answer,
respectively, resolution dependence, parameter cherry-picking and whether the
reported differences are stopping artifacts. Treat the single-block ball's
corner degeneracy as a limitation rather than silently capping every refinement.
An independent comparator/implementation is needed for an algorithmic-superiority
claim; a PDE experiment is needed only when claiming downstream solution accuracy.

Full geometric certification is needed before claiming guaranteed valid 3D
meshes. It is not necessary to disguise that missing feature: state the current
scope and decide with coauthors whether it is essential to the research claim.
A full performance study needs warm-up, repeated timings and consistent resource
settings; present runtimes as descriptive until then.

**Do not prioritize now:** GUI cosmetics, another installer, animation, additional
export formats or a larger unstructured gallery of examples. They do not resolve
the functional identity, novelty, admissibility or experimental-design questions.

## Primary references consulted

- Knupp, P. M. (2001). *Algebraic Mesh Quality Metrics*. SIAM J. Sci. Comput. 23(1),
  193–218. DOI: 10.1137/S1064827500371499. Table 2 distinguishes inverse mean ratio
  from Winslow and nondimensional Winslow.
- Huang, W., Kamenski, L., Russell, R. D. (2015). *A comparative numerical study of
  meshing functionals for variational mesh adaptation*. J. Math. Study 48(2),
  168–186. DOI: 10.4208/jms.v48n2.15.04. arXiv:1503.04709.
- Johnen, A., Weill, J.-C., Remacle, J.-F. (2017). *Robust and efficient validation
  of the linear hexahedral element*. arXiv:1706.01613v3. A Jacobian-polynomial
  validation method, not a justification for finite-point positivity alone.
