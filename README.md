# Structural Affinity

Reference implementation of **Structural Affinity: An Algorithm for Coherent Cage-Based Shape Design**
(Aviraj Newatia, Shengan Yang, Hantao Zhong, Rika Antonova, University of Cambridge).

**[Project page](https://camral.github.io/structural-affinity/)** ·
**[Paper (PDF)](https://camral.github.io/structural-affinity/static/pdf/structural-affinity.pdf)**

## Summary

In cage-based deformation (CBD), each handle of the control cage moves independently. This can cause
spiky, distorted and self-intersecting shapes.

Structural Affinity couples the handle motion with a second cage:

- The **inner cage** is the control cage of the CBD pipeline. It has N handles.
- The **outer cage** is a coarser convex cage that strictly encloses the inner cage. It has M vertices.

The outer cage gives an affinity matrix C (N × N). C tells how much each handle follows the other handles.
You calculate C one time, offline. At deformation time, the method does one sparse matrix product. The
method does not add loss terms.

![CBD and CBD-SA. One handle moves.](assets/quickstart_2d.png)

## Requirements

- Python 3.10 or later.
- numpy and scipy.
- Optional: PyTorch, for `structural_affinity_torch.py`. matplotlib, for the examples.

The implementation is one file: [`structural_affinity.py`](structural_affinity.py). Copy this file into
your project. For a PyTorch pipeline, also copy [`structural_affinity_torch.py`](structural_affinity_torch.py).

## Quick start

1. Make the outer cage from the inner cage:

   ```python
   import structural_affinity as sa

   outer_verts, outer_faces = sa.icosahedral_outer_cage(cage_verts)
   ```

2. Calculate the affinity. Do this step one time for each inner cage:

   ```python
   affinity = sa.StructuralAffinity(cage_verts, outer_verts, outer_faces, alpha=0.3, density=0.4)
   ```

3. Apply the affinity to the handle displacements `D` (N, 3):

   ```python
   deformed_cage = cage_verts + affinity.propagate(D)
   ```

In 2D, `cage_verts` has shape (N, 2) and `outer_faces` is `None`.

## Outer cage

The outer cage must:

- Be **convex**.
- **Strictly enclose** all handles of the inner cage.

If the outer cage does not obey these rules, `StructuralAffinity` raises `ValueError`.

Three functions make an outer cage. Each function returns `(outer_verts, outer_faces)`:

| function | vertices | paper experiment |
|---|---|---|
| `icosahedral_outer_cage(cage_verts, padding=0.1)` | 12 in 3D, 7 in 2D | Neural Cages |
| `box_outer_cage(cage_verts, padding=0.05)` | 8 in 3D, 4 in 2D | Robot fingertips |
| `convex_outer_cage(cage_verts, padding=0.05)` | the hull vertices of the inner cage | Wing optimization |

Use `icosahedral_outer_cage` first. A coarse outer cage is sufficient.

Note: for a convex inner cage, `convex_outer_cage` keeps all inner handles. The outer cage is then not
coarser than the inner cage.

To use your own outer cage, supply the arrays directly:

```python
affinity = sa.StructuralAffinity(cage_verts, my_outer_verts, my_outer_faces, alpha=0.3, density=0.4)
```

- `my_outer_verts`: shape (M, 3).
- `my_outer_faces`: shape (F, 3). The faces must have a consistent orientation.

The outer cage controls which handles move together. The result of the method depends on this choice.

## Parameters

| parameter | range | effect |
|---|---|---|
| `alpha` | ≥ 0 | Affinity strength. `0` gives plain CBD. A larger value makes the handles move more together. |
| `density` | [0, 1] | Fraction of off-diagonal pairs in C to keep, largest values first. `0` removes all coupling. `1` keeps C dense. |
| `eps` | > 0 | Alternative to `density`. Keep only the off-diagonal entries ≥ `eps`. |

Start with `alpha` in [0.1, 0.45] and `density` in [0.2, 0.7].

Values in the paper:

| experiment | `alpha` | `density` | outer cage |
|---|---|---|---|
| Neural Cages (chair / table / car) | 0.3 / 0.2 / 0.4 | 0.4 | `icosahedral_outer_cage` |
| Robot fingertips | 0.35 | 0.15 | `box_outer_cage`, padding 0.05 |
| Wing, perturbed starts | 0.05 | 0.025 | `convex_outer_cage` |

Note: the research code of the paper calculated `density` over all N × N entries, including the diagonal.
This file uses only the off-diagonal entries, as the paper describes. The difference is small. For example,
at N = 42, a full-matrix density of 0.4 is equal to 0.385 here.

## Integration into a CBD pipeline

Structural Affinity changes only the step that adds the displacements to the inner cage:

```python
# before
deformed_cage = cage_verts + D
# after
deformed_cage = cage_verts + affinity.propagate(D)
```

Do not change the other parts of the pipeline:

- The weights that bind the mesh to the inner cage (MVC, Green coordinates, BBW or other).
- The optimizer.
- The losses.

`propagate` also accepts a batch with shape (B, N, d).

To save C and use it again, use `StructuralAffinity.from_affinity_matrix(C, alpha)`.

## PyTorch

`AffinityPropagation` is a `torch.nn.Module`. Its gradients go through the propagation to the offsets.

```python
from structural_affinity_torch import AffinityPropagation

propagate = AffinityPropagation(affinity.C, alpha=affinity.alpha).to(device)
deformed_cage = cage + propagate(net(x))        # net(x) gives the offsets, shape (B, N, 3)
```

## Method

| step | time | operation | function |
|---|---|---|---|
| 1 | offline | Make a convex outer cage that strictly encloses the inner cage. | `icosahedral_outer_cage`, `box_outer_cage`, `convex_outer_cage` |
| 2 | offline | Calculate the mean-value coordinates `W` (N, M) of each inner handle in the outer cage. | `mvc` |
| 3 | offline | Calculate `C_ij = Σ_k min(w_ik, w_jk)`. C is symmetric, in [0, 1], with a diagonal of 1. | `affinity_matrix` |
| 4 | offline | Keep the diagonal and the largest off-diagonal entries of C. | `sparsify` |
| 5 | each step | Calculate `ΔV = A D`, with `A = (1 − α) I + α Cᵀ`. | `propagate` |

The `StructuralAffinity` constructor does steps 2 to 4. `StructuralAffinity.propagate` does step 5.

Properties:

- A handle with displacement `D_i` moves by `D_i`.
- Each other handle `j` also moves by `α · C_ij · D_i`.
- The cost of step 5 is O(N + L). L is the number of off-diagonal entries in C.
- For a dense C and α < 1, A is invertible. Thus all cage motions are possible. Structural Affinity changes
  which motions are easy to find. It does not remove motions.

## Limitations

- **Translation.** The update is not normalized. If all handles move by `z`, handle `j` moves by
  `(1 + α Σ_{i≠j} C_ij) z`. Control the global pose separately. The experiments in the paper do this.
- **Invertibility after sparsification.** Sparsification can make A singular. A stays invertible if
  `α ‖C − I‖₂ < 1`.
- **Nonconvex outer cage.** A nonconvex outer cage can give negative mean-value coordinates.
  `affinity_matrix` does not accept negative weights. Refer to Appendix A of the paper.
- **Memory.** `affinity_matrix` uses O(N² M) memory. For cages with many thousands of handles, calculate C in
  blocks of rows.

## Examples

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the examples:

| command | content |
|---|---|
| `python examples/01_quickstart_2d.py` | 2D. Move one handle. Compare CBD and CBD-SA (first figure). |
| `python examples/02_mesh_3d.py` | 3D fingertip in an inner cage with 42 handles. Export OBJ files (second figure). |
| `python examples/03_drop_into_pipeline.py` | Add Structural Affinity to an existing deformer. Measure the time. |
| `python examples/04_optimizer_loop.py` | Evolution strategy, with and without Structural Affinity. |
| `python examples/05_torch_learned.py` | PyTorch training loop with `AffinityPropagation`. |

The examples write their output to `examples/out/`. The code makes all shapes. There are no asset files.

![3D fingertip. One handle moves.](assets/mesh_3d.png)

## Tests

```bash
pytest -q
```

Each test examines one claim of the paper:

- The mean-value coordinates sum to 1 and reproduce the point. In a convex cage, they are positive.
- C is symmetric, with a diagonal of 1. C is equal to 1 minus the total-variation distance. A dense C is
  positive semidefinite.
- Without the cap, the weighted average of the ratios is always 1.
- α = 0 gives plain CBD. One handle displacement propagates as the Method section describes.
- The eigenvalues of the dense operator A are ≥ 1 − α.
- The PyTorch result is equal to the numpy result. It passes `gradcheck`.

## Repository contents

```
structural_affinity.py         implementation (numpy, scipy)
structural_affinity_torch.py   PyTorch propagation
examples/                      examples
tests/                         tests
site/                          project page (GitHub Pages)
```

This repository contains the method only. The code for the experiments in the paper (Neural Cages,
OpenAeroStruct wings, LEAP hand co-design) is in a different research codebase.

## Citation

```bibtex
@misc{newatia2026structuralaffinity,
  title  = {Structural Affinity: An Algorithm for Coherent Cage-Based Shape Design},
  author = {Newatia, Aviraj and Yang, Shengan and Zhong, Hantao and Antonova, Rika},
  year   = {2026},
  url    = {https://camral.github.io/structural-affinity/}
}
```

## Licence

- Code: MIT.
- Project page: CC BY-SA 4.0. The page uses the [Nerfies](https://github.com/nerfies/nerfies.github.io)
  template.
