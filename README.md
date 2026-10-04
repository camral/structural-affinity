# Structural Affinity — Project Page

Project page for **Structural Affinity: An Algorithm for Coherent Cage-Based Shape Design**
(Newatia, Yang, Zhong, Antonova — University of Cambridge).

**Live:** https://camral.github.io/structural-affinity/

Static site (`index.html` + `static/`), no build step. Every push to `main` redeploys through
`.github/workflows/pages.yml` (GitHub Actions → Pages). One-time repo setting: *Settings → Pages → Source: GitHub Actions*.

Preview locally: `python -m http.server` and open http://localhost:8000.

## What's on the page

| Section | Source |
|---|---|
| Playground (CBD vs CBD-SA) | `static/js/playground.js`: 2D Floater MVC, affinity `C_ij = Σ_k min(w_ik, w_jk)`, density sparsification, `ΔV = D + α(C − I)ᵀD` |
| Neural Cages chart | Table 1 numbers, hard-coded in `static/js/index.js` (`NC`) |
| Wing paired-runs chart | `analysis/airfoil_evaluation/refined_convex38_paired.csv` (matched valid pairs), in `static/js/index.js` (`AERO`) |
| Figures | copied from the paper's `imgs/` and downscaled into `static/images/` |
| Paper PDF | `static/pdf/structural-affinity.pdf` |

## Credits

Built on the [Nerfies](https://github.com/nerfies/nerfies.github.io) project page template (CC BY-SA 4.0).
