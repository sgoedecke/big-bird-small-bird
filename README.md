# Big Bird Small Bird

A "size it up" game with Victorian bird silhouettes. You resize/move the red bird relative to the black one, lock in, and see how close you were. Endless rounds, no backend.

## Play

```
cd web && python3 -m http.server 8080    # http://127.0.0.1:8080
```

## Silhouette pipeline

All steps are cached/resumable. Python env is managed with `uv`.

| Step | Command | Output |
|---|---|---|
| 1. Species + lengths | `uv run python pipeline/species.py` | `data/species.json` – Victorian species (core + introduced) from Garnett et al. 2015 *Australian Bird Data* (`data/raw/abd.csv`), matched to iNaturalist taxa with Victorian research-grade observation counts |
| 2. Photos | `uv run python pipeline/fetch_photos.py [slugs…] [-n 20] [--min-obs 20]` | `data/photos/<slug>/` – most-faved CC-licensed (non-ND) iNaturalist photos from Victoria (falls back to Australia) + `meta.json` with attribution |
| 3. Segment + rank | `uv run python pipeline/segment.py [slugs…]` | `data/silhouettes/<slug>/` – BiRefNet masks (MPS), CLIP side-on pose score, silhouette PNGs, auto bill/tail endpoints, `candidates.json` ranked by score |
| 4. Review | `uv run python pipeline/review.py` | http://127.0.0.1:8777 – pick the best candidate per species (keys 1–6, `x` reject), fix bill/tail endpoints. Saves `data/review.json` |
| 4b. Image review (agents) | `uv run python pipeline/agent_review.py sheet\|pick\|reject\|merge\|fix\|review-sheet …` | Review from rendered grid images: `pick` writes `data/agent_review/`, `merge` folds into `review.json` (never overwriting human picks), `fix` corrects endpoints of an existing pick in place, `review-sheet` renders current picks |
| 5. Export | `uv run python pipeline/export.py [--auto]` | `web/birds.json` + `web/sils/*.png`. `--auto` uses the top-ranked candidate for unreviewed species |

Debug helper: `uv run python pipeline/contact_sheet.py out.png <slug>…` renders photos + silhouettes + endpoints.

### Scaling

Each silhouette is scaled so that its bill-tip → tail-tip distance, measured as the shortest path *inside* the silhouette (so curled necks and cocked tails count roughly as they would on a stretched-out specimen), equals the species' body length from Garnett et al.

## Credits

- Body lengths: Garnett, S.T. et al. (2015) Biological, ecological, conservation and legal information for all species and subspecies of Australian bird. *Scientific Data* 2:150061.
- Silhouettes are traced from CC-licensed iNaturalist observation photos; attribution is shown in-game after each round.

## Deploy

`web/` is a static site served by GitHub Pages from the `gh-pages` branch: https://sgoedecke.github.io/big-bird-small-bird/

To redeploy after changing `web/`, commit to `main` and run:

```
git push -f origin $(git commit-tree HEAD:web -m "Deploy web/"):refs/heads/gh-pages
```
