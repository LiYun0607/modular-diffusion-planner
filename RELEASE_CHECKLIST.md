# Release checklist — ITSC 2026 code drop

This file tracks the decisions and manual steps still needed before the
repository goes public on GitHub. Tick items off as they are done; delete
this file before the first public push.

## Decisions you need to make

- [ ] **Repository host**
  - Personal account (`liyun…/modular-diffusion-planner`)?
  - Lab account (`tsukada-lab/…`)?
  - TIER IV org (Simon may need to OK this since we modify their package)?
- [ ] **Repository name**
  - Suggested: `modular-diffusion-planner` (matches the anonymous URL `modular-diffusion-planner-3DDA`)
- [ ] **Visibility** — public from day 1, or start private and flip after camera-ready?
- [ ] **License** — Apache 2.0 (default, matches upstream Tier4 package).
      If you want a different license you must keep Apache 2.0 for the
      `planner/` subtree per the upstream license, but the rest of the repo
      could use a different license (not recommended — keep it uniform).
- [ ] **Co-author contact emails** — confirm the four UTokyo authors and
      Simon Thompson are happy with their emails appearing in `CITATION.bib`
      / `NOTICE` (they are taken from the camera-ready paper).

## Items flagged by automated scan

- [ ] `planner/autoware_diffusion_planner/package.xml` lists four `tier4.jp`
      maintainer emails (Sanchez, Saito, Sakayori, Sakoda) inherited from
      upstream. Keep them for Apache 2.0 attribution, **and** add Li Yun
      (UTokyo) as an additional `<maintainer>` for the modular fork.
- [ ] `planner/autoware_diffusion_planner/src/diffusion_planner_core.cpp:191`
      has `// TODO(Daniel): add static objects` carried over from upstream
      style. Decide: leave as-is (acknowledges upstream author) or rewrite
      `// TODO: add static objects`.

## Pre-push tasks

- [ ] **Update the paper** — replace
      `https://anonymous.4open.science/r/modular-diffusion-planner-3DDA/`
      in `main.tex` (line ~47) and the arxiv version with the final GitHub URL.
      Re-compile and upload to arXiv.
- [ ] **Add `docs/graphsurgeon_recipe.md`** explaining how to split the
      monolithic `diffusion_planner.onnx` into `context_encoder.onnx` +
      `dit_core.onnx` (referenced in README §1).
- [ ] **Drop sample frame data** — provide a small (`<50 MB`) tarball of
      AWSIM-captured frames so reviewers can run `latency_benchmark.py` and
      `offline_solver_benchmark.py` without redoing simulation. Either:
      - upload as a GitHub Release asset, or
      - point to a Zenodo DOI in the README.
- [ ] **Clean test data** — confirm `planner/autoware_diffusion_planner/test_map/`
      doesn't contain anything proprietary (lanelet2 map for Tier4 internal
      testing). Replace with a sample.osm if needed.
- [ ] **Sanity-build** — drop `planner/autoware_diffusion_planner` into a
      fresh Autoware workspace and run `colcon build` + `colcon test` to
      confirm it still compiles after de-anonymization edits.

## Pre-flight scans (rerun before push)

```bash
# No absolute paths from your dev machine
grep -rn "/root/\|/home/" --include="*.py" --include="*.cpp" --include="*.hpp" .

# No internal hostnames or credentials
grep -rEn "tier4\.jp|wide\.ad\.jp|api[_-]key|password|secret|token" --include="*.py" --include="*.cpp" --include="*.hpp" --include="*.md" .

# No Tier4 internal references that shouldn't go public
grep -rin "internal\|confidential\|do not distribute" --include="*.py" --include="*.cpp" --include="*.hpp" --include="*.md" .
```

## After-push tasks

- [ ] Add a GitHub Pages project page with the figures (or just link
      `docs/itsc2026_paper.pdf`).
- [ ] Tag a release: `v1.0-itsc2026` once the camera-ready is final.
- [ ] Update memory: replace the anonymous URL note in the user's auto-memory
      with the final GitHub URL (so future Claude Code sessions know where to
      look).
- [ ] Notify co-authors and link from your University of Tokyo lab page.

## Files NOT to commit

- `planner/autoware_diffusion_planner/media/*.gif` — these are large
  upstream marketing GIFs from Tier4. Keep them only if you want the
  README to render with them; otherwise add `media/*.gif` to `.gitignore`.
- Any `*.onnx`, `*.engine`, `*.plan` — too large; documented in `.gitignore`.
- `~/autoware_data/` — never include checkpoint artefacts.
