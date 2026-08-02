# MODULUS-ELF Acceptance Rubric

## Gate 0: Repository bootstrap

Accepted when:

- the repository identifies upstream ELF as the baseline;
- work is isolated to a feature branch;
- the PR is draft until baseline evidence exists.

## Gate 1: Official ELF-B OpenWebText reproduction

Accepted when a run record contains:

- upstream commit SHA;
- local command invocation;
- hardware/backend description;
- config path and any overrides;
- checkpoint identifier;
- generated output path;
- Gen. PPL and unigram entropy;
- run seed(s);
- whether WandB and online PPL evaluation were enabled.

Blocking conditions:

- checkpoint fails to load;
- evaluator does not complete;
- results diverge strongly from upstream sanity target without a documented reason;
- config or sampler was changed before baseline reproduction.

## Gate 2: Behavior-preserving geometry-off overlay

Accepted when:

- `latent_geometry=off` delegates to upstream-equivalent logic;
- a smoke run confirms no shape or JIT regressions;
- generated metrics remain plausibly concordant with the unpatched baseline under matched seed/config.

## Gate 3: Hybrid-sphere training/evaluation

Accepted when:

- the training config differs only in explicit MODULUS geometry fields and output metadata;
- radius and tangency diagnostics are emitted;
- train/eval commands are recorded;
- results are compared against the accepted Gate 1 baseline, not against memory or paper-only numbers.

## Promotion rule

A MODULUS-ELF result is not admitted unless all prior gates clear. The null result is valuable and should be recorded if geometry does not improve the frontier.
