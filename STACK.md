# Fork layout and the restack

This branch carries tooling only. It is assembled into `main` so the scripts
are present on the branch we run - `restack.sh` checks out the base while
assembling, so tooling that is not in the build disappears mid-run.

## The rule

**Features and fixes go on branches. `main` is an assembly of those branches and
nothing else.** A change authored directly on `main` has no PR branch behind it,
so the next rebuild silently drops it. `restack.sh` refuses to run when it finds
one.

## Files

| file | what it is |
|---|---|
| `stack.branches` | source of truth: the base, every PR branch and the base it sits on, and the order they assemble in |
| `restack.sh` | rebases each branch onto its base, then assembles `main` |
| `verify-restack.sh` | proves an assembly lost nothing; run before accepting one |

## When a conflict stops it

```
# resolve the files, then
git cherry-pick --continue
./restack.sh --continue
```

`--continue` resumes from the current HEAD using the remaining-branch list in
`.git/stack-progress`. It finishes the in-progress cherry-pick itself, and if
there is none it checks by commit subject whether that branch actually landed
before moving past it - **aborting a cherry-pick and then resuming is how a
branch gets silently skipped**, which happened once here and cost a whole
assembly. Without it a re-run restarts from the base and hits every
conflict already resolved - which, with rerere replay off, means resolving them
all again by hand.

## Normal use

```
./restack.sh --dry-run                        # the plan, touches nothing
./restack.sh                                  # rebase all branches, rebuild main
./verify-restack.sh main <previous-main-sha>   # prove nothing was lost
```

## Rebuilding without risking main

```
git tag pre-restack-$(date +%Y%m%d) main
./restack.sh --assemble-only --into stack/trial
./verify-restack.sh stack/trial pre-restack-<date>
```

`--assemble-only` skips the rebases, so no branch is rewritten. Compare the
trial, and only then point `main` at it.

## Recovery

Every run saves `refs/restack-backup/<branch>` before touching anything.

```
git update-ref refs/heads/<branch> refs/restack-backup/<branch>
```

## Two things that will bite

**Node inputs are positional.** Saved workflows store widget values by index, so
the order branches assemble in decides whether they still load correctly.
`feat/rank-gpu-pinning` inserts an input at index 4 and
`feat/initializer-input-ux` regroups the tail: rank-pinning must assemble first.
`verify-restack.sh` CHECK 4 compares the resulting order against a reference and
fails if it moved.

**Never enable rerere replay against an unvetted cache.** `rr-cache` lives in the
common git dir, so an exploratory merge in *any* worktree records into the same
cache, and rerere does not distinguish a resolution you committed from one
recorded during a merge you aborted. Replaying the latter corrupts the assembly
silently - it produced a duplicated comment block and two `usp_mlp_forward`
definitions here once. `restack.sh` therefore runs with `rerere.enabled=false`
pinned on every conflict-capable call, whatever the shared config says, and
replay is opt-in via `--rerere`. Use it only when the cache holds resolutions
from assemblies you verified.

Do not "fix" a suspect cache by deleting it: that directory is shared with the
production repo and accumulates resolutions from every previous session. Turn
replay off instead.

**Patch-id is exact.** `restack.sh`'s stray guard compares patch-ids, so a commit
reworded or reworked on `main` reads as stray even when its content sits on a
branch. That is deliberate - it refuses rather than guesses. `verify-restack.sh`
settles it by comparing trees and by proving each branch merges into the
assembly as a no-op.

## Branches out of the build on purpose

* `tooling/fsdp-probes` - measurement scaffolding. Any timing taken with it
  armed is unquotable, so it is never assembled.
* `tooling/stack` - this branch is the exception: it IS assembled, and has to
  be. `restack.sh` checks out the base and cherry-picks, so unless the tooling
  rides along the working tree loses the script partway through and cannot be
  re-run after a conflict stop. It assembles **first** for the same reason.
* Anything the manifest declares but leaves out of `assemble`, with the reason
  in a comment beside it.
