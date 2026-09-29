#!/usr/bin/env bash
#----------------------------------------------------------------------------#
# restack.sh: maintain this fork's PR branches and rebuild the combined main. #
#                                                                            #
#   ./restack.sh                   rebase every PR branch onto its declared   #
#                                  base, then assemble 'into' from all of     #
#                                  them.                                      #
#   ./restack.sh --dry-run         show the plan, touch nothing.              #
#   ./restack.sh --assemble-only   skip the rebases; only assemble. Use this  #
#                                  to build a trial without rewriting any     #
#                                  branch.                                    #
#   ./restack.sh --into <branch>   write the assembly somewhere other than    #
#                                  the manifest's 'into' (e.g. a trial ref).  #
#   ./restack.sh --rerere          let rerere replay recorded conflict         #
#                                  resolutions. OFF by default: see the        #
#                                  warning below.                             #
#   ./restack.sh --continue        resume an assembly that stopped on a        #
#                                  conflict, from the current HEAD, after      #
#                                  'git cherry-pick --continue'. Without this  #
#                                  a re-run restarts from the base and hits    #
#                                  every earlier conflict again.               #
#                                                                            #
# Two concerns, per stack.branches:                                           #
#   * PHASE 1 rebases each 'branch <name> onto <ref>' onto its base, 'base'   #
#     (upstream) or another branch. Independent fixes sit on 'base' so their   #
#     PR diffs never drag each other in; real chains stack on their parent.   #
#   * PHASE 2 assembles the combined branch by cherry-picking every branch's  #
#     OWN commits in 'assemble' order. It is a throwaway build ref of copied  #
#     commits, never pushed as a PR.                                          #
#                                                                            #
# GUARD: any non-merge commit on the combined branch whose patch is not       #
# present in some PR branch is work authored directly on it; restack refuses  #
# so a rebuild cannot silently drop it. Move it to a branch first.            #
#                                                                            #
# Patch-id is exact, so a commit that was reworded or reworked on the         #
# combined branch reads as stray even when its content is on a branch under   #
# a different commit. That is deliberately noisy: verify-restack.sh is what   #
# proves equivalence by tree, and this guard only refuses to proceed blindly. #
#                                                                            #
# RERERE IS OFF UNLESS ASKED FOR, and every git call here sets it explicitly  #
# rather than inheriting whatever the shared config says. rr-cache lives in   #
# the common git dir, so an exploratory merge in any worktree records into    #
# the same cache - and rerere does not distinguish a resolution you committed #
# from one recorded during a merge you aborted. Replaying the latter silently #
# corrupts the assembly. Use --rerere only when the cache holds resolutions   #
# from real assemblies you verified.                                          #
#----------------------------------------------------------------------------#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$REPO/stack.branches"
cd "$REPO"

if [ -t 1 ]; then R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; B=$'\e[1m'; Z=$'\e[0m'
else R=""; G=""; Y=""; B=""; Z=""; fi
say()  { printf '%s\n' "${B}==>${Z} $*"; }
ok()   { printf '%s\n' "  ${G}ok${Z}  $*"; }
warn() { printf '%s\n' "  ${Y}!! ${Z} $*"; }
die()  { printf '%s\n' "${R}FAIL${Z} $*" >&2; exit 1; }

DRY=0; ASSEMBLE_ONLY=0; INTO_OVERRIDE=""; RERERE=0; CONTINUE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)       DRY=1 ;;
        --assemble-only) ASSEMBLE_ONLY=1 ;;
        --rerere)        RERERE=1 ;;
        --continue)      CONTINUE=1 ;;
        --into)          shift; INTO_OVERRIDE="${1:-}" ;;
        *)               die "unknown argument '$1'" ;;
    esac
    shift
done

# Pinned on every conflict-capable call so ambient config cannot change what
# this script does.
if [ "$RERERE" -eq 1 ]; then
    RR=(-c rerere.enabled=true -c rerere.autoupdate=true)
else
    RR=(-c rerere.enabled=false)
fi

#-- parse manifest ----------------------------------------------------------#
[ -f "$MANIFEST" ] || die "missing $MANIFEST"
BASE=""; INTO=""; BRANCHES=(); declare -A ONTO; ASSEMBLE=()
while read -r kind a b c _; do
    case "${kind:-}" in
        base)   BASE="$a" ;;
        into)   INTO="$a" ;;
        branch) [ "$b" = "onto" ] || die "branch '$a' missing 'onto <ref>'"
                BRANCHES+=("$a"); ONTO["$a"]="$c" ;;
        ""|\#*) : ;;
    esac
done < <(sed 's/#.*//' "$MANIFEST")
while read -r kind rest; do
    [ "${kind:-}" = "assemble" ] && read -r -a ASSEMBLE <<< "$rest"
done < <(sed 's/#.*//' "$MANIFEST")

[ -n "$INTO_OVERRIDE" ] && INTO="$INTO_OVERRIDE"
[ -n "$BASE" ]              || die "manifest has no 'base' line"
[ -n "$INTO" ]              || die "manifest has no 'into' line"
[ "${#BRANCHES[@]}" -gt 0 ] || die "manifest lists no branches"
[ "${#ASSEMBLE[@]}" -gt 0 ] || die "manifest has no 'assemble' line"

git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repo"
if ! git diff --quiet HEAD -- || ! git diff --cached --quiet; then
    die "working tree has uncommitted tracked changes; commit them first"
fi
for b in "${BRANCHES[@]}"; do
    git rev-parse --verify --quiet "$b" >/dev/null || die "branch '$b' does not exist"
done
for b in "${ASSEMBLE[@]}"; do
    [ -n "${ONTO[$b]:-}" ] || die "assemble lists '$b', which no 'branch' line declares"
done

say "fetching upstream"
[ "$DRY" -eq 1 ] || git fetch upstream >/dev/null 2>&1 || warn "git fetch upstream failed, using local $BASE"
git rev-parse --verify --quiet "$BASE" >/dev/null || die "base ref '$BASE' not found"
BASE_SHA="$(git rev-parse "$BASE")"
say "base $BASE @ ${BASE_SHA:0:12}   into $INTO   branches=${#BRANCHES[@]}"
[ "$RERERE" -eq 1 ] && warn "rerere replay ENABLED - only safe if rr-cache holds verified resolutions" \
                    || ok "rerere replay off; every conflict stops for a human"

base_ref_of() { local o="${ONTO[$1]}"; [ "$o" = "base" ] && echo "$BASE_SHA" || git rev-parse "$o"; }
PROGRESS="$(git rev-parse --git-path stack-progress)"

#-- GUARD: no stray work authored directly on the combined branch ----------#
# Skipped when resuming: the partial assembly legitimately holds commits whose
# patch no longer matches their branch, because a conflict was just resolved.
if [ "$CONTINUE" -eq 0 ] && git rev-parse --verify --quiet "$INTO" >/dev/null; then
    if [ "$(git rev-parse "$INTO")" = "$(git rev-parse --verify --quiet refs/stack-assembled || true)" ]; then
        ok "$INTO is the last assembly (refs/stack-assembled), skipping stray scan"
    else
        declare -A BRANCH_PATCH=()
        for b in "${BRANCHES[@]}"; do
            while read -r c; do [ -n "$c" ] || continue
                pid="$(git show "$c" | git patch-id --stable | awk '{print $1}')"
                [ -n "$pid" ] && BRANCH_PATCH["$pid"]=1
            done < <(git rev-list --no-merges "$(base_ref_of "$b")..$b")
        done
        stray=()
        while read -r c; do [ -n "$c" ] || continue
            pid="$(git show "$c" | git patch-id --stable | awk '{print $1}')"
            [ -n "${BRANCH_PATCH[$pid]:-}" ] || stray+=("$c")
        done < <(git rev-list --no-merges "$BASE_SHA..$INTO" 2>/dev/null)
        if [ "${#stray[@]}" -gt 0 ]; then
            warn "$INTO has ${#stray[@]} commit(s) whose patch is not in any PR branch:"
            for c in "${stray[@]}"; do printf '        %s\n' "$(git log -1 --format='%h %s' "$c")"; done
            warn "each must be accounted for: folded onto a branch, or deliberately dropped"
            warn "prove the outcome with ./verify-restack.sh <assembled> <reference>"
            [ "$DRY" -eq 1 ] || die "refusing to rebuild over unaccounted work"
        else
            ok "$INTO carries no stray commits"
        fi
    fi
fi

#-- plan -------------------------------------------------------------------#
if [ "$ASSEMBLE_ONLY" -eq 0 ]; then
    say "PHASE 1 rebase PR branches onto their base:"
    for b in "${BRANCHES[@]}"; do printf '        %-42s onto %s\n' "$b" "${ONTO[$b]}"; done
else
    say "PHASE 1 skipped (--assemble-only): no branch will be rewritten"
fi
say "PHASE 2 assemble $INTO (own commits, in order):"
printf '        %s\n' "${ASSEMBLE[*]}"
[ "$DRY" -eq 1 ] && { say "dry-run: nothing changed"; exit 0; }

#-- recovery refs ---------------------------------------------------------#
say "saving recovery refs under refs/restack-backup/*"
for b in "${BRANCHES[@]}"; do git update-ref "refs/restack-backup/$b" "$(git rev-parse "$b")"; done
git rev-parse --verify --quiet "$INTO" >/dev/null \
    && git update-ref "refs/restack-backup/$INTO" "$(git rev-parse "$INTO")"
ok "recovery refs saved (restore: git update-ref refs/heads/<b> refs/restack-backup/<b>)"

#-- PHASE 1 ---------------------------------------------------------------#
if [ "$ASSEMBLE_ONLY" -eq 0 ]; then
    for b in "${BRANCHES[@]}"; do
        o="${ONTO[$b]}"
        [ "$o" = "base" ] && newbase="$BASE_SHA" || newbase="$(git rev-parse "$o")"
        if git rev-parse --verify --quiet "refs/stack-base/$b" >/dev/null; then
            oldbase="$(git rev-parse "refs/stack-base/$b")"
        else
            oldbase="$(git merge-base "$newbase" "$b")"
        fi
        say "rebase $b onto ${newbase:0:12} (cut ${oldbase:0:12})"
        if [ "$newbase" = "$oldbase" ] || git merge-base --is-ancestor "$newbase" "$b"; then
            ok "$b already on base @ $(git rev-parse --short "$b")"
        else
            git "${RR[@]}" rebase --onto "$newbase" "$oldbase" "$b" \
                || die "conflict rebasing $b: resolve, 'git rebase --continue', then re-run"
            ok "$b @ $(git rev-parse --short "$b")"
        fi
        git update-ref "refs/stack-base/$b" "$newbase"
    done
fi

#-- PHASE 2 ---------------------------------------------------------------#
if [ "$CONTINUE" -eq 1 ]; then
    [ -f "$PROGRESS" ] || die "--continue but no assembly in progress ($PROGRESS missing)"
    mapfile -t ASSEMBLE < "$PROGRESS"
    head="${ASSEMBLE[0]:-}"
    if [ -e "$(git rev-parse --git-path CHERRY_PICK_HEAD)" ]; then
        # Finish it here rather than trusting that it was finished elsewhere.
        say "completing the in-progress cherry-pick of $head"
        git diff --name-only --diff-filter=U | grep -q . \
            && die "unresolved conflicts remain; resolve and 'git add' them first"
        git add -u
        git "${RR[@]}" -c core.editor=true cherry-pick --continue >/dev/null 2>&1 || true
        [ -e "$(git rev-parse --git-path CHERRY_PICK_HEAD)" ] \
            && die "the cherry-pick of $head is still not finished"
    else
        # No pick in progress: the head branch was either completed or ABORTED.
        # Assuming it completed is how a branch gets silently skipped, so verify
        # by subject before dropping it.
        base="$(git merge-base "$(base_ref_of "$head")" "$head")"
        while read -r subj; do
            [ -z "$subj" ] && continue
            git log --no-merges --format='%s' "$BASE_SHA..HEAD" | grep -qxF "$subj" || {
                warn "$head is not in the assembly (missing \"$subj\") - re-attempting it"
                ASSEMBLE=("${ASSEMBLE[@]}"); head=""; break; }
        done < <(git log --no-merges --format='%s' "$base..$head")
    fi
    [ -n "$head" ] && ASSEMBLE=("${ASSEMBLE[@]:1}")
    [ "${#ASSEMBLE[@]}" -gt 0 ] || { rm -f "$PROGRESS"; say "assembly already complete"; git branch -f "$INTO" HEAD; git checkout -q "$INTO"; git update-ref refs/stack-assembled "$(git rev-parse HEAD)"; exit 0; }
    say "resuming at $(git rev-parse --short HEAD); ${#ASSEMBLE[@]} branch(es) left"
else
    say "assembling $INTO from $BASE @ ${BASE_SHA:0:12}"
    git checkout -q --detach "$BASE_SHA"
fi
REMAINING=("${ASSEMBLE[@]}")
for b in "${ASSEMBLE[@]}"; do
    # record what is still to do BEFORE attempting it, so a conflict stop leaves
    # this branch at the head of the remaining list for --continue to retry.
    printf '%s\n' "${REMAINING[@]}" > "$PROGRESS"
    range="$(base_ref_of "$b")..$b"
    n="$(git rev-list --count --no-merges "$range")"
    if [ "$n" -eq 0 ]; then ok "$b: no own commits, skipped"; REMAINING=("${REMAINING[@]:1}"); continue; fi
    replayed=0
    if ! git "${RR[@]}" cherry-pick "$range" >/tmp/rl-restack-cp.out 2>&1; then
        tries=0
        # Only rerere can clear a conflict without a human, so without it there
        # is nothing to drive and the pick stops below.
        while [ "$RERERE" -eq 1 ] \
              && [ -e "$(git rev-parse --git-path CHERRY_PICK_HEAD)" ] \
              && [ -z "$(git diff --name-only --diff-filter=U)" ] \
              && [ "$tries" -lt "$n" ]; do
            git add -u
            git "${RR[@]}" -c core.editor=true cherry-pick --continue >>/tmp/rl-restack-cp.out 2>&1 || true
            replayed=$((replayed+1)); tries=$((tries+1))
        done
        if [ -e "$(git rev-parse --git-path CHERRY_PICK_HEAD)" ]; then
            printf '%s\n' "  conflicted files:"; git diff --name-only --diff-filter=U | sed 's/^/        /'
            die "cherry-pick conflict assembling $b: resolve, 'git cherry-pick --continue', then './restack.sh --continue'"
        fi
    fi
    if [ "$replayed" -gt 0 ]; then
        ok "$b: +$n commit(s) -> $(git rev-parse --short HEAD) (${Y}$replayed rerere${Z})"
    else
        ok "$b: +$n commit(s) -> $(git rev-parse --short HEAD)"
    fi
    REMAINING=("${REMAINING[@]:1}")
done
rm -f "$PROGRESS"
TIP="$(git rev-parse HEAD)"
git branch -f "$INTO" "$TIP"
git checkout -q "$INTO"
git update-ref refs/stack-assembled "$TIP"
ok "$INTO -> assembled @ ${TIP:0:12}"
say "${G}assembled${Z} ${#ASSEMBLE[@]} branches; $INTO @ ${TIP:0:12}"
say "now prove nothing was lost:  ./verify-restack.sh $INTO <reference-ref>"
