#!/usr/bin/env bash
#----------------------------------------------------------------------------#
# verify-restack.sh: prove an assembly lost nothing.                         #
#                                                                            #
#   ./verify-restack.sh <assembled> [reference]                              #
#                                                                            #
# restack.sh's guard is patch-id based, so it refuses when a commit was       #
# reworded on the combined branch even though its content is on a PR branch.  #
# This checks the thing that actually matters - the resulting tree - and is    #
# what should gate accepting a rebuild.                                       #
#                                                                            #
#   CHECK 1  every commit of every branch reached the assembly, by subject.    #
#            Catches a branch skipped entirely - which a resume can do if a    #
#            cherry-pick was aborted rather than completed.                   #
#   CHECK 2  the assembly's tree vs a reference (the branch we ran before),    #
#            file by file. Anything here is either the upstream advance or     #
#            work that moved; both need a human to look once.                 #
#   CHECK 3  no file present in the reference has vanished from the assembly. #
#   CHECK 4  RayInitializerAdvanced's input order is unchanged, because       #
#            ComfyUI node inputs are POSITIONAL and saved workflows store      #
#            values by position.                                             #
#----------------------------------------------------------------------------#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$REPO/stack.branches"
cd "$REPO"

if [ -t 1 ]; then R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; B=$'\e[1m'; Z=$'\e[0m'
else R=""; G=""; Y=""; B=""; Z=""; fi
say()  { printf '\n%s\n' "${B}==>${Z} $*"; }
ok()   { printf '%s\n' "  ${G}PASS${Z} $*"; }
bad()  { printf '%s\n' "  ${R}FAIL${Z} $*"; FAILED=$((FAILED+1)); }
note() { printf '%s\n' "  ${Y}··${Z}   $*"; }
FAILED=0

ASSEMBLED="${1:-}"
REFERENCE="${2:-}"
[ -n "$ASSEMBLED" ] || { echo "usage: $0 <assembled-ref> [reference-ref]" >&2; exit 2; }
git rev-parse --verify --quiet "$ASSEMBLED" >/dev/null || { echo "no such ref: $ASSEMBLED" >&2; exit 2; }

BASE=""; BRANCHES=()
while read -r kind a b c _; do
    case "${kind:-}" in
        base)   BASE="$a" ;;
        branch) BRANCHES+=("$a") ;;
    esac
done < <(sed 's/#.*//' "$MANIFEST")
declare -A ONTO=()
while read -r kind a b c _; do
    [ "${kind:-}" = "branch" ] && ONTO["$a"]="$c"
done < <(sed 's/#.*//' "$MANIFEST")
ASSEMBLE=()
while read -r kind rest; do
    [ "${kind:-}" = "assemble" ] && read -r -a ASSEMBLE <<< "$rest"
done < <(sed 's/#.*//' "$MANIFEST")
BASE_SHA="$(git rev-parse "$BASE")"
base_ref_of() { local o="${ONTO[$1]}"; [ "$o" = "base" ] && echo "$BASE_SHA" || git rev-parse "$o"; }

printf '%s\n' "${B}verify${Z} assembled=$(git rev-parse --short "$ASSEMBLED")  base=$BASE @ $(git rev-parse --short "$BASE")"
[ -n "$REFERENCE" ] && printf '        reference=%s @ %s\n' "$REFERENCE" "$(git rev-parse --short "$REFERENCE")"

#-- CHECK 1: every branch commit reached the assembly --------------------------#
say "CHECK 1  every commit of every branch in the build reached the assembly"
# The assembly is built by cherry-pick, which preserves the subject, so a missing
# subject means the commit was never applied. This is the check that catches a
# branch silently skipped - e.g. when a cherry-pick was aborted and the resume
# assumed it had completed. Content drift is CHECK 2's job, not this one.
ASSEMBLED_SUBJECTS="$(git log --no-merges --format='%s' "$BASE..$ASSEMBLED")"
for b in "${ASSEMBLE[@]}"; do
    base="$(git merge-base "$(base_ref_of "$b")" "$b")"
    missing=0
    while read -r subj; do
        [ -z "$subj" ] && continue
        printf '%s\n' "$ASSEMBLED_SUBJECTS" | grep -qxF "$subj" || {
            bad "$b: commit NOT in the assembly - \"$subj\""; missing=1; }
    done < <(git log --no-merges --format='%s' "$base..$b")
    [ "$missing" -eq 0 ] && ok "$b ($(git rev-list --count --no-merges "$base..$b") commit(s))"
done
for b in "${BRANCHES[@]}"; do
    inbuild=0; for a in "${ASSEMBLE[@]}"; do [ "$a" = "$b" ] && inbuild=1; done
    [ "$inbuild" -eq 0 ] && note "$b is declared but not in 'assemble' (deliberately out of the build)"
done

#-- CHECK 2: tree vs reference --------------------------------------------#
if [ -n "$REFERENCE" ]; then
    say "CHECK 2  assembly tree vs reference, file by file"
    if git diff --quiet "$REFERENCE" "$ASSEMBLED"; then
        ok "trees are IDENTICAL to $REFERENCE"
    else
        UP_ONLY="$(git diff --name-only "$(git merge-base "$REFERENCE" "$BASE")" "$BASE" | sort -u)"
        git diff --name-status "$REFERENCE" "$ASSEMBLED" | while read -r st f rest; do
            if printf '%s\n' "$UP_ONLY" | grep -qxF "$f"; then
                printf '  %s··%s   %-58s %s\n' "$Y" "$Z" "$f" "(also changed by the upstream advance)"
            else
                printf '  %s!!%s   %-58s %s\n' "$R" "$Z" "$f" "$st  <-- REVIEW: not explained by upstream"
            fi
        done
        note "files marked REVIEW are not necessarily wrong - a merge resolution lives"
        note "only in the assembly by design - but each must be explained."
    fi

    #-- CHECK 3: nothing vanished -----------------------------------------#
    say "CHECK 3  no file from the reference is missing in the assembly"
    gone="$(git diff --name-status --diff-filter=D "$REFERENCE" "$ASSEMBLED" | awk '{print $2}')"
    if [ -z "$gone" ]; then ok "no files deleted"
    else printf '%s\n' "$gone" | while read -r f; do bad "deleted: $f"; done; fi
fi

#-- CHECK 4: positional node inputs --------------------------------------#
say "CHECK 4  RayInitializerAdvanced WIDGET order (widgets_values is positional)"
dump_inputs() {
python3 - "$1" <<'PY'
import ast, subprocess, sys
rev = sys.argv[1]
src = subprocess.run(["git","show",f"{rev}:src/raylight/nodes.py"],capture_output=True,text=True).stdout
if not src: sys.exit(3)
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.ClassDef) and node.name == "RayInitializerAdvanced":
        for fn in node.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "INPUT_TYPES":
                for st in ast.walk(fn):
                    if isinstance(st, ast.Dict):
                        for k, v in zip(st.keys, st.values):
                            if isinstance(k, ast.Constant) and k.value in ("required","optional") and isinstance(v, ast.Dict):
                                for kk, vv in zip(v.keys, v.values):
                                    if not isinstance(kk, ast.Constant): continue
                                    spec = vv.elts[0] if isinstance(vv, (ast.Tuple, ast.List)) and vv.elts else None
                                    # A bare ALL_CAPS type string that is not a widget
                                    # primitive is a node link: no widgets_values slot,
                                    # so where it sits cannot shift saved values.
                                    if isinstance(spec, ast.Constant) and isinstance(spec.value, str) \
                                       and spec.value.isupper() and spec.value not in ("INT","FLOAT","STRING","BOOLEAN"):
                                        continue
                                    print(f"{k.value}:{kk.value}")
PY
}
A_IN="$(dump_inputs "$ASSEMBLED")"
if [ -z "$A_IN" ]; then
    bad "could not read RayInitializerAdvanced inputs from the assembly"
elif [ -n "$REFERENCE" ]; then
    R_IN="$(dump_inputs "$REFERENCE")"
    if [ "$A_IN" = "$R_IN" ]; then
        ok "input order identical to the reference ($(printf '%s\n' "$A_IN" | wc -l) inputs)"
    else
        bad "input ORDER CHANGED - saved workflows will silently shift values:"
        diff <(printf '%s\n' "$R_IN") <(printf '%s\n' "$A_IN") | sed 's/^/        /'
    fi
else
    note "no reference given; assembly order is:"; printf '%s\n' "$A_IN" | nl -ba | sed 's/^/        /'
fi

say "result"
if [ "$FAILED" -eq 0 ]; then
    printf '%s\n' "  ${G}${B}all checks passed${Z} - the assembly contains every branch in the build"
    exit 0
else
    printf '%s\n' "  ${R}${B}$FAILED check(s) failed${Z} - do not accept this assembly"
    exit 1
fi
