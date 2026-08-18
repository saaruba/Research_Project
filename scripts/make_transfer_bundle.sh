#!/usr/bin/env bash
# ============================================================================
# Bundle ONLY the files the lab PC needs that git does not carry.
#
#     bash scripts/make_transfer_bundle.sh
#     bash scripts/make_transfer_bundle.sh /path/to/output.tar.gz
#
# ----------------------------------------------------------------------------
# WHY THIS IS SMALL
# ----------------------------------------------------------------------------
# The project folder is around 70 GB, and almost none of it is needed to RUN
# the simulation:
#
#   dataset/1 ... dataset/60   ~70 GB   raw PLUS-HRI recordings. Needed ONLY to
#                                       re-derive the offline results, which
#                                       are already derived. NOT needed to run
#                                       the robot.
#   models/actors/LIRS-HMLG    7.8 GB   the whole human actor library. The
#                                       world uses exactly ONE mesh from it.
#
# So this bundle takes the trained model, the Gazebo assets the world actually
# references, and nothing else - roughly 140 MB. Everything else travels
# through git.
#
# On the lab PC:
#     git clone <repo> && cd Research_Project
#     tar xzf tiago_sim_bundle.tar.gz          # unpacks into place
#     bash scripts/install_sim_stack.sh
# ============================================================================

set -o pipefail

PROJECT=/workspaces/Research_Project
OUT="${1:-/tmp/tiago_sim_bundle.tar.gz}"

cd "$PROJECT" || { echo "Cannot find $PROJECT" >&2; exit 1; }

echo "============================================================"
echo "  BUILDING TRANSFER BUNDLE"
echo "============================================================"

# ---------------------------------------------------------------------------
# What goes in, and why each one is required at RUN time.
# ---------------------------------------------------------------------------
ITEMS=()

add() {
    if [ -e "$1" ]; then
        ITEMS+=("$1")
        printf "  %-58s %s\n" "$1" "$(du -sh "$1" 2>/dev/null | cut -f1)"
    else
        echo "  MISSING (skipped): $1" >&2
    fi
}

echo ""
echo "Trained models - bc_policy_node loads these:"
add dataset/processed/models/approach_pose_random_forest_tuned.joblib
add dataset/processed/models/approach_pose_mlp_tuned.joblib
add dataset/processed/models/approach_pose_evaluation.json
add dataset/processed/models/approach_pose_metrics.json
add dataset/processed/models/grid_search_results.csv

echo ""
echo "Gazebo assets referenced by the world file:"
# The dining table (model:// reference) and the one human mesh the world
# actually uses. The rest of LIRS-HMLG - 7.8 GB of other actors - is not
# referenced by restaurant_testing.world and would be dead weight.
add models/restaurant_furniture
add models/restaurant_decor
add models/actors/LIRS-HMLG/Male/m_suit

echo ""
echo "Existing results, so the lab PC can aggregate across both machines:"
add dataset/processed/sim_results

if [ ${#ITEMS[@]} -eq 0 ]; then
    echo "Nothing to bundle." >&2
    exit 1
fi

echo ""
echo "Compressing to $OUT ..."
tar czf "$OUT" "${ITEMS[@]}" || { echo "tar failed" >&2; exit 1; }

echo ""
echo "============================================================"
echo "  DONE: $OUT   ($(du -sh "$OUT" | cut -f1))"
echo "============================================================"
echo ""
echo "Copy it to the lab PC, then there:"
echo "    cd <cloned repo>"
echo "    tar xzf $(basename "$OUT")"
echo "    bash scripts/install_sim_stack.sh"
echo ""
echo "NOTE: the actor mesh is referenced by ABSOLUTE PATH inside the world"
echo "file (file:///workspaces/Research_Project/models/actors/...). If the lab"
echo "PC clones to a different directory, fix it with:"
echo "    grep -rl '/workspaces/Research_Project' src/tiago_social_worlds/worlds/"
echo "    sed -i 's|/workspaces/Research_Project|<new path>|g' \\"
echo "        src/tiago_social_worlds/worlds/*.world"
echo "============================================================"
