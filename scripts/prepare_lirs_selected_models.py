#!/usr/bin/env python3
"""
Prepare the six LIRS-HMLG character folders used by restaurant_humans_48_people.world.

What it does:
1. Verifies talk.dae, sit.dae, and walk.dae exist for every selected character.
2. Creates one .bak copy of each DAE before changing it.
3. Repairs MakeHuman/LIRS image references that start with /<model-folder>/...
   so Gazebo can resolve them relative to the DAE file.
4. Reports any image references that still cannot be resolved.

Run from anywhere:
    python3 /workspaces/Research_Project/scripts/prepare_lirs_selected_models.py
"""

from pathlib import Path
import re
import shutil
import sys

ROOT = Path("/workspaces/Research_Project/models/actors/LIRS-HMLG")

MODEL_DIRS = [
    ROOT / "Male" / "m_suit",
    ROOT / "Male" / "m_casual_3",
    ROOT / "Female" / "w_casual",
    ROOT / "Female" / "w_casual_2",
    ROOT / "Children" / "c_m_casual",
    ROOT / "Children" / "c_w_casual",
]

ANIMATIONS = ("talk.dae", "sit.dae", "walk.dae")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tga", ".bmp")

def normalise_ref(ref: str, model_dir: Path) -> str:
    """Convert a LIRS absolute-like texture reference to a local relative path."""
    ref = ref.strip()

    # Already valid relative path.
    if not ref.startswith("/"):
        return ref

    # Typical LIRS form:
    # /m_suit/textures/M_Suit_01.png
    # /w_casual_2/denim_jacket.png
    model_prefix = f"/{model_dir.name}/"
    if ref.startswith(model_prefix):
        return ref[len(model_prefix):]

    # Fallback: if an absolute-like ref contains the current model folder name,
    # retain everything after that model folder.
    marker = f"/{model_dir.name}/"
    if marker in ref:
        return ref.split(marker, 1)[1]

    return ref.lstrip("/")


def main() -> int:
    failures = 0

    print("=" * 72)
    print("LIRS-HMLG SELECTED MODEL PREPARATION")
    print("=" * 72)

    for model_dir in MODEL_DIRS:
        print(f"\nMODEL: {model_dir}")

        if not model_dir.is_dir():
            print("  ERROR: model directory does not exist.")
            failures += 1
            continue

        for animation in ANIMATIONS:
            dae = model_dir / animation

            if not dae.is_file():
                print(f"  ERROR: missing {animation}")
                failures += 1
                continue

            backup = dae.with_suffix(dae.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(dae, backup)
                print(f"  Backup created: {backup.name}")

            text = dae.read_text(errors="ignore")
            refs = re.findall(r"<init_from>(.*?)</init_from>", text)

            replacements = {}
            for ref in refs:
                if not ref.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                new_ref = normalise_ref(ref, model_dir)
                if new_ref != ref:
                    replacements[ref] = new_ref

            for old, new in replacements.items():
                text = text.replace(
                    f"<init_from>{old}</init_from>",
                    f"<init_from>{new}</init_from>",
                )

            dae.write_text(text)
            print(f"  {animation}: repaired {len(replacements)} path reference(s)")

            # Re-read and validate image references.
            updated = dae.read_text(errors="ignore")
            updated_refs = re.findall(r"<init_from>(.*?)</init_from>", updated)

            unresolved = []
            for ref in updated_refs:
                if not ref.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                candidate = model_dir / ref
                if not candidate.exists():
                    unresolved.append(ref)

            if unresolved:
                print(f"    WARNING: {len(unresolved)} unresolved texture(s):")
                for ref in unresolved[:10]:
                    print(f"      - {ref}")
                if len(unresolved) > 10:
                    print(f"      ... and {len(unresolved)-10} more")
                failures += 1
            else:
                print("    OK: all referenced image textures resolve.")

    print("\n" + "=" * 72)
    if failures:
        print(f"FINISHED WITH {failures} issue(s). Review warnings before launching.")
        return 1

    print("SUCCESS: all selected LIRS models are prepared for Gazebo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
