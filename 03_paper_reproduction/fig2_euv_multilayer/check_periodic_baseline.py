#!/usr/bin/env python3
"""Check the 40-pair Mo/Si reference stack at 13.5 nm.

The Mo and Si thicknesses are 3.32 nm and 3.77 nm per period.
Expected reflectance range: 8.71%. Expected PDPS: 6.26 degrees.
The script exits with status 1 if the check fails.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from common_ablation import check_periodic_baseline


def main():
    print("=" * 50)
    print("  EUV Periodic Mo/Si Baseline Check")
    print("  Structure: [3.32 Mo / 3.77 Si] nm x 40 pairs")
    print("  Wavelength: 13.5 nm")
    print("  Expected: dR = 8.71%, PDPS = 6.26 deg")
    print("=" * 50)
    print()

    ok = check_periodic_baseline()

    if ok:
        print("\n[PASS] Periodic baseline verified. SMM kernel is correct.")
        sys.exit(0)
    else:
        print("\n[FAIL] Periodic baseline FAILED. Check SMM implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
