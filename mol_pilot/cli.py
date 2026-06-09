"""
MolPilot command-line interface.

Usage examples
--------------
# Batch-score a SMILES file:
    molpilot score compounds.smi --model solubility.joblib --output scores.csv

# Generate design suggestions for one molecule:
    molpilot design "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1" --model sol.joblib

# Run the Streamlit UI:
    molpilot ui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_score(args: argparse.Namespace) -> None:
    import pandas as pd
    from mol_pilot.io import iter_smiles_file
    from mol_pilot.models import load_predictor

    predictor = load_predictor(args.model)

    smiles_list: list[str] = []
    names: list[str] = []
    for smi, mol in iter_smiles_file(args.input, smiles_col=args.smiles_col):
        smiles_list.append(smi)
        names.append(smi)

    if not smiles_list:
        print("No valid SMILES found in input file.", file=sys.stderr)
        sys.exit(1)

    preds = predictor.predict(smiles_list)

    df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "predicted_property": preds,
        }
    )
    df = df.sort_values("predicted_property", ascending=not args.descending).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"Scored {len(df)} molecules → {args.output}")


def _cmd_design(args: argparse.Namespace) -> None:
    from mol_pilot.design import generate_analogs
    from mol_pilot.models import load_predictor

    predictor = load_predictor(args.model)
    df = generate_analogs(
        args.smiles,
        predictor,
        n_analogs=args.n_analogs,
        include_mmp=not args.no_mmp,
        optimize_direction=args.direction,
    )

    if df.empty:
        print("No analogs generated.", file=sys.stderr)
        sys.exit(1)

    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\n→ Design suggestions saved to {args.output}")


def _cmd_ui(_args: argparse.Namespace) -> None:
    import subprocess
    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="molpilot",
        description="MolPilot — medicinal-chemistry molecular design copilot",
    )
    parser.add_argument("--version", action="version", version="molpilot 0.1.0")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── score ──────────────────────────────────────────────────────────────
    score_p = sub.add_parser("score", help="Batch-score molecules from a SMILES file")
    score_p.add_argument("input", help="Input SMILES file (one SMILES per line)")
    score_p.add_argument("--model", required=True, metavar="PATH",
                         help="Path to trained predictor (.joblib)")
    score_p.add_argument("--output", default="predictions.csv", metavar="PATH",
                         help="Output CSV file (default: predictions.csv)")
    score_p.add_argument("--smiles-col", type=int, default=0, metavar="N",
                         help="Column index of SMILES (default: 0)")
    score_p.add_argument("--descending", action="store_true",
                         help="Sort output by descending predicted value")

    # ── design ─────────────────────────────────────────────────────────────
    design_p = sub.add_parser("design", help="Generate analog suggestions for a query molecule")
    design_p.add_argument("smiles", help="Query molecule SMILES (quote it)")
    design_p.add_argument("--model", required=True, metavar="PATH",
                          help="Path to trained predictor (.joblib)")
    design_p.add_argument("--n-analogs", type=int, default=25, metavar="N",
                          help="Maximum analogs to return (default: 25)")
    design_p.add_argument("--direction", choices=["maximize", "minimize"],
                          default="maximize", help="Optimization direction (default: maximize)")
    design_p.add_argument("--no-mmp", action="store_true",
                          help="Disable MMP-based suggestions (faster)")
    design_p.add_argument("--output", default="design_suggestions.csv", metavar="PATH",
                          help="Output CSV file (default: design_suggestions.csv)")

    # ── ui ─────────────────────────────────────────────────────────────────
    sub.add_parser("ui", help="Launch the interactive Streamlit UI")

    args = parser.parse_args(argv)

    if args.command == "score":
        _cmd_score(args)
    elif args.command == "design":
        _cmd_design(args)
    elif args.command == "ui":
        _cmd_ui(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
