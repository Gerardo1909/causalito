"""
Script para analizar datos etiquetados y entrenar modelo bayesiano.

Uso:
    python scripts/analyze_confidence.py --csv-path data/eval_log.csv
"""

import argparse
from pathlib import Path

from confidence.data_analyzer import DataAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analiza datos etiquetados con modelo Beta-Binomial"
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/eval_log.csv"),
        help="Ruta al CSV de evaluación",
    )

    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"[ERROR] No se encontró: {args.csv_path}")
        print("Primero ejecuta la aplicación para generar datos.")
        return

    # Analizar
    analyzer = DataAnalyzer(args.csv_path)
    high_conf, low_conf = analyzer.analyze_confidence_split()

    # Mostrar resultados
    print("\\n" + "=" * 60)
    print("ANÁLISIS BAYESIANO DE CONFIANZA")
    print("=" * 60)
    print(high_conf.summary())
    print(low_conf.summary())

    # Comparación
    print("\\n" + "=" * 60)
    print("COMPARACIÓN")
    print("=" * 60)
    improvement = (
        high_conf.inference_result.posterior_mean
        - low_conf.inference_result.posterior_mean
    )
    print(f"Mejora en P(correcta): {improvement:+.3f}")
    print(
        f"Relación: {high_conf.inference_result.posterior_mean / max(low_conf.inference_result.posterior_mean, 0.001):.1f}x"
    )


if __name__ == "__main__":
    main()
