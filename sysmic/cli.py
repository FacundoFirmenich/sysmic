"""
FRACTAL SYSTEM PRO - CLI
Interfaz de línea de comandos para el sistema de análisis fractal.
"""

import argparse
import sys
import pandas as pd
from pathlib import Path
from .system import Sysmic
from .infrastructure import SystemConfiguration, CertificationLevel

def main():
    parser = argparse.ArgumentParser(description="Sysmic v6.0")
    parser.add_argument("catalog", help="Path to input catalog CSV")
    parser.add_argument("--output", "-o", help="Output directory", default="results")
    parser.add_argument("--level", "-l", help="Certification level", 
                       choices=['basic', 'scientific', 'commercial', 'certified'],
                       default='scientific')
    parser.add_argument("--types", "-t", nargs="+", help="Analysis types to run")
    
    args = parser.parse_args()
    
    # Configurar
    config = SystemConfiguration(
        certification_level=CertificationLevel(args.level),
        output_dir=Path(args.output)
    )
    
    # Inicializar sistema
    system = Sysmic(config)
    
    # Cargar datos
    try:
        catalog_path = Path(args.catalog)
        if not catalog_path.exists():
            print(f"Error: File not found: {catalog_path}")
            sys.exit(1)
            
        print(f"Loading catalog: {catalog_path}")
        catalog = pd.read_csv(catalog_path)
        
        # Ejecutar análisis
        print(f"Starting analysis (Level: {args.level})...")
        result = system.analyze_catalog(catalog, analysis_types=args.types)
        
        if result.success:
            print("\nAnalysis Success!")
            print(f"ID: {result.analysis_id}")
            print(f"Time: {result.computation_time:.2f}s")
            
            # Print summary
            if result.fractal_dimension:
                cons = result.fractal_dimension.get('consensus', {})
                print(f"\nFractal Dimension (D): {cons.get('value', 'N/A'):.3f}")
                
            report_file = Path(args.output) / f"{result.analysis_id}_report.txt"
            with open(report_file, 'w') as f:
                f.write(result.generate_report())
            print(f"Report saved to: {report_file}")
            
        else:
            print("\nAnalysis Failed!")
            for err in result.errors:
                print(f"- {err}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Critical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
