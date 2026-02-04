"""
Sysmic Interactive Session.
Proporciona la interfaz interactiva principal para el usuario.
"""

import sys
import time
import pandas as pd
from pathlib import Path
from .system import Sysmic
from .infrastructure import SystemConfiguration, CertificationLevel
from .ascii_art import show_splash, print_status, print_menu_header, input_clean, Colors

class SysmicInteractive:
    def __init__(self):
        self.config = SystemConfiguration()
        self.system = Sysmic(self.config)
        self.current_catalog = None
        self.catalog_path = None
        
    def start(self):
        show_splash()
        self._main_menu()
        
    def _main_menu(self):
        while True:
            print_menu_header("MAIN MENU")
            print("1. Load Catalog (CSV)")
            print("2. Run Full Analysis")
            print("3. System Status")
            print("4. Configure System")
            print("5. Exit")
            
            choice = input_clean("Select option")
            
            if choice == '1':
                self._load_catalog_menu()
            elif choice == '2':
                self._run_analysis_menu()
            elif choice == '3':
                self._show_status()
            elif choice == '4':
                self._configure_menu()
            elif choice == '5':
                print_status("Shutting down Sysmic framework...", "SYSTEM")
                sys.exit(0)
            else:
                print_status("Invalid option", "WARNING")
                
    def _load_catalog_menu(self):
        print_menu_header("LOAD CATALOG")
        path_str = input_clean("Enter path to CSV file")
        
        path = Path(path_str)
        if not path.exists():
            print_status(f"File not found: {path}", "ERROR")
            return
            
        try:
            print_status("Reading CSV...", "INFO")
            self.current_catalog = pd.read_csv(path)
            self.catalog_path = path
            
            # Simple validation
            required = ['latitude', 'longitude', 'depth', 'mag', 'time']
            missing = [c for c in required if c not in self.current_catalog.columns]
            
            if missing:
                print_status(f"Missing columns: {missing}", "WARNING")
            else:
                print_status(f"Catalog loaded successfully: {len(self.current_catalog)} events", "SUCCESS")
                
        except Exception as e:
            print_status(f"Error loading catalog: {e}", "ERROR")

    def _run_analysis_menu(self):
        if self.current_catalog is None:
            print_status("No catalog loaded. Please load a catalog first.", "WARNING")
            return
            
        print_menu_header("RUN ANALYSIS")
        print("1. Standard Scan (Fractal + Temporal)")
        print("2. Deep Scan (Standard + Multifractal)")
        print("3. Scientific Validation Scan (Deep + Validation)")
        print("4. Full Certified Audit (All Modules + Bayesian)")
        
        choice = input_clean("Select analysis profile")
        
        types = []
        if choice == '1':
            types = ['fractal_dimension', 'temporal']
        elif choice == '2':
            types = ['fractal_dimension', 'temporal', 'multifractal']
        elif choice == '3':
            types = ['fractal_dimension', 'temporal', 'multifractal', 'validation']
        elif choice == '4':
            types = ['fractal_dimension', 'temporal', 'multifractal', 'validation', 'features', 'bayesian']
        else:
            print_status("Invalid profile", "WARNING")
            return
            
        print_status(f"Starting analysis on {len(self.current_catalog)} events...", "SYSTEM")
        start_time = time.time()
        
        try:
            result = self.system.analyze_catalog(self.current_catalog, analysis_types=types)
            
            if result.success:
                print("\n" + "="*50)
                print_status("ANALYSIS COMPLETE", "SUCCESS")
                print(f"ID: {result.analysis_id}")
                print(f"Time: {result.computation_time:.2f}s")
                
                if result.fractal_dimension:
                     cons = result.fractal_dimension.get('consensus', {})
                     print(f"{Colors.BOLD}Fractal Dimension:{Colors.ENDC} {cons.get('value', 'N/A'):.4f}")
                     
                # Save report
                output_dir = Path("results")
                output_dir.mkdir(exist_ok=True)
                report_path = output_dir / f"{result.analysis_id}_report.txt"
                with open(report_path, 'w') as f:
                    f.write(result.generate_report())
                print_status(f"Report saved to: {report_path}", "INFO")
                
            else:
                print_status("Analysis failed.", "ERROR")
                for err in result.errors:
                    print(f"- {err}")
                    
        except Exception as e:
            print_status(f"Critical execution error: {e}", "ERROR")
            
    def _show_status(self):
        print_menu_header("SYSTEM STATUS")
        print(f"System ID: {self.config.system_id}")
        print(f"Version: {self.config.version}")
        print(f"Active Level: {self.config.certification_level.name}")
        print(f"Loaded Modules: {list(self.system.modules.keys())}")
        if self.current_catalog is not None:
            print(f"Active Catalog: {self.catalog_path} ({len(self.current_catalog)} events)")
        else:
            print("Active Catalog: None")
            
    def _configure_menu(self):
        print_menu_header("CONFIGURATION")
        print(f"Current Level: {self.config.certification_level.name}")
        print("Available Levels: LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4")
        
        new_level = input_clean("Enter new level name (or ENTER to cancel)")
        if new_level:
            try:
                # Map input string to Enum
                level_enum = CertificationLevel[new_level.upper()]
                self.config = SystemConfiguration(certification_level=level_enum)
                # Re-init system with new config
                self.system = Sysmic(self.config)
                print_status(f"System re-initialized at {level_enum.name}", "SUCCESS")
            except KeyError:
                print_status("Invalid level name", "ERROR")

if __name__ == '__main__':
    session = SysmicInteractive()
    session.start()
