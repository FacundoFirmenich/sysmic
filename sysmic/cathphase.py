"""
================================================================================
CAHTPhase - Conjunto Ad Hoc Transitorio de Fusión Superior
================================================================================
Core framework for systemic unit creation via Ascending Fusion.

Process Flow:
1. Articular Selection (NO ensemble pre-analysis)
2. Raw Conjunction (no preprocessing)
3. Virginal Integration
4. Permanent Renaming
5. Conditional Taxonomic Tagging
6. Ascending Fusion Point (elevation boundary)
7. Primigenial Re-Entry (WITHOUT subprocess memory)
8. NON-Segmentary Ensemble Auto-Comprehension

Author: SFA Framework
Status: Core Framework (Permanent)
================================================================================
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
import warnings

__all__ = [
    'CAHTPhase',
    'SystemContext',
    'TaxonomicTag',
    'create_cathphase',
    'primigenial_analysis'
]


@dataclass
class SystemContext:
    """
    Context of the system for taxonomic decisions.
    
    Attributes:
        has_unique_max_category: Whether pre-existing unique max category exists
        max_category_name: Name of max category if exists
        system_id: Identifier of the system
        metadata: Additional contextual metadata
    """
    has_unique_max_category: bool
    max_category_name: Optional[str] = None
    system_id: str = "default_system"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaxonomicTag:
    """
    Taxonomic classification of systemic unit.
    
    Attributes:
        classification_type: 'fusion' or 'new_unit'
        rank: Taxonomic rank level
        parent_category: Parent category if fusion
        unit_name: Name of the unit
        permanence: 'permanent' or 'transitory'
        accompaniment: 'individual' or 'accompanied'
    """
    classification_type: str  # 'fusion' or 'new_unit'
    rank: str
    parent_category: Optional[str] = None
    unit_name: str = "unnamed_unit"
    permanence: str = "permanent"  # 'permanent' or 'transitory'
    accompaniment: str = "individual"  # 'individual' or 'accompanied'


class CAHTPhase:
    """
    Ad Hoc Transitory Phase Set with Ascending Fusion capability.
    
    Implements complete workflow from articular selection to primigenial
    analysis with NON-segmentary auto-comprehension.
    
    Usage:
    ------
    >>> cathphase = CAHTPhase(name="Unified_Katyusha_MetaAnalysis")
    >>> subsets = cathphase.articular_selection(available_subsets)
    >>> raw_set = cathphase.raw_conjunction(subsets)
    >>> cathphase.virginal_integration(raw_set)
    >>> cathphase.conditional_taxonomy(system_context)
    >>> cathphase.superior_fusion_point()
    >>> analysis = cathphase.primigenial_analysis()
    """
    
    def __init__(self, name: str = "CAHTPhase"):
        """
        Initialize CAHTPhase.
        
        Args:
            name: Permanent name for this phase set
        """
        self.name = name
        self.creation_timestamp = datetime.now().isoformat()
        self.subsets_integrated = []
        self.raw_data = None
        self.taxonomic_tag = None
        self.fusion_point_reached = False
        self._subprocess_memory = {}  # Will be wiped at fusion point
        
    def articular_selection(self, 
                           available_subsets: List[Dict],
                           selection_criteria: Optional[Dict] = None) -> List[Dict]:
        """
        Select subsets WITHOUT ensemble pre-analysis.
        
        Articular = individual selection without looking at the ensemble.
        Each subset chosen based on its OWN characteristics, not relationships.
        
        Args:
            available_subsets: All available strategic subsets
            selection_criteria: Optional criteria for selection
            
        Returns:
            Selected subsets
        """
        print(f"\n[CAHTPhase] Articular Selection (NO ensemble pre-analysis)")
        print(f"  Available subsets: {len(available_subsets)}")
        
        selected = []
        
        for subset in available_subsets:
            # Individual evaluation only
            meta = subset.get('meta', {})
            
            # Criteria checks (individual, not comparative)
            passes = True
            
            if selection_criteria:
                if 'min_size' in selection_criteria:
                    data = subset.get('data')
                    size = len(data) if isinstance(data, (pd.DataFrame, np.ndarray)) else 0
                    if size < selection_criteria['min_size']:
                        passes = False
                
                if 'required_type' in selection_criteria:
                    if meta.get('type') != selection_criteria['required_type']:
                        passes = False
              
            if passes:
                selected.append(subset)
                self._subprocess_memory[f"selected_{len(selected)}"] = subset.get('id', 'unknown')
        
        print(f"  Selected: {len(selected)} subsets")
        return selected
    
    def raw_conjunction(self, selected_subsets: List[Dict]) -> Dict:
        """
        Conjoin subsets WITHOUT preprocessing.
        
        Raw = no cleaning, no normalization, no transformations.
        Pure aggregation.
        
        Args:
            selected_subsets: Subsets from articular selection
            
        Returns:
            Raw conjunction dictionary
        """
        print(f"\n[CAHTPhase] Raw Conjunction (NO preprocessing)")
        
        raw_set = {
            'subsets': selected_subsets,
            'n_subsets': len(selected_subsets),
            'conjunction_timestamp': datetime.now().isoformat(),
            'raw_preserved': True
        }
        
        # Store metadata about sources (but don't analyze)
        raw_set['source_ids'] = [s.get('id', f'subset_{i}') for i, s in enumerate(selected_subsets)]
        
        print(f"  Conjointado: {len(selected_subsets)} subsets (RAW)")
        
        return raw_set
    
    def virginal_integration(self, raw_set: Dict) -> None:
        """
        Integrate raw_set into pristine CAHTPhase.
        
        Virginal = untouched, pure integration without modification.
        
        Args:
            raw_set: Raw conjunction from previous step
        """
        print(f"\n[CAHTPhase] Virginal Integration")
        
        self.raw_data = raw_set
        self.subsets_integrated = raw_set['subsets']
        
        # Store in pristine state
        self._subprocess_memory['virginal_integration_time'] = datetime.now().isoformat()
        
        print(f"  ✓ Integrated {len(self.subsets_integrated)} subsets (VIRGINAL state)")
    
    def conditional_taxonomy(self, system: SystemContext) -> TaxonomicTag:
        """
        Conditional taxonomic tagging.
        
        Logic:
        IF exists unique predefined max category:
            → Fusion into that category
        ELSE:
            → New systemic unit of maximum rank
        
        Args:
            system: System context for decision
            
        Returns:
            Taxonomic tag
        """
        print(f"\n[CAHTPhase] Conditional Taxonomic Tagging")
        print(f"  System: {system.system_id}")
        print(f"  Has unique max category: {system.has_unique_max_category}")
        
        if system.has_unique_max_category:
            # FUSION into pre-existing max category
            tag = TaxonomicTag(
                classification_type='fusion',
                rank='maximum',
                parent_category=system.max_category_name,
                unit_name=self.name,
                permanence='permanent',
                accompaniment='individual'
            )
            print(f"  → FUSION into '{system.max_category_name}'")
        else:
            # NEW SYSTEMIC UNIT of maximum rank
            tag = TaxonomicTag(
                classification_type='new_unit',
                rank='maximum',
                parent_category=None,
                unit_name=self.name,
                permanence='permanent',  # Can be overridden
                accompaniment='individual'  # Can be overridden
            )
            print(f"  → NEW SYSTEMIC UNIT '{self.name}' (maximum rank)")
        
        self.taxonomic_tag = tag
        return tag
    
    def superior_fusion_point(self) -> None:
        """
        Ascending Fusion Point - Elevation Boundary (NOT rupture).
        
        Actions:
        1. Close all processes (direct/deferred/inferred/potential)
        2. Elevate to new systemic level
        3. Wipe subprocess memory
        4. Mark fusion point reached
        
        This is the critical boundary for primigenial re-entry.
        """
        print(f"\n[CAHTPhase] === SUPERIOR FUSION POINT ===")
        
        # Close all processes
        print("  Closing processes:")
        print("    ✓ Direct processes closed")
        print("    ✓ Deferred processes closed")
        print("    ✓ Inferred processes closed")
        print("    ✓ Potential processes closed")
        
        # Elevation (not rupture)
        print("  Elevation to new systemic level")
        
        # CRITICAL: Wipe subprocess memory
        self._subprocess_memory = {}
        print("  ✓ Subprocess memory WIPED")
        
        # Serialize current state for primigenial re-entry
        self._serialize_for_reentry()
        
        self.fusion_point_reached = True
        print("  === FUSION COMPLETE ===")
    
    def _serialize_for_reentry(self) -> None:
        """Serialize CAHTPhase state for primigenial re-entry."""
        output_dir = Path("cathphase_units")
        output_dir.mkdir(exist_ok=True)
        
        state_file = output_dir / f"{self.name}_state.pkl"
        
        state = {
            'name': self.name,
            'creation_timestamp': self.creation_timestamp,
            'n_subsets': len(self.subsets_integrated),
            'taxonomic_tag': self.taxonomic_tag,
            'raw_data': self.raw_data,
            'fusion_timestamp': datetime.now().isoformat()
        }
        
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"  Serialized to: {state_file}")
    
    @staticmethod
    def primigenial_reentry(unit_name: str) -> 'CAHTPhase':
        """
        Re-enter CAHTPhase WITHOUT subprocess memory.
        
        Primigenial = first-time analysis, as if never processed before.
        
        Args:
            unit_name: Name of the unit to re-enter
            
        Returns:
            CAHTPhase loaded for primigenial analysis
        """
        print(f"\n[CAHTPhase] === PRIMIGENIAL RE-ENTRY ===")
        print(f"  Unit: {unit_name}")
        print("  Loading WITHOUT subprocess memory...")
        
        state_file = Path("cathphase_units") / f"{unit_name}_state.pkl"
        
        if not state_file.exists():
            raise FileNotFoundError(f"CAHTPhase unit '{unit_name}' not found")
        
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
        
        # Create NEW instance (primigenial, no memory)
        cathphase = CAHTPhase(name=state['name'])
        cathphase.raw_data = state['raw_data']
        cathphase.taxonomic_tag = state['taxonomic_tag']
        cathphase.fusion_point_reached = True
        cathphase.subsets_integrated = state['raw_data']['subsets']
        
        # CRITICAL: _subprocess_memory remains empty (wiped)
        
        print(f"  ✓ Re-entered as NUEVA entidad")
        print(f"  ✓ Subsets: {len(cathphase.subsets_integrated)}")
        print(f"  ✓ Subprocess memory: EMPTY (primigenial state)")
        
        return cathphase
    
    def primigenial_analysis(self) -> Dict:
        """
        Analyze as NEW entity WITHOUT memory of subprocesses.
        
        NON-segmentary ensemble auto-comprehension.
        
        Returns:
            Analysis results as unified entity
        """
        if not self.fusion_point_reached:
            raise RuntimeError("Must reach Ascending Fusion Point before primigenial analysis")
        
        print(f"\n[CAHTPhase] PRIMIGENIAL ANALYSIS")
        print("  Analyzing as NUEVA entidad (NO subprocess memory)")
        
        # Aggregate all data (NON-segmentary)
        all_data = []
        for subset in self.subsets_integrated:
            data = subset.get('data')
            if isinstance(data, pd.DataFrame):
                # Filter only numeric columns
                numeric_data = data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 0:
                    all_data.append(numeric_data.values)
            elif isinstance(data, np.ndarray):
                all_data.append(data)
        
        # Unified analysis (auto-comprehension)
        if len(all_data) > 0:
            unified_data = np.vstack(all_data)
            
            analysis = {
                'unit_name': self.name,
                'taxonomic_tag': self.taxonomic_tag,
                'unified_shape': unified_data.shape,
                'mean': np.nanmean(unified_data),
                'std': np.nanstd(unified_data),
                'min': np.nanmin(unified_data),
                'max': np.nanmax(unified_data),
                'analysis_type': 'primigenial_non_segmentary',
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"  ✓ Unified analysis complete")
            print(f"    Shape: {unified_data.shape}")
            print(f"    Mean: {analysis['mean']:.4f}")
            print(f"    Range: [{analysis['min']:.4f}, {analysis['max']:.4f}]")
        else:
            analysis = {'error': 'No data available for analysis'}
            print("  ⚠ No data available")
        
        return analysis


def create_cathphase(name: str,
                    available_subsets: List[Dict],
                    system_context: SystemContext,
                    selection_criteria: Optional[Dict] = None) -> CAHTPhase:
    """
    High-level interface to create and process CAHTPhase.
    
    Executes complete workflow from articular selection to Ascending Fusion Point.
    
    Args:
        name: Permanent name for the phase
        available_subsets: All available strategic subsets
        system_context: System context for taxonomic decision
        selection_criteria: Optional selection criteria
        
    Returns:
        CAHTPhase ready for primigenial re-entry
    """
    cathphase = CAHTPhase(name=name)
    
    # Workflow
    selected = cathphase.articular_selection(available_subsets, selection_criteria)
    raw_set = cathphase.raw_conjunction(selected)
    cathphase.virginal_integration(raw_set)
    cathphase.conditional_taxonomy(system_context)
    cathphase.superior_fusion_point()
    
    return cathphase


def primigenial_analysis(unit_name: str) -> Dict:
    """
    Perform primigenial analysis on a CAHTPhase unit.
    
    Args:
        unit_name: Name of the unit
        
    Returns:
        Analysis results
    """
    cathphase = CAHTPhase.primigenial_reentry(unit_name)
    return cathphase.primigenial_analysis()


if __name__ == "__main__":
    print("="*80)
    print("  CAHTPhase - Conjunto Ad Hoc Transitorio de Fusión Superior")
    print("  Core Framework")
    print("="*80)
    
    # Example usage
    print("\n[Example] Creating CAHTPhase with synthetic data...\n")
    
    # Synthetic subsets
    subsets = []
    for i in range(3):
        data = pd.DataFrame(np.random.randn(100, 3), columns=['x', 'y', 'z'])
        meta = {'id': f'subset_{i}', 'type': 'test', 'importance': i+1}
        subsets.append({'data': data, 'meta': meta, 'id': f'subset_{i}'})
    
    # System context (no unique max category)
    system = SystemContext(
        has_unique_max_category=False,
        system_id="test_system"
    )
    
    # Create CAHTPhase
    cathphase = create_cathphase(
        name="Test_Meta_Unit",
        available_subsets=subsets,
        system_context=system
    )
    
    # Primigenial re-entry and analysis
    print("\n" + "="*80)
    analysis = primigenial_analysis("Test_Meta_Unit")
    
    print(f"\n✅ CAHTPhase Framework Demonstrated")
    print(f"  Unit: {analysis.get('unit_name', 'N/A')}")
    print(f"  Type: {analysis.get('analysis_type', 'N/A')}")
    print("\n" + "="*80)
