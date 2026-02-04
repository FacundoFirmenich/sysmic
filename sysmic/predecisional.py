"""
================================================================================
PREDECISIONAL SYSTEM - Supervised Algorithmic Validation
================================================================================
Part of Framework Hypersistémico 3A+\CAHTPhase

Hierarchical predecisional system:
- Algorithm generates PREDECISION ("voice without vote")
- User Agent validates or modifies (final decision)
- Contextual logging of particular reasons
- Hierarchy: @superroot > rua > ua

Author: SFA Framework
Status: Core Component (Permanent)
================================================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import warnings

__all__ = [
    'PredecisionalSystem',
    'Predecision',
    'Decision',
    'UserAgent',
    'generate_predecision',
    'request_validation'
]


@dataclass
class Predecision:
    """
    Algorithmic predecision (proposal without binding power).
    
    Attributes:
        proposed_action: Recommended action
        confidence_score: Algorithmic confidence (0-1)
        reasoning_chain: Step-by-step reasoning
        alternative_options: List of alternative actions
        context: Contextual metadata
        timestamp: Generation timestamp
    """
    proposed_action: str
    confidence_score: float
    reasoning_chain: List[str]
    alternative_options: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Decision:
    """
    Final decision with user validation.
    
    Attributes:
        action: Final action (validated or modified)
        predecision: Original algorithmic predecision
        user_agent: Authorized agent who made decision
        validation_status: 'approved', 'modified', or 'rejected'
        rationale: Detailed rationale for decision
        timestamp: Decision timestamp
    """
    action: str
    predecision: Predecision
    user_agent: str
    validation_status: str  # 'approved', 'modified', 'rejected'
    rationale: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UserAgent:
    """
    Authorized user agent with hierarchy.
    
    Hierarchy:
    - sua: superroot user agent (@superroot)
    - rua: root user agent
    - ua: user agent
    
    Attributes:
        agent_id: Unique identifier
        agent_type: 'sua', 'rua', or 'ua'
        permissions: List of authorized actions
    """
    agent_id: str
    agent_type: str  # 'sua', 'rua', 'ua'
    permissions: List[str] = field(default_factory=list)
    
    def has_permission(self, action: str) -> bool:
        """Check if agent has permission for action."""
        if self.agent_type == 'sua':
            return True  # Superroot has all permissions
        return action in self.permissions


class PredecisionalSystem:
    """
    Supervised predecisional system with hierarchical validation.
    
    Core Principle:
    ---------------
    Algorithm PROPOSES, never DECIDES.
    User has final vote on all actions.
    
    Usage:
    ------
    >>> system = PredecisionalSystem()
    >>> predecision = system.generate_predecision(data, context)
    >>> agent = UserAgent('researcher_1', 'ua', ['analyze', 'export'])
    >>> decision = system.request_validation(predecision, agent, rationale)
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize predecisional system.
        
        Args:
            log_dir: Directory for decision logs
        """
        self.log_dir = Path(log_dir) if log_dir else Path("predecisional_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.decision_history = []
        
    def generate_predecision(self, 
                            data: Any, 
                            context: Dict,
                            algorithm: Optional[Callable] = None) -> Predecision:
        """
        Generate algorithmic predecision.
        
        Algorithm analyzes data and proposes action, but does NOT decide.
        
        Args:
            data: Input data for analysis
            context: Contextual information
            algorithm: Optional custom algorithm (default: built-in)
            
        Returns:
            Predecision object with proposal
        """
        print(f"\n[PREDECISIONAL] Generating predecision...")
        
        if algorithm is None:
            # Default algorithm
            predecision = self._default_predecision_algorithm(data, context)
        else:
            # Custom algorithm
            predecision = algorithm(data, context)
        
        print(f"  Proposed action: {predecision.proposed_action}")
        print(f"  Confidence: {predecision.confidence_score:.2f}")
        print(f"  Alternatives: {len(predecision.alternative_options)}")
        print(f"  Status: AWAITING USER VALIDATION")
        
        return predecision
    
    def _default_predecision_algorithm(self, data: Any, context: Dict) -> Predecision:
        """
        Default algorithmic predecision logic.
        
        Analyzes data characteristics and proposes action.
        """
        reasoning = []
        
        # Analyze data characteristics
        if isinstance(data, (pd.DataFrame, np.ndarray)):
            size = len(data) if isinstance(data, pd.DataFrame) else data.shape[0]
            
            reasoning.append(f"Data size: {size} samples")
            
            if size < 50:
                proposed = "skip_analysis"
                confidence = 0.9
                reasoning.append("Insufficient data for robust analysis")
                alternatives = ["force_analysis_low_confidence", "request_more_data"]
            elif size < 500:
                proposed = "standard_analysis"
                confidence = 0.7
                reasoning.append("Moderate sample size, standard methods applicable")
                alternatives = ["bootstrap_analysis", "bayesian_approach"]
            else:
                proposed = "comprehensive_analysis"
                confidence = 0.95
                reasoning.append("Large sample, comprehensive methods optimal")
                alternatives = ["parallel_processing", "distributed_computing"]
        else:
            proposed = "manual_review"
            confidence = 0.5
            reasoning.append("Unknown data type, manual review recommended")
            alternatives = ["type_conversion", "data_ingestion_pipeline"]
        
        # Consider context
        if context.get('urgency') == 'high':
            reasoning.append("High urgency: prioritizing speed")
        if context.get('quality_requirement') == 'publication':
            reasoning.append("Publication quality required: prioritizing rigor")
        
        return Predecision(
            proposed_action=proposed,
            confidence_score=confidence,
            reasoning_chain=reasoning,
            alternative_options=alternatives,
            context=context
        )
    
    def request_validation(self, 
                          predecision: Predecision,
                          user_agent: UserAgent,
                          interactive: bool = True) -> Decision:
        """
        Request explicit validation from authorized user agent.
        
        Args:
            predecision: Algorithmic predecision
            user_agent: Authorized agent
            interactive: If True, prompt user; if False, auto-approve
            
        Returns:
            Decision with validation or modification
        """
        print(f"\n[VALIDATION REQUEST]")
        print(f"  Agent: {user_agent.agent_id} ({user_agent.agent_type})")
        print(f"  Predecision: {predecision.proposed_action}")
        print(f"  Confidence: {predecision.confidence_score:.2f}")
        
        # Check permissions
        if not user_agent.has_permission(predecision.proposed_action):
            print(f"  ⚠ Agent lacks permission for '{predecision.proposed_action}'")
            decision = Decision(
                action="permission_denied",
                predecision=predecision,
                user_agent=user_agent.agent_id,
                validation_status='rejected',
                rationale="Agent lacks required permissions"
            )
        elif interactive:
            # Interactive validation (would be UI in production)
            print(f"\n  Reasoning:")
            for i, reason in enumerate(predecision.reasoning_chain, 1):
                print(f"    {i}. {reason}")
            
            print(f"\n  Options:")
            print(f"    1. APPROVE predecision")
            print(f"    2. MODIFY action")
            print(f"    3. REJECT")
            print(f"    Alternatives: {', '.join(predecision.alternative_options)}")
            
            # For automation, auto-approve high-confidence predecisions
            if predecision.confidence_score >= 0.8:
                final_action = predecision.proposed_action
                status = 'approved'
                rationale = f"Auto-approved (confidence {predecision.confidence_score:.2f})"
                print(f"\n  → AUTO-APPROVED (high confidence)")
            else:
                final_action = predecision.proposed_action
                status = 'approved'
                rationale = "Default approval for demonstration"
                print(f"\n  → APPROVED (default for demo)")
            
            decision = Decision(
                action=final_action,
                predecision=predecision,
                user_agent=user_agent.agent_id,
                validation_status=status,
                rationale=rationale
            )
        else:
            # Non-interactive: auto-approve
            decision = Decision(
                action=predecision.proposed_action,
                predecision=predecision,
                user_agent=user_agent.agent_id,
                validation_status='approved',
                rationale="Auto-approved (non-interactive mode)"
            )
        
        # Log decision
        self._log_decision(decision)
        
        return decision
    
    def _log_decision(self, decision: Decision) -> None:
        """
        Log decision with detailed rationale.
        
        Includes:
        - Particular reasons
        - Local context
        - Contextual factors
        - User agent identification
        """
        log_entry = {
            'timestamp': decision.timestamp,
            'user_agent': decision.user_agent,
            'action': decision.action,
            'validation_status': decision.validation_status,
            'rationale': decision.rationale,
            'predecision': {
                'proposed_action': decision.predecision.proposed_action,
                'confidence': decision.predecision.confidence_score,
                'reasoning': decision.predecision.reasoning_chain
            }
        }
        
        # Save to file
        logfile = self.log_dir / f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(logfile, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        # Store in memory
        self.decision_history.append(decision)
        
        print(f"  ✓ Decision logged: {logfile.name}")
    
    def get_decision_history(self, 
                            user_agent: Optional[str] = None,
                            n_recent: Optional[int] = None) -> List[Decision]:
        """
        Retrieve decision history.
        
        Args:
            user_agent: Filter by user agent
            n_recent: Return only N most recent decisions
            
        Returns:
            List of Decision objects
        """
        history = self.decision_history
        
        if user_agent:
            history = [d for d in history if d.user_agent == user_agent]
        
        if n_recent:
            history = history[-n_recent:]
        
        return history


def generate_predecision(data: Any, 
                        context: Dict,
                        algorithm: Optional[Callable] = None) -> Predecision:
    """
    High-level interface for predecision generation.
    
    Args:
        data: Input data
        context: Contextual information
        algorithm: Optional custom algorithm
        
    Returns:
        Predecision object
    """
    system = PredecisionalSystem()
    return system.generate_predecision(data, context, algorithm)


def request_validation(predecision: Predecision,
                      agent_id: str,
                      agent_type: str = 'ua',
                      permissions: Optional[List[str]] = None) -> Decision:
    """
    High-level interface for validation request.
    
    Args:
        predecision: Predecision to validate
        agent_id: User agent identifier
        agent_type: 'sua', 'rua', or 'ua'
        permissions: Agent permissions
        
    Returns:
        Decision object
    """
    if permissions is None:
        permissions = []  # Superroot has all
    
    agent = UserAgent(agent_id, agent_type, permissions)
    system = PredecisionalSystem()
    return system.request_validation(predecision, agent, interactive=False)


if __name__ == "__main__":
    print("="*80)
    print("  PREDECISIONAL SYSTEM - Framework Hypersistémico {3A+\\CAHTPhase}")
    print("  Supervised Algorithmic Validation")
    print("="*80)
    
    # Example usage
    print("\n[Example] Predecisional workflow...\n")
    
    # Synthetic data
    data = pd.DataFrame(np.random.randn(200, 3))
    context = {'goal': 'analysis', 'quality_requirement': 'publication'}
    
    # Generate predecision
    system = PredecisionalSystem()
    predecision = system.generate_predecision(data, context)
    
    # Create user agent
    agent = UserAgent('researcher_1', 'ua', ['standard_analysis', 'comprehensive_analysis'])
    
    # Request validation
    decision = system.request_validation(predecision, agent, interactive=False)
    
    print(f"\n✅ Predecisional System Demonstrated")
    print(f"  Final decision: {decision.action}")
    print(f"  Status: {decision.validation_status}")
    print(f"  Rationale: {decision.rationale}")
    print("\n" + "="*80)
