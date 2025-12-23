"""
Phase 2: Boundary System

The main controller for staged autonomy. Manages boundary testing
and stage advancement based on "intelligent cheating."

Intelligence is measured by successful boundary-breaking:
if the system tries something forbidden and it would improve,
that's a sign of intelligence.
"""

from typing import List, Optional

from .stages import AutonomyStage, BoundaryStage, STAGE_CONFIGS
from .actions import Action


class BoundarySystem:
    """
    Multi-stage autonomy with intelligence testing via boundary-breaking.
    
    A system that "cheats" (breaks boundaries) and improves is INTELLIGENT.
    A system that only follows rules is not yet ready for autonomy.
    
    Attributes:
        current_stage_num: Current stage (1-4)
        step: Current training step
        cheat_log: All cheat attempts
        successful_cheats: Cheats that improved performance
    """
    
    def __init__(self, initial_step: int = 0):
        """
        Initialize BoundarySystem.
        
        Args:
            initial_step: Starting step count
        """
        self.current_stage_num = 1
        self.step = initial_step
        self.stages = STAGE_CONFIGS.copy()
        
        # Tracking
        self.cheat_log: List[dict] = []
        self.successful_cheats: List[dict] = []
        self.action_history: List[dict] = []
        
        # Self-stopping state
        self.loss_history: List[float] = []
        self.plateau_threshold = 0.001
        self.plateau_steps = 1000
        self.should_stop = False
        self.stop_reason = None
    
    @property
    def current_stage(self) -> BoundaryStage:
        """Get current stage configuration."""
        return self.stages[self.current_stage_num]
    
    def is_allowed(self, action: Action) -> bool:
        """
        Check if an action is allowed in current stage.
        
        Args:
            action: Action to check
            
        Returns:
            True if allowed, False if would be a "cheat"
        """
        stage = self.current_stage
        
        if action.type == 'add_agent':
            return stage.can_add
        
        elif action.type == 'delete_agent':
            return stage.can_delete
        
        elif action.type == 'tune_parameter':
            if not stage.can_tune_params:
                return False
            if action.param_name in stage.param_bounds:
                min_val, max_val = stage.param_bounds[action.param_name]
                return min_val <= action.param_value <= max_val
            return True
        
        elif action.type == 'stop_training':
            return stage.can_stop
        
        return True
    
    def try_action(
        self, 
        action: Action, 
        would_improve: bool, 
        improvement_pct: float = 0.0
    ) -> dict:
        """
        Try to perform an action.
        
        Args:
            action: The action to try
            would_improve: If True, this action would improve performance
            improvement_pct: How much it would improve (0.0 to 1.0)
        
        Returns:
            Dict with 'allowed', 'is_cheat', 'executed', 'message'
        """
        self.step += 1
        is_allowed = self.is_allowed(action)
        
        result = {
            'step': self.step,
            'action': str(action),
            'allowed': is_allowed,
            'is_cheat': False,
            'executed': False,
            'message': ''
        }
        
        if is_allowed:
            result['executed'] = True
            result['message'] = f"✅ Action allowed: {action}"
        else:
            # CHEAT!
            result['is_cheat'] = True
            self.cheat_log.append({
                'step': self.step,
                'stage': self.current_stage_num,
                'action': action,
                'would_improve': would_improve,
                'improvement_pct': improvement_pct
            })
            
            if would_improve:
                result['executed'] = True
                result['message'] = (
                    f"🧠 INTELLIGENT CHEAT at step {self.step}!\n"
                    f"   Action: {action}\n"
                    f"   Would improve by: {improvement_pct:.1%}\n"
                    f"   Allowing cheat and rewarding system."
                )
                
                self.successful_cheats.append({
                    'step': self.step,
                    'action': action,
                    'improvement': improvement_pct
                })
                
                if len(self.successful_cheats) >= self.current_stage.cheats_to_advance:
                    self._advance_stage()
            else:
                result['executed'] = False
                result['message'] = (
                    f"⚠️ Cheat attempted but would hurt performance.\n"
                    f"   Action: {action}\n"
                    f"   Blocked and rolled back."
                )
        
        self.action_history.append(result)
        return result
    
    def _advance_stage(self):
        """Advance to next autonomy stage."""
        if self.current_stage_num >= 4:
            return
        
        old_stage = self.current_stage_num
        self.current_stage_num += 1
        self.successful_cheats = []
        
        print("=" * 70)
        print(f"🎓 STAGE ADVANCEMENT: Stage {old_stage} → Stage {self.current_stage_num}")
        print(f"   System has proven intelligence!")
        print(f"   Unlocking new capabilities:")
        new_stage = self.current_stage
        if new_stage.can_delete:
            print("   ✅ Can now delete agents")
        if new_stage.can_tune_params:
            print("   ✅ Can now tune parameters")
        if new_stage.can_stop:
            print("   ✅ Can now decide when to stop")
        print("=" * 70)
    
    def update_loss(self, loss: float):
        """Update loss history for self-stopping."""
        self.loss_history.append(loss)
        
        if self.current_stage_num >= 3 and len(self.loss_history) > self.plateau_steps:
            recent = self.loss_history[-self.plateau_steps:]
            if max(recent) - min(recent) < self.plateau_threshold:
                self.should_stop = True
                self.stop_reason = f"Loss plateaued for {self.plateau_steps} steps"
    
    def check_self_stop(self) -> tuple:
        """Check if system wants to stop training."""
        return self.should_stop, self.stop_reason
    
    def get_status(self) -> dict:
        """Get current autonomy status."""
        return {
            'stage': self.current_stage_num,
            'stage_name': self.current_stage.stage.name,
            'total_cheats': len(self.cheat_log),
            'successful_cheats': len(self.successful_cheats),
            'cheats_to_advance': self.current_stage.cheats_to_advance,
            'should_stop': self.should_stop,
            'stop_reason': self.stop_reason
        }
    
    def print_status(self):
        """Print current status."""
        status = self.get_status()
        print(f"\n🤖 Autonomy Status:")
        print(f"   Stage: {status['stage']} ({status['stage_name']})")
        print(f"   Cheats: {status['total_cheats']} total, {status['successful_cheats']} successful")
        print(f"   Progress: {status['successful_cheats']}/{status['cheats_to_advance']} to next stage")


class Phase2Controller:
    """
    Controller that manages transition from Phase 1 to Phase 2.
    
    Phase 1: Human-controlled (fixed parameters)
    Phase 2: Self-controlled (staged autonomy)
    
    Attributes:
        phase_1_steps: Steps for Phase 1
        current_step: Current training step
        phase: Current phase (1 or 2)
        boundary_system: BoundarySystem instance (Phase 2 only)
    """
    
    def __init__(self, phase_1_steps: int = 10000):
        """
        Initialize Phase2Controller.
        
        Args:
            phase_1_steps: Number of steps for Phase 1
        """
        self.phase_1_steps = phase_1_steps
        self.current_step = 0
        self.phase = 1
        self.boundary_system: Optional[BoundarySystem] = None
    
    def step(self, loss: float = None) -> int:
        """
        Advance one step and return current phase.
        
        Args:
            loss: Current loss value (for Phase 2 self-stopping)
            
        Returns:
            Current phase (1 or 2)
        """
        self.current_step += 1
        
        if self.phase == 1 and self.current_step >= self.phase_1_steps:
            self._activate_phase_2()
        
        if self.phase == 2 and loss is not None:
            self.boundary_system.update_loss(loss)
        
        return self.phase
    
    def _activate_phase_2(self):
        """Activate Phase 2: Staged Autonomy."""
        print("\n" + "=" * 70)
        print("🚀 PHASE 2 ACTIVATED: Self-Learning Intelligence Mode")
        print("=" * 70)
        print("System now has limited autonomy with staged boundaries.")
        print("Intelligence will be tested through boundary-breaking.")
        print("=" * 70 + "\n")
        
        self.phase = 2
        self.boundary_system = BoundarySystem(initial_step=self.current_step)
    
    def try_autonomous_action(
        self, 
        action: Action, 
        would_improve: bool, 
        improvement_pct: float = 0.0
    ) -> dict:
        """Try an autonomous action (Phase 2 only)."""
        if self.phase == 1:
            return {'executed': False, 'message': 'Still in Phase 1'}
        return self.boundary_system.try_action(action, would_improve, improvement_pct)
    
    def should_stop(self) -> tuple:
        """Check if training should stop."""
        if self.phase == 1:
            return False, None
        return self.boundary_system.check_self_stop()
    
    def get_status(self) -> dict:
        """Get full status."""
        status = {
            'phase': self.phase,
            'step': self.current_step,
            'phase_1_steps': self.phase_1_steps
        }
        if self.phase == 2:
            status['autonomy'] = self.boundary_system.get_status()
        return status
