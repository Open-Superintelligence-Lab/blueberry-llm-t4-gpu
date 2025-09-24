"""
Enhanced Muon Optimizer with Momentum Warmup Support

This module extends the original Muon optimizer to support various
momentum warmup schedules for research purposes.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Callable, Dict, Any


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.half()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


class MomentumScheduler:
    """Handles momentum scheduling for warmup experiments"""
    
    def __init__(self, schedule_config: Dict[str, Any], max_steps: int):
        self.schedule_config = schedule_config
        self.max_steps = max_steps
        self.step = 0
        
    def get_momentum(self, step: int) -> float:
        """Get momentum value for current step"""
        self.step = step
        schedule_type = self.schedule_config['type']
        warmup_steps = self.schedule_config.get('warmup_steps', self.max_steps // 4)
        
        if schedule_type == 'fixed':
            return self.schedule_config['momentum']
        
        elif schedule_type == 'linear':
            start_momentum = self.schedule_config['start_momentum']
            end_momentum = self.schedule_config['end_momentum']
            if step < warmup_steps:
                progress = step / warmup_steps
                return start_momentum + (end_momentum - start_momentum) * progress
            return end_momentum
        
        elif schedule_type == 'cosine':
            start_momentum = self.schedule_config['start_momentum']
            end_momentum = self.schedule_config['end_momentum']
            if step < warmup_steps:
                progress = step / warmup_steps
                cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
                return start_momentum + (end_momentum - start_momentum) * cosine_factor
            return end_momentum
        
        elif schedule_type == 'exponential':
            start_momentum = self.schedule_config['start_momentum']
            end_momentum = self.schedule_config['end_momentum']
            exponent = self.schedule_config.get('exponent', 2.0)
            if step < warmup_steps:
                progress = step / warmup_steps
                exp_factor = math.pow(progress, exponent)
                return start_momentum + (end_momentum - start_momentum) * exp_factor
            return end_momentum
        
        elif schedule_type == 'step':
            steps = self.schedule_config['steps']
            if step < warmup_steps:
                step_size = warmup_steps // len(steps)
                step_index = min(step // step_size, len(steps) - 1)
                return steps[step_index]
            return steps[-1]
        
        elif schedule_type == 'adaptive':
            # Simplified adaptive - would need gradient variance tracking
            base_momentum = self.schedule_config['base_momentum']
            if step < warmup_steps:
                progress = step / warmup_steps
                return base_momentum * progress
            return base_momentum
        
        elif schedule_type == 'delayed_linear':
            start_momentum = self.schedule_config['start_momentum']
            end_momentum = self.schedule_config['end_momentum']
            delay_steps = self.schedule_config['delay_steps']
            if step < delay_steps:
                return start_momentum
            elif step < delay_steps + warmup_steps:
                progress = (step - delay_steps) / warmup_steps
                return start_momentum + (end_momentum - start_momentum) * progress
            return end_momentum
        
        elif schedule_type == 'sigmoid':
            start_momentum = self.schedule_config['start_momentum']
            end_momentum = self.schedule_config['end_momentum']
            steepness = self.schedule_config.get('steepness', 5.0)
            if step < warmup_steps:
                progress = step / warmup_steps
                sigmoid_factor = 1 / (1 + math.exp(-steepness * (progress - 0.5)))
                return start_momentum + (end_momentum - start_momentum) * sigmoid_factor
            return end_momentum
        
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")


class MuonWithMomentumWarmup(torch.optim.Optimizer):
    """Muon optimizer with momentum warmup support for research"""
    
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, 
                 momentum_scheduler: Optional[MomentumScheduler] = None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.momentum_scheduler = momentum_scheduler
        self.step_count = 0
        
    @torch.no_grad()
    def step(self):
        """Override step to use dynamic momentum"""
        if self.momentum_scheduler:
            # Update momentum for all parameter groups
            current_momentum = self.momentum_scheduler.get_momentum(self.step_count)
            for group in self.param_groups:
                group['momentum'] = current_momentum
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
        
        self.step_count += 1
    
    def get_current_momentum(self) -> float:
        """Get the current momentum value"""
        if self.momentum_scheduler:
            return self.momentum_scheduler.get_momentum(self.step_count)
        return self.param_groups[0]['momentum']
    
    def get_step_count(self) -> int:
        """Get the current step count"""
        return self.step_count


class MuonOptimizerFactory:
    """Factory for creating Muon optimizers with different configurations"""
    
    @staticmethod
    def create_standard_muon(params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        """Create standard Muon optimizer (baseline)"""
        return MuonWithMomentumWarmup(params, lr, momentum, nesterov, ns_steps)
    
    @staticmethod
    def create_muon_with_warmup(params, schedule_config: Dict[str, Any], 
                               max_steps: int, lr=0.02, nesterov=True, ns_steps=5):
        """Create Muon optimizer with momentum warmup"""
        momentum_scheduler = MomentumScheduler(schedule_config, max_steps)
        return MuonWithMomentumWarmup(
            params, lr, momentum=0.95, nesterov=nesterov, ns_steps=ns_steps,
            momentum_scheduler=momentum_scheduler
        )
    
    @staticmethod
    def create_hybrid_optimizer(model, muon_schedule_config: Dict[str, Any], 
                               max_steps: int, lr=0.02, weight_decay=0.1):
        """Create hybrid Muon + AdamW optimizer setup"""
        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if (param.ndim == 2 and 
                'token_embedding' not in name and 
                'norm' not in name and 
                param.requires_grad):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
        print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

        muon_optimizer = MuonOptimizerFactory.create_muon_with_warmup(
            muon_params, muon_schedule_config, max_steps, lr
        )
        adamw_optimizer = torch.optim.AdamW(
            adamw_params, lr=lr*0.1, weight_decay=weight_decay
        )

        return [muon_optimizer, adamw_optimizer]


# Convenience functions for common momentum schedules
def create_linear_warmup_schedule(start_momentum=0.0, end_momentum=0.95, warmup_steps=250):
    """Create linear momentum warmup schedule"""
    return {
        'name': 'linear_warmup',
        'type': 'linear',
        'start_momentum': start_momentum,
        'end_momentum': end_momentum,
        'warmup_steps': warmup_steps,
    }

def create_cosine_warmup_schedule(start_momentum=0.0, end_momentum=0.95, warmup_steps=250):
    """Create cosine momentum warmup schedule"""
    return {
        'name': 'cosine_warmup',
        'type': 'cosine',
        'start_momentum': start_momentum,
        'end_momentum': end_momentum,
        'warmup_steps': warmup_steps,
    }

def create_exponential_warmup_schedule(start_momentum=0.1, end_momentum=0.95, 
                                     warmup_steps=250, exponent=2.0):
    """Create exponential momentum warmup schedule"""
    return {
        'name': 'exponential_warmup',
        'type': 'exponential',
        'start_momentum': start_momentum,
        'end_momentum': end_momentum,
        'warmup_steps': warmup_steps,
        'exponent': exponent,
    }

def create_fixed_momentum_schedule(momentum=0.95):
    """Create fixed momentum schedule (baseline)"""
    return {
        'name': f'fixed_{int(momentum*100):03d}',
        'type': 'fixed',
        'momentum': momentum,
        'warmup_steps': 0,
    }
