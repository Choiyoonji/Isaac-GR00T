"""
Camera MoE (Mixture of Experts) for GR00T N1d6 Multi-Camera Input Processing

This module implements a router-based gating mechanism for fusing multiple camera inputs.
Unlike traditional MoE with expert networks, this uses direct feature gating.

Camera Setup for GR00T:
- cam1: L515/External camera (base, always included)
- cam2: Left Wrist camera (gated)
- cam3: Right Wrist camera (gated)

Router Input: cam1 features + prompt + state
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CameraRouterConfig:
    """Configuration for Camera Router."""
    
    embed_dim: int = 2048  # Eagle backbone embedding dimension
    state_dim: int = 29  # Robot state dimension
    num_experts: int = 2  # Number of auxiliary cameras to gate (cam2, cam3)
    router_hidden_dim: int = 512  # Router MLP hidden dimension
    router_temperature: float = 1.0  # Softmax temperature for routing weights
    use_gumbel_softmax: bool = False  # Use Gumbel-Softmax during training
    gumbel_temperature: float = 1.0  # Gumbel-Softmax temperature
    use_attention_pooling: bool = True  # Use learnable attention pooling for prompt
    use_learnable_scales: bool = True  # Use learnable scaling for gated features
    

class CameraRouter(nn.Module):
    """
    Router that predicts gating weights for auxiliary cameras.
    
    Input: cam1 (base) features + prompt embeddings + robot state
    Output: Routing weights [batch, num_experts] for cam2 and cam3
    """
    
    def __init__(self, config: CameraRouterConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.state_dim = config.state_dim
        self.num_experts = config.num_experts
        self.temperature = config.router_temperature
        self.use_gumbel = config.use_gumbel_softmax
        self.gumbel_temp = config.gumbel_temperature
        self.use_attention_pooling = config.use_attention_pooling
        
        # Input dimension: cam1_pooled (embed_dim) + prompt_pooled (embed_dim) + state (state_dim)
        input_dim = self.embed_dim + self.embed_dim + self.state_dim
        
        # Learnable attention pooling for prompt (optional)
        if self.use_attention_pooling:
            self.prompt_attention_query = nn.Linear(self.embed_dim, self.embed_dim)
            nn.init.normal_(self.prompt_attention_query.weight, mean=0.0, std=0.02)
        
        # Router MLP: input -> 512 -> 256 -> num_experts
        self.router_linear1 = nn.Linear(input_dim, config.router_hidden_dim)
        self.router_linear2 = nn.Linear(config.router_hidden_dim, config.router_hidden_dim // 2)
        self.router_linear3 = nn.Linear(config.router_hidden_dim // 2, self.num_experts)
        
        # Initialize weights
        nn.init.normal_(self.router_linear1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.router_linear2.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.router_linear3.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router_linear1.bias)
        nn.init.zeros_(self.router_linear2.bias)
        nn.init.zeros_(self.router_linear3.bias)
    
    def _attention_pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Learnable attention pooling for prompt tokens.
        
        Args:
            tokens: [batch, seq_len, embed_dim]
        
        Returns:
            pooled: [batch, embed_dim]
        """
        # Query: [batch, 1, embed_dim]
        query = self.prompt_attention_query(tokens.mean(dim=1, keepdim=True))
        
        # Attention scores: [batch, seq_len]
        scale = self.embed_dim ** 0.5
        attention_scores = torch.sum(query * tokens, dim=-1) / scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weighted sum: [batch, embed_dim]
        pooled = torch.sum(tokens * attention_weights.unsqueeze(-1), dim=1)
        return pooled
    
    def _gumbel_softmax(
        self, logits: torch.Tensor, temperature: float = 1.0, hard: bool = False
    ) -> torch.Tensor:
        """
        Gumbel-Softmax sampling for differentiable discrete sampling.
        
        Args:
            logits: [batch, num_experts]
            temperature: Temperature for Gumbel-Softmax
            hard: If True, use straight-through estimator
        
        Returns:
            samples: [batch, num_experts]
        """
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        gumbel_logits = (logits + gumbel_noise) / temperature
        
        # Softmax
        y_soft = F.softmax(gumbel_logits, dim=-1)
        
        if hard:
            # Straight-through estimator
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft
    
    def forward(
        self,
        cam1_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        state: torch.Tensor,
        train: bool = True,
    ) -> torch.Tensor:
        """
        Compute routing weights for auxiliary cameras.
        
        Args:
            cam1_tokens: Base camera features [batch, seq_len, embed_dim]
            prompt_tokens: Prompt embeddings [batch, prompt_len, embed_dim]
            state: Robot state [batch, state_dim]
            train: Training mode flag
        
        Returns:
            routing_weights: [batch, num_experts] where [:, 0]=cam2, [:, 1]=cam3
        """
        # 1. Pool cam1 tokens (mean pooling)
        cam1_pooled = cam1_tokens.mean(dim=1)  # [batch, embed_dim]
        
        # 2. Pool prompt tokens (attention pooling or mean pooling)
        if self.use_attention_pooling:
            prompt_pooled = self._attention_pool(prompt_tokens)  # [batch, embed_dim]
        else:
            prompt_pooled = prompt_tokens.mean(dim=1)  # [batch, embed_dim]
        
        # 3. Concatenate all features
        combined_features = torch.cat([cam1_pooled, prompt_pooled, state], dim=-1)
        # Shape: [batch, embed_dim + embed_dim + state_dim]
        
        # 4. Router MLP
        x = F.relu(self.router_linear1(combined_features))
        x = F.relu(self.router_linear2(x))
        logits = self.router_linear3(x)  # [batch, num_experts]
        
        # 5. Temperature scaling
        logits = logits / self.temperature
        
        # 6. Compute routing weights
        if train and self.use_gumbel:
            routing_weights = self._gumbel_softmax(logits, temperature=self.gumbel_temp)
        else:
            routing_weights = F.softmax(logits, dim=-1)
        
        return routing_weights
    
    def compute_routing_loss(
        self,
        routing_weights: torch.Tensor,
        target_camera_idx: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Compute Focal Loss for routing.
        
        Focal Loss = -α * (1 - p_t)^γ * log(p_t)
        
        Args:
            routing_weights: [batch, num_experts] predicted routing weights
            target_camera_idx: [batch] ground truth camera indices (0=cam2, 1=cam3)
            alpha: Positive/negative example balance weight
            gamma: Focusing parameter (higher = more focus on hard examples)
        
        Returns:
            loss: Scalar routing loss
        """
        # Gather target camera probabilities
        p_t = torch.gather(routing_weights, 1, target_camera_idx.unsqueeze(1))
        p_t = torch.clamp(p_t, 1e-10, 1.0 - 1e-10)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - p_t, gamma)
        
        # Focal Loss
        loss = -alpha * focal_weight * torch.log(p_t)
        return loss.mean()


class CameraMoE(nn.Module):
    """
    Camera Mixture of Experts using soft gating with learnable scaling.
    
    Architecture:
    - cam1 (base): Always included, no gating
    - cam2, cam3 (wrist): Gated by router weights
    - Soft gating: F_gated = γ ⊙ (w ⊙ F_original)
    """
    
    def __init__(self, config: CameraRouterConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.use_learnable_scales = config.use_learnable_scales
        
        # Initialize router
        self.router = CameraRouter(config)
        
        # Learnable scaling parameters for gated cameras
        if self.use_learnable_scales:
            self.scale_cam2 = nn.Parameter(torch.ones(1, 1, self.embed_dim))
            self.scale_cam3 = nn.Parameter(torch.ones(1, 1, self.embed_dim))
        
        # Projection layer to fuse concatenated features back to embed_dim
        # Input: 3 * embed_dim (cam1 + cam2_gated + cam3_gated)
        # Output: embed_dim
        self.projection = nn.Linear(3 * self.embed_dim, self.embed_dim)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)
    
    def forward(
        self,
        cam1_tokens: torch.Tensor,
        cam2_tokens: Optional[torch.Tensor],
        cam3_tokens: Optional[torch.Tensor],
        prompt_tokens: torch.Tensor,
        state: torch.Tensor,
        train: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multi-camera features using router-based soft gating.
        
        Args:
            cam1_tokens: Base camera (L515) [batch, seq_len, embed_dim]
            cam2_tokens: Left Wrist camera [batch, seq_len, embed_dim] or None
            cam3_tokens: Right Wrist camera [batch, seq_len, embed_dim] or None
            prompt_tokens: Prompt embeddings [batch, prompt_len, embed_dim]
            state: Robot state [batch, state_dim]
            train: Training mode flag
        
        Returns:
            output_tokens: Fused features [batch, seq_len, embed_dim]
            routing_weights: Camera gating weights [batch, num_experts]
        """
        batch_size, seq_len, _ = cam1_tokens.shape
        
        # 1. Compute routing weights using cam1, prompt, and state
        routing_weights = self.router(
            cam1_tokens=cam1_tokens,
            prompt_tokens=prompt_tokens,
            state=state,
            train=train,
        )  # [batch, 2] where [:, 0]=cam2, [:, 1]=cam3
        
        # 2. Handle missing cameras (zero padding)
        if cam2_tokens is None:
            cam2_tokens = torch.zeros_like(cam1_tokens)
        if cam3_tokens is None:
            cam3_tokens = torch.zeros_like(cam1_tokens)
        
        # 3. Apply soft gating to auxiliary cameras
        # Extract routing weights for each camera
        w_cam2 = routing_weights[:, 0:1].unsqueeze(1)  # [batch, 1, 1]
        w_cam3 = routing_weights[:, 1:2].unsqueeze(1)  # [batch, 1, 1]
        
        # Apply gating
        if self.use_learnable_scales:
            # Two-step gating: w ⊙ F, then γ ⊙ (w ⊙ F)
            cam2_weighted = cam2_tokens * w_cam2  # [batch, seq_len, embed_dim]
            cam3_weighted = cam3_tokens * w_cam3
            
            cam2_gated = cam2_weighted * self.scale_cam2  # Learnable magnitude compensation
            cam3_gated = cam3_weighted * self.scale_cam3
        else:
            # Simple gating (for backward compatibility)
            cam2_gated = cam2_tokens * w_cam2
            cam3_gated = cam3_tokens * w_cam3
        
        # 4. Concatenate all camera features
        concatenated = torch.cat([
            cam1_tokens,  # Base camera (always included)
            cam2_gated,   # Left Wrist (gated)
            cam3_gated,   # Right Wrist (gated)
        ], dim=-1)  # [batch, seq_len, 3 * embed_dim]
        
        # 5. Project back to original dimension
        output_tokens = self.projection(concatenated)  # [batch, seq_len, embed_dim]
        
        return output_tokens, routing_weights
    
    def compute_routing_loss(
        self,
        routing_weights: torch.Tensor,
        cam2_activate: torch.Tensor,
        cam3_activate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute routing loss from ground truth camera labels.
        
        Args:
            routing_weights: [batch, 2] predicted routing weights
            cam2_activate: [batch] binary labels (1=cam2 should be active)
            cam3_activate: [batch] binary labels (1=cam3 should be active)
        
        Returns:
            loss: Scalar routing loss
        """
        # Convert binary labels to target camera index
        # Priority: cam2 > cam3 > default cam2
        target_camera_idx = torch.where(
            cam2_activate == 1,
            torch.zeros_like(cam2_activate),  # 0 = cam2
            torch.where(
                cam3_activate == 1,
                torch.ones_like(cam3_activate),  # 1 = cam3
                torch.zeros_like(cam2_activate)  # default = cam2
            )
        )
        
        return self.router.compute_routing_loss(routing_weights, target_camera_idx)
