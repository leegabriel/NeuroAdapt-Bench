from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, *args, **kwargs):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(self.model, self.optimizer, *args, **kwargs)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def binary_entropy(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.sigmoid(x)
    ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return ent   # shape [B]


def extract_logits(outputs):
    """Extract logits from either a tensor output or a dict-style model output."""
    if isinstance(outputs, dict):
        if "logit" not in outputs:
            raise ValueError("Dict model outputs must contain a 'logit' tensor for Tent")
        logits = outputs["logit"]
    elif torch.is_tensor(outputs):
        logits = outputs
    else:
        raise ValueError(f"Unsupported model output type for Tent: {type(outputs)}")

    if logits.ndim == 2 and logits.size(1) == 1:
        logits = logits.squeeze(1)
    return logits


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(model, optimizer, *args, **kwargs):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(*args, **kwargs)
    logits = extract_logits(outputs)
    # adapt
    if logits.ndim == 2:
        loss = softmax_entropy(logits).mean(0)
    elif logits.ndim == 1:
        loss = binary_entropy(logits).mean()
    else:
        raise ValueError(f"Unsupported logit shape: {logits.shape}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    # Check the frozen parameters
    # for name, param in model.named_parameters():
    #     print(f"Requires_grad: {param.requires_grad}, Parameter: {name}")
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()])
    has_ln = any([isinstance(m, nn.LayerNorm) for m in model.modules()])
    has_gn = any([isinstance(m, nn.GroupNorm) for m in model.modules()])
    assert has_bn or has_ln or has_gn, "tent needs normalization for its optimization"
