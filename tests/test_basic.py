import torch
import pytest

def test_rmsnorm_shape():
    from shared_modules.components import RMSNorm
    norm = RMSNorm(dim=512)
    x = torch.randn(8, 128, 512)
    out = norm(x)
    assert out.shape == x.shape

def test_config_loading():
    from utils.config import Config
    cfg = Config({"test": {"key": "value"}})
    assert cfg.test.key == "value"
