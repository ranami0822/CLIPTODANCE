from adapters import Estimator, Judge
from adapters.cloudflare import CloudflareEstimator, CloudflareJudge


def test_cloudflare_adapters_satisfy_interfaces():
    assert isinstance(CloudflareJudge(), Judge)
    assert isinstance(CloudflareEstimator(), Estimator)
