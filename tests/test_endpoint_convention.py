import inspect

import dmft_l2_nonlinear


def test_measured_strict_past_endpoint_is_the_default():
    parameter = inspect.signature(dmft_l2_nonlinear.solve).parameters["onsager"]
    assert parameter.default == 0.0
