from data.generate_synthetic_data import TSFNs


def test_generate_synthetic_data():
    length = 30
    kwargs = {
        "value": 1.0,
        "amplitude": 10.0,
        "frequency": 0.1,
        "phase": 0.0,
        "slope": 1.0,
        "intercept": 1.0,
        "period": 1.0,
        "pattern": "sine",
        "drift": 0.0,
        "volatility": 0.0,
        "start_value": 3.0,
        "ar_params": [0.8, -0.2, 0.1],
        "mean": 0.,
        "noise_level": 0.1,
        "burn_in": 3,
    }
    for key, fn in TSFNs.items():
        x = fn(length, **kwargs)
        print(key)
        assert x.shape == (length,)
