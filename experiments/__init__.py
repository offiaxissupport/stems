"""Experiment harness for the clean evaluation phase.

``scenario``    which buildings, which weather period, which constraint caps
``controllers`` the ablation arms (policy x safety layer)
``runner``      one (scenario, arm, seed): train, evaluate, check actuators
``ablation``    the resumable, parallel grid
``aggregate``   means with 95% confidence intervals and paired differences
"""
