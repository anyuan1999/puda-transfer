"""
Unified interface for cross-domain transfer methods on PIDSMaker.

All methods follow the same interface:
    1. train(source_cfg, source_data) -> trained model
    2. adapt(model, source_data, target_data) -> adapted model  
    3. evaluate(model, target_cfg, target_data) -> metrics dict
"""
