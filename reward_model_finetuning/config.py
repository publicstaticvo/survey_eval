from ruamel.yaml import YAML
from dataclasses import dataclass, field


@dataclass
class P2QConfig:
    model_name_or_path: str
    model_save_path: str
    data_path: list[str]
    data_split_test_sets: float
    data_min_score_diff: float
    train_epochs: int
    gradient_accumulation: int
    gradient_checkpointing: bool
    learning_rate: float
    max_seq_len: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    seed: int
    warmup_type: str
    warmup_ratio: float
    weight_decay: float
    deepspeed_config: dict = field(default_factory=dict)
    local_rank: int = -1
    
    @classmethod
    def init_from_yaml(cls, config_path: str):
        loader = YAML(typ='safe')
        with open(config_path) as f: config = loader.load(f)
        return cls(
            model_name_or_path=config['model']['name_or_path'],
            model_save_path=config['model']['save_path'],
            data_path=config['data']['path'],
            data_split_test_sets=config['data']['split_test_sets'],
            data_min_score_diff=config['data']['min_score_diff'],
            train_epochs=config['train']['epochs'],
            gradient_accumulation=config['train']['gradient_accumulation'],
            gradient_checkpointing=config['train']['gradient_checkpointing'],
            learning_rate=config['train']['learning_rate'],
            max_seq_len=config['train']['max_seq_len'],
            per_device_train_batch_size=config['train']['per_device_train_batch_size'],
            per_device_eval_batch_size=config['train']['per_device_eval_batch_size'],
            seed=config['train']['seed'],
            warmup_type=config['train']['warmup_type'],
            warmup_ratio=config['train']['warmup_ratio'],
            weight_decay=config['train']['weight_decay'],
            deepspeed_config = config['deepspeed']
        )
