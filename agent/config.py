import os
import json
import yaml
from dataclasses import dataclass
    

@dataclass(frozen=True)
class Config:
    tool_config: str
    paper_path: str
    query: str

    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path) as f: config = yaml.safe_load(f)
        return cls(
            query=config['query'],
            paper_path=config['paper_path'],
            tool_config=config['tool_config'],
        )
    
    def __str__(self):
        return self.__dict__