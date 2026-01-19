from tools import ToolConfig, PaperParser
from agent import build_agent
from config import Config
import argparse


def parse_test_papers():
    parser = PaperParser()
    import glob, json
    for f in glob.glob("pdf/*.xml"):
        with open(f) as p: xml = p.read()
        paper = parser.parse(xml).get_skeleton()
        with open(f"{f[:-4]}.json", "w") as p: json.dump(paper, p, indent=2, ensure_ascii=False)
        print(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="agent.yaml")
    args = parser.parse_args()

    # prepare config and paper
    config = Config.from_yaml(args.config)
    paper = PaperParser().parse(config.paper_path).get_skeleton()
    tool_config = ToolConfig.from_yaml(config.tool_config)
    # Initialize the agent
    build_agent(tool_config).invoke({"query": config.query, "review_paper": paper})
    # Post process

if __name__ == "__main__":
    parse_test_papers()