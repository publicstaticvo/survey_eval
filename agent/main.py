from tools import ToolConfig, PDFParser
from agent import build_agent
from config import Config
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="agent.yaml")
    args = parser.parse_args()

    # prepare config and paper
    config = Config.from_yaml(args.config)
    paper = PDFParser().parse_pdf(config.paper_path).get_skeleton()
    tool_config = ToolConfig.from_yaml(config.tool_config)
    # Initialize the agent
    agent = build_agent(tool_config)
    result_state = agent.invoke({"query": config.query, "review_paper": paper})
    # Post process

if __name__ == "__main__":
    main()