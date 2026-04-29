import asyncio
import argparse

try:
    from .tools.utility.grobidpdf.paper_parser import PaperParser
    from .tools.utility.request_utils import SessionManager
    from .tools.utility.tool_config import ToolConfig
    from .agent import build_agent
    from .config import Config
except ImportError:
    from tools.utility.grobidpdf.paper_parser import PaperParser
    from tools.utility.request_utils import SessionManager
    from tools.utility.tool_config import ToolConfig
    from agent import build_agent
    from config import Config


def load_paper(path: str):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return PaperParser().parse(content).get_skeleton()

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
    paper = load_paper(config.paper_path)
    tool_config = ToolConfig.from_yaml(config.tool_config)
    agent = build_agent(tool_config)

    async def _run():
        await SessionManager.init()
        try:
            return await agent.evaluate(config.query, paper)
        finally:
            await SessionManager.close()

    result = asyncio.run(_run())
    print(result)

if __name__ == "__main__":
    parse_test_papers()
