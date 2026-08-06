import asyncio
import logging
from agent.tools.preprocess.multilabel_extract import LabelParagraphExtractor
from agent.tools.utility.tool_config import ToolConfig
from agent.tools.utility.request_utils import SessionManager

async def main():
    logging.basicConfig(level=logging.INFO)
    await SessionManager.init()
    try:
        client=LabelParagraphExtractor(ToolConfig(), "CONTRIBUTION")
        result=await asyncio.wait_for(client.call(inputs={"paragraph":"[p0-s0|text] We present a survey of dialogue systems.","items":[{"item_id":"p0-s0","text":"We present a survey of dialogue systems.","environment_type":"text"}]}), timeout=180)
        print(result)
    finally:
        await SessionManager.close()

asyncio.run(main())