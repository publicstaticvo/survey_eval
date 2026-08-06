import asyncio
import logging
from agent.tools.preprocess.multilabel_extract import LabelParagraphExtractor
from agent.tools.utility.content_walk import split_content_to_paragraph
from agent.tools.utility.latex_parser import LatexPaperParser
from agent.tools.utility.tool_config import ToolConfig
from agent.tools.utility.request_utils import SessionManager

async def main():
    logging.basicConfig(level=logging.INFO)
    p=LatexPaperParser().parse(r'agent/test_inputs/sgen_surveys/dialogue_systems/final_survey_refined.tex')
    paragraph=split_content_to_paragraph(p,include_abstract=True)[0]
    items=[]
    for i,s in enumerate(paragraph):
        items.append({'item_id':f'p0-s{i}','text':s.text,'environment_type':s.environment_type})
    rendered='\n'.join(f"[{x['item_id']}|{x['environment_type']}] {x['text']}" for x in items)
    await SessionManager.init()
    try:
        client=LabelParagraphExtractor(ToolConfig(), 'CONTRIBUTION')
        result=await asyncio.wait_for(client.call(inputs={'paragraph':rendered,'items':items}), timeout=240)
        print(result)
    finally:
        await SessionManager.close()
asyncio.run(main())