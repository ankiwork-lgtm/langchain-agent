from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: dict[str, any],
        prompts: list[str],
        **kwargs: any,
        # return super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs
    ) -> any:

        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("********")

    def on_llm_end(self, response: LLMResult, **kwargs: any) -> any:
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("********")

    # return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
