from ..core.llm import LeoLLMClient

class Agent:
    def __init__(self, 
                 name: str,
                 llm: LeoLLMClient,
                 system_prompt: str):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        
    def __repr__(self) -> str:
        return self.__str__()
