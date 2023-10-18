"""
Wrapper class definition for LLM agent
"""

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from contextlib import redirect_stdout
import io

class llmagent:
    """This class is a wrapper to create LLM agents
    """
    model_path = "./"
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    model = []
    
    sys_prompt = "You are a happy person."
    
    def __init__(self, mpath, sysp) -> None:
        self.model_path = mpath
        self.sys_prompt = sysp
        self.model = LlamaCpp(model_path=self.model_path,
                            temperature = 0.75,
                            max_tokens=4096,
                            top_p=1,
                            callback_manager=self.callback_manager,
                            streaming=True,
                            stream_prefix=True,
                            verbose=True)
    
    def prompt_creator(self, message, chat_history):
        texts = [f"<s>[INST] <<SYS>>\n{self.sys_prompt}\n<</SYS>>\n\n"]
        for user_input, response in chat_history:
            texts.append(f"[{user_input.strip()} [/INST] {response.strip()} </s><s>[INST] ")
        texts.append(f"{message.strip()} [/INST]")
        return "".join(texts)
    
    # This needs to be fixed?
    def generate(self, dialog):
        result = self.model(prompt=dialog, stream=True,)
        print("- Result --------")
        print(result)
        print("-----------------")
        outputs= []
        for part in result:
            text = part["choices"][0]["text"]
            outputs.append(text)
            yield "".join(outputs)
        
    def respond(self, message, history):
        dialog = self.prompt_creator(message, history)
             
        result = self.model.stream(dialog)
        outputs= ""
        for part in result:
            text = part["choices"][0]["text"]
            outputs = outputs + text
            yield outputs
            
            

