import llmagent
import gradio as gr

# To download a model, refer to https://github.com/ggerganov/llama.cpp#obtaining-and-using-the-facebook-llama-2-model
model_path = "/yourpath/to/model"
system_prompt = "You are a cynical but funny llama"

la = llmagent.llmagent(mpath=model_path, sysp=system_prompt)

llmdemo = gr.ChatInterface(fn=la.respond, title="Llama 2 7B-chat",
                            retry_btn=None, undo_btn=None, clear_btn=None)

llmdemo.queue().launch()
