from transformers import pipeline
from PIL import Image

import gradio as gr

title="dpt"


pipe = pipeline(task="depth-estimation", model="AI models\\dpt")

def get_depth(image):
    image=Image.fromarray(image)
    return pipe(image)["depth"]

demo = gr.Interface(
    get_depth,
    inputs=[gr.Image()],
    outputs=[gr.Image()],
    title=title,
)

if __name__ == "__main__":
    demo.launch()