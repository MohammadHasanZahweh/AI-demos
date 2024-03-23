import gradio as gr
from utils.KOSMOS_utils import run_example,draw_entity_boxes_on_image
from PIL import Image
title = "KOSMOS"
description = "Kosmos is a VLM with multiple functionalities such as grounding, image question answering and such"

def KOMOSGounding(image):
    entities, processed_text, _processed_text = run_example("<grounding>An image of",image,show=True)
    image=Image.fromarray(image)
    image = draw_entity_boxes_on_image(image,entities)
    return image ,processed_text

demo = gr.Interface(
    fn=KOMOSGounding,
    inputs=[gr.Image()],
    outputs=[gr.Image(),gr.Text()],
    title=title,
    description=description
)

if __name__=="__main__":
    demo.launch()