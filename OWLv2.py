import gradio as gr
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
import numpy as np
import cv2
from torchvision.ops import nms

from config import path_dict

device="cuda" if torch.cuda.is_available() else "cpu" 

processor = Owlv2Processor.from_pretrained(path_dict["OWL_V2"])
model = Owlv2ForObjectDetection.from_pretrained(path_dict["OWL_V2"]).to(device=device)

title = "OWL V2"
description = "OWL is a state-of-the-art model for zero-shot object detection. It can identify objects in images without being explicitly trained on those objects. This makes it incredibly versatile and powerful for a wide range of applications. Try it out by uploading an image, and see what objects OWL can detect!"

@torch.no_grad()
def OWL_V2(text_prompt, image, threshold, iou_threshold):
    texts=text_prompt.split(";")
    # print(image.shape)
    inputs = processor(text=texts, images=image, return_tensors="pt").to(device=device)
    outputs = model(**inputs)
    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([[image.shape[0],image.shape[1]]]).to(device=device)
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs=outputs, threshold=threshold/100, target_sizes=target_sizes)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    # text = texts[i]
    boxes, scores, labels = results[i]["boxes"].cpu().detach(), results[i]["scores"].cpu().detach(), results[i]["labels"].cpu().detach()
    image1=np.array(image)
    for l in range(len(texts)):
        # print([b for b,s,la in zip(boxes, scores, labels) if la==l])
        bees=torch.tensor([b.numpy() for b,s,la in zip(boxes, scores, labels) if la==l])
        sees=torch.tensor([s for b,s,la in zip(boxes, scores, labels) if la==l])
        if len(sees)==0:
            continue
        ret=nms(bees,sees,iou_threshold/100)
        bees=bees[ret]
        sees=sees[ret]
        for i in range(len(ret)):
            score=sees[i]
            box=bees[i]
            box=torch.clip(box,min=0)
            box=box.detach().numpy().astype(np.uint)
            image1= cv2.rectangle(np.array(image1), box[0:2], box[2:4], color=(255,0,0), thickness=2)
            image1= cv2.putText(image1, str(texts[l])+" "+str(float(score)), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36,255,12), 2)
    return image1

demo = gr.Interface(
    fn=OWL_V2,
    inputs=["text", gr.Image(),gr.Slider(value=20),gr.Slider(value=40)],
    outputs=[gr.Image()],
    title=title,
    description=description
)

if __name__=="__main__":
    demo.launch()
