import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# Instantiate the model and the processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")

#Create the function
def generate_caption(image):
    # Now directly using the PIL Image object
    inputs = processor(images=image, return_tensors ='pt')
    outputs = model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def caption_image(image):
    """
    Takes a PIL Image input and returns a caption.
    """
    try:
        caption = generate_caption(image)
        return caption
    except Exception as e:
        return f"An error ocurred: {str(e)}"

interface = gr.Interface(
    fn = caption_image,
    inputs = gr.Image(type="pil"),
    outputs = "text",
    title = "Image Captioning with BLIP",
    description = "Upload an image to generate a caption."
)
interface.launch()
