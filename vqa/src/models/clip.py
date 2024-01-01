from transformers import CLIPProcessor, CLIPModel

def get_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor
