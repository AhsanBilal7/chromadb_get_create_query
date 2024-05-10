import clip
import torch
from numpy import ndarray
from typing import List
from PIL import Image 
from chromadb import EmbeddingFunction, Documents, Embeddings

class ClipEmbeddingsfunction(EmbeddingFunction):
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):  
        """
            Initialize the ClipEmbeddingsfunction.

            Args:
                model_name (str, optional): The name of the CLIP model to use. Defaults to "ViT-B/32".
                device (str, optional): The device to use for inference (e.g., "cpu" or "cuda"). Defaults to "cpu".
        """

        self.device = device
        self.model, self.preprocess = clip.load(model_name, self.device)

    def __call__(self, input: Documents)-> Embeddings:
        """
            Compute embeddings for a batch of images.

            Args:
                input (Documents): A list of image file paths.

            Returns:
                Embeddings: A list of image embeddings.
        """
            
        list_of_embeddings = []
        for image_path in input:
            image = Image.open(image_path)
            image = image.resize((224, 224))
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_image(image_input).cpu().detach().numpy()

            list_of_embeddings.append([float(value) for value in embeddings[0]]) 

        return list_of_embeddings
    
    def embed_image(self, input: str)-> ndarray:  
        """
            Compute embeddings for a single image.

            Args:
                input (str): The file path of the image or "cropped_image.png".

            Returns:
                ndarray: The image embedding.
        """

        if (input!="cropped_image.png"): 
            input = input.resize((224, 224)) 
            input = self.preprocess(input).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_image(input).cpu().detach().numpy()

            return embeddings[0]
        else:
            input = Image.open(input)
            try:
                input = input.resize((224, 224))
            except:
                print("can't resize image!")

            input = self.preprocess(input).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_image(input).cpu().detach().numpy()

            return embeddings[0]
    
    def get_text_embeddings(self, text: str) -> List[ndarray]:
        """
            Compute embeddings for a text.

            Args:
                text (str): The input text.

            Returns:
                List[ndarray]: A list containing the text embedding.
        """
                
        text_token = clip.tokenize(text)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_token).cpu().detach().numpy()
        return list(text_embeddings[0])