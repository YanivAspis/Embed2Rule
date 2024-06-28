import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from lavis.models import load_model_and_preprocess

class WeakLabeller:
    def __init__(self, config, device, metadata):
        self._config = config
        self._device = device
        self._labeller_model, self._visual_preprocessor, self._text_preprocessor = (
            load_model_and_preprocess(name=config["weak_labelling"]["model"],
                                      model_type=config["weak_labelling"]["model_type"],
                                      is_eval=True,
                                      device=device)
        )
        self._metadata = metadata
        self._setup_texts()
        self._img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config["weak_labelling"]["image_size"], interpolation=InterpolationMode.BICUBIC, antialias=True) if config["weak_labelling"]["resize"]
            else transforms.CenterCrop(config["weak_labelling"]["image_size"]),
            self._convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(config["weak_labelling"]["mean"], config["weak_labelling"]["std"])
        ])

    def _setup_texts(self):
        self._texts = dict()
        for latent_concept_name, latent_concept in self._metadata.latent_concepts.items():
            if latent_concept_name not in self._texts:
                if "prompt" in self._config["weak_labelling"]:
                    self._texts[latent_concept_name] = [self._text_preprocessor["eval"](self._config["weak_labelling"]["prompt"] + f" {value}") for value in latent_concept.values]
                else:
                    self._texts[latent_concept_name] = [self._text_preprocessor["eval"](f"{value}") for value in latent_concept.values]

    def _convert_to_rgb(self, img):
        return img.convert("RGB")

    @torch.no_grad()
    def _preprocess_images(self, imgs_tensor):
        imgs_tensor = [self._img_transforms(img) for img in imgs_tensor]
        imgs_tensor = torch.stack(imgs_tensor)
        return imgs_tensor

    @torch.no_grad()
    def classify_images(self, images, latent_concept_name):
        images = self._preprocess_images(images).to(self._device)
        probs = list()
        for text in self._texts[latent_concept_name]:
            texts = [text] * len(images)
            itm_output = self._labeller_model({
                "image": images,
                "text_input": texts
            }, match_head="itm")
            probs.append(torch.nn.functional.softmax(itm_output, dim=1)[:, 1])
        probs = torch.stack(probs, dim=1)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs