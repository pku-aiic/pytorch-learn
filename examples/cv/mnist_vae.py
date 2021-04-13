import os
import ptlearn

from torch import Tensor
from typing import Tuple
from ptlearn.types import tensor_dict_type
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from ptlearn.misc.toolkit import to_device
from ptlearn.misc.toolkit import save_images
from ptlearn.misc.toolkit import eval_context


def batch_callback(batch: Tuple[Tensor, Tensor]) -> tensor_dict_type:
    img, labels = batch
    return {ptlearn.INPUT_KEY: img, ptlearn.LABEL_KEY: labels.view(-1, 1)}


class VAECallback(ptlearn.TrainerCallback):
    def log_artifacts(self, trainer: ptlearn.Trainer) -> None:
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        with eval_context(trainer.model):
            outputs = trainer.model(0, batch, trainer.state)
        original = batch[ptlearn.INPUT_KEY]
        reconstructed = outputs[ptlearn.PREDICTIONS_KEY]
        image_folder = os.path.join(trainer.workplace, "images")
        os.makedirs(image_folder, exist_ok=True)
        save_images(
            original,
            os.path.join(image_folder, f"step={trainer.state.step}.png"),
        )
        save_images(
            reconstructed,
            os.path.join(image_folder, f"step={trainer.state.step}_recon.png"),
        )


data_base = ptlearn.data_dict["dl"]
loader_base = ptlearn.loader_dict["dl"]

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ]
)

train_data = data_base(MNIST("data", transform=transform, download=True))
valid_data = data_base(MNIST("data", train=False, transform=transform, download=True))

train_pt_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # type: ignore
valid_pt_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # type: ignore

train_loader = loader_base(train_pt_loader, batch_callback)
valid_loader = loader_base(valid_pt_loader, batch_callback)

loss = ptlearn.loss_dict["vae"]({"kld_ratio": 0.1})
vae = ptlearn.VAE(28, 1)
inference = ptlearn.DLInference(vae)
pt_trainer = ptlearn.Trainer(
    valid_portion=1.0,
    callback=VAECallback(),
)
pt_trainer.fit(loss, vae, inference, train_loader, valid_loader, cuda="0")
