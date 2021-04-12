import ptlearn

import numpy as np

from typing import Tuple


def get_xy(n) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.random([n, 2]) * 2.0 - 1.0
    y = (np.prod(x, axis=1, keepdims=True) > 0).astype(np.int64)
    return x, y


data_base = ptlearn.data_dict["ml"]
loader_base = ptlearn.loader_dict["ml"]

train_xy, valid_xy = map(get_xy, [10000, 100])
train_loader = loader_base(data_base(*train_xy), True)
valid_loader = loader_base(data_base(*valid_xy), False)

loss = ptlearn.loss_dict["cross_entropy"]()
fcnn = ptlearn.FCNN(2, 2, [16, 16])
inference = ptlearn.MLInference(fcnn)
trainer = ptlearn.Trainer(num_epoch=40)
trainer.fit(loss, fcnn, inference, train_loader, valid_loader, cuda="0")

test_x = np.array(
    [
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
    ],
    np.float32,
)
test_loader = loader_base(data_base(test_x, None), False)
outputs = inference.get_outputs(trainer.device, test_loader)
print(outputs.forward_results["predictions"].argmax(1))
