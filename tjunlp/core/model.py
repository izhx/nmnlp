from typing import Dict, Union, List, Any

import torch


class Model(torch.nn.Module):
    """
    Abstract model class with more function.
    """

    def __init__(self, criterion=None):
        super().__init__()
        self.criterion = criterion
        self.evaluating: bool = False
        self.freeze_modules: List[torch.nn.Module] = list()

    def forward(self, *inputs) -> Dict[str, Any]:
        """
        Defines the forward pass of the model. In addition, to facilitate easy training,
        this method is designed to compute a loss function defined by a user.

        The input is comprised of everything required to perform a
        training update, `including` labels - you define the signature here!
        It is down to the user to ensure that inference can be performed
        without the presence of these labels. Hence, any inputs not available at
        inference time should only be used inside a conditional block.

        The intended sketch of this method is as follows::

            def forward(self, input1, input2, targets=None):
                ....
                ....
                output1 = self.layer1(input1)
                output2 = self.layer2(input2)
                output_dict = {"output1": output1, "output2": output2}
                if targets is not None:
                    # Function returning a scalar torch.Tensor, defined by the user.
                    loss = self._compute_loss(output1, output2, targets)
                    output_dict["loss"] = loss
                return output_dict

        Parameters
        ----------
        inputs:
            Tensors comprising everything needed to perform a training update, `including` labels,
            which should be optional (i.e have a default value of ``None``).  At inference time,
            simply pass the relevant inputs, not including the labels.

        Returns
        -------
        output_dict: ``Dict[str, torch.Tensor]``
            The outputs from the model. In order to train a model using the
            :class:`~tju.core.Trainer` api, you must provide a "loss" key pointing to a
            scalar ``torch.Tensor`` representing the loss to be optimized.
        """
        raise NotImplementedError

    def get_metrics(self, reset=False, **kwargs) -> Dict[str, float]:
        raise NotImplementedError

    @staticmethod
    def is_best(metric: Dict[str, float], former: Dict[str, float]) -> bool:
        raise NotImplementedError

    def train_mode(self, device):
        """
        Keep specific modules at eval model.
        """
        self.to(device)
        self.evaluating = False
        self.train()
        for module in self.freeze_modules:
            module.eval()
        return self

    def eval_mode(self, device):
        self.to(device)
        self.evaluating = True
        return self.eval()

    def test_mode(self, device):
        self.to(device)
        self.evaluating = False
        return self.eval()

    @classmethod
    def from_archive(self, path):
        pass  # TODO(izhx)
