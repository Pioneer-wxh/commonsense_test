from transformers import Trainer, TrainingArguments
import torch

__all__ = ["LORATrainer", "LORATrainingArguments"]

class LORATrainingArguments(TrainingArguments):
    def __init__(self, gamma=4e-4, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

class LORATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 首先调用父类的 compute_loss 获取原始任务损失
        task_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        # 计算正交约束损失
        ortho_loss = self.calc_ortho(model)
        
        if ortho_loss is not None:
            # 使用gamma参数作为正交损失的权重
            total_loss = task_loss + self.args.gamma * ortho_loss
        else:
            total_loss = task_loss
            
        return (total_loss, outputs) if return_outputs else total_loss

    @staticmethod
    def calc_ortho(model):
        """
        Calculate the average orthogonal regularizer loss for LORA matrices in the model.

        Args:
            model: The PyTorch model containing LORA layers.

        Returns:
            float or None: The average orthogonality loss, or None if no LORA matrices are found.
        """
        ortho_loss = 0.0
        den = 0
        for name, param in model.named_parameters():
            if "LORA_A" in name:
                a = param
                ia = torch.eye(a.shape[0], device=a.device)
                ia.requires_grad = False
                a = a @ a.T - ia
                ortho_loss += torch.norm(a, p="fro")
                den += 1
            elif "LORA_B" in name:
                b = param
                ib = torch.eye(b.shape[1], device=b.device)
                ib.requires_grad = False
                b = b.T @ b - ib
                ortho_loss += torch.norm(b, p="fro")
                den += 1
        if den != 0:
            return ortho_loss / den
        else:
            return None
