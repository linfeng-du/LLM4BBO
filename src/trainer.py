from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


class OfflineRLDataCollator(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        output = super().torch_call(examples)
        output['reward'] = torch.tensor([example['reward'] for example in examples])
        return output


class OfflineRLTrainer(SFTTrainer):

    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()
        self._signature_columns.append('reward')

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, CausalLMOutputWithPast]:
        _, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        logits = outputs.logits[:, :-1, :].contiguous()
        labels = inputs['labels'][:, 1:].contiguous()

        logps = torch.log_softmax(logits, dim=-1)
        token_logps = logps.gather(
            dim=-1, index=labels.unsqueeze(dim=2).clamp(min=0)
        ).squeeze(dim=2)

        token_logps.masked_fill_(labels.eq(other=-100), value=0.)
        completion_logp = token_logps.sum(dim=1)
        loss = -torch.mean(completion_logp * inputs['reward'])

        return (loss, outputs) if return_outputs else loss
