from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm
from peft import get_peft_model, PeftModel, LoraConfig
from transformers import get_scheduler, PreTrainedModel, PreTrainedTokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import SchedulerType
from transformers.trainer_pt_utils import get_parameter_names

_TEMP_LORA_NAME = "__temp_lora__"


def _create_lr_scheduler(
    num_training_steps: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler_type: Union[str, SchedulerType],
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    return lr_scheduler


def _get_decay_parameter_names(model: PreTrainedModel) -> List[str]:
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def _create_optimizer(
    model: PreTrainedModel,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay_parameters = _get_decay_parameter_names(model=model)
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0
        }
    ]
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-7
    )


class _Dataset(Dataset):
    def __init__(self, input_ids, prefix_length, chunk_size):
        self.dataset = []
        self.input_ids = input_ids

        init_window_size = 1024
        input_length = input_ids.size(-1)

        self.dataset.append(
            {
                "start": 0,
                "middle": 0,
                "end": min(input_length, init_window_size)
            }
        )
        if input_length >= init_window_size:
            for i in range(init_window_size, input_length, chunk_size):
                self.dataset.append(
                    {
                        "start": max(0, i - prefix_length),
                        "middle": i,
                        "end": min(i + chunk_size, input_length)
                    }
                )
                if i + chunk_size >= input_length:
                    break

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ins = self.dataset[idx]
        start, middle, end = ins["start"], ins["middle"], ins["end"]

        if middle == start:
            input_ids = self.input_ids[:, start: end]
            attention_mask = torch.ones_like(input_ids)
            return {
                "input_ids": input_ids,
                "labels": input_ids,
                "attention_mask": attention_mask,
            }
        context_input_ids = self.input_ids[:, start: middle]
        context_labels = torch.zeros_like(input=context_input_ids) - 100
        inference_input_ids = self.input_ids[:, middle: end]

        window_input_ids = torch.cat(tensors=[context_input_ids, inference_input_ids], dim=-1)
        return {
            "input_ids": window_input_ids,
            "attention_mask": torch.ones_like(window_input_ids),
            "labels": torch.cat(tensors=[context_labels, inference_input_ids], dim=-1),
        }


class _SimpleDataset(Dataset):
    def __init__(self, input_ids, prefix_length, chunk_size):
        context_input_ids = input_ids[:, :-chunk_size]
        context_labels = torch.zeros_like(input=context_input_ids) - 100
        inference_input_ids = input_ids[:, -chunk_size:]
        self.dataset = [
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": torch.cat(tensors=[context_labels, inference_input_ids], dim=-1)
            }
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]


@dataclass
class TempLoraTuningConfig:
    learning_rate: float = field(default=5e-5)
    num_epoch: int = field(default=1)
    weight_decay: float = field(default=0.0)
    lr_scheduler_type: Union[str, SchedulerType] = field(default=SchedulerType.CONSTANT_WITH_WARMUP)
    warmup_steps: int = field(default=2)


class TempLoraCompletionPipeline(TextGenerationPipeline):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        lora_config: Optional[LoraConfig] = None,
        lora_model_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        prefix_length: int = 1024,
        stride_length: int = 1024,
        lora_training_config: TempLoraTuningConfig = TempLoraTuningConfig(),
        **kwargs
    ):
        if not isinstance(model, PreTrainedModel):
            raise TypeError("Only support pytorch backend.")
        if not lora_config and not lora_model_path:
            raise ValueError("At least one of lora_config or lora_model_path should be specified.")
        if prefix_length + stride_length > model.config.max_position_embeddings:
            raise ValueError("sum of prefix_length and stride_length must smaller than model max_position_embeddings.")

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            framework="pt",
            batch_size=1,
            **kwargs
        )

        self.tokenizer.padding_side = "right"

        if gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
        self.model.enable_input_require_grads()

        self.initial_lora_model_path = lora_model_path
        self.lora_config = lora_config
        self.prefix_length = prefix_length
        self.stride_length = stride_length
        self.lora_training_config = lora_training_config

    def _create_temp_lora(self):
        if isinstance(self.model, PeftModel):
            return

        if not self.initial_lora_model_path:
            self.model = get_peft_model(self.model, self.lora_config, _TEMP_LORA_NAME)
        else:
            self.model = PeftModel.from_pretrained(
                self.model,
                self.initial_lora_model_path,
                _TEMP_LORA_NAME,
                is_trainable=True
            )

    def _update_temp_lora(self, input_ids, dataset_cls, lora_training_config: Optional[TempLoraTuningConfig] = None):
        self.model.train()

        if not lora_training_config:
            lora_training_config = self.lora_training_config

        dataset = dataset_cls(input_ids, prefix_length=self.prefix_length, chunk_size=self.stride_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            sampler=SequentialSampler(data_source=dataset),
            num_workers=0,
            collate_fn=lambda i: i[0],
        )
        optimizer = _create_optimizer(
            self.model,
            lora_training_config.learning_rate,
            lora_training_config.weight_decay
        )
        lr_scheduler = _create_lr_scheduler(
            num_training_steps=len(dataset) * lora_training_config.num_epoch,
            optimizer=optimizer,
            lr_scheduler_type=lora_training_config.lr_scheduler_type,
            warmup_steps=lora_training_config.warmup_steps
        )

        with torch.cuda.amp.autocast():
            for epoch in range(lora_training_config.num_epoch):
                total_loss = 0
                progress_bar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{lora_training_config.num_epoch}")
                for step, batch in enumerate(progress_bar):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                    optimizer.zero_grad()

                    if step % 10 == 0:
                        progress_bar.set_postfix(loss=(total_loss / 10).item())
                        total_loss = 0

        self.model.eval()

    def _delete_temp_lora(self):
        if not isinstance(self.model, PeftModel):
            return

        old_model = self.model
        self.model = self.model.get_base_model()
        del old_model

    def forward(self, model_inputs, **forward_params):
        model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
        model_outputs = self._forward(model_inputs, **forward_params)
        model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))

        return model_outputs

    def _inference_with_lora_update(self, model_inputs, origin_input_ids, **generate_kwargs):
        self.model: PreTrainedModel
        self.model.eval()

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        min_tokens_to_generate = generate_kwargs["min_new_tokens"]
        max_tokens_to_generate = max(generate_kwargs["min_new_tokens"], generate_kwargs["max_new_tokens"])
        model_max_length = self.model.config.max_position_embeddings
        input_length = input_ids.shape[-1]

        inference_context = self.get_inference_context()

        if max_tokens_to_generate + input_length <= model_max_length:
            with self.device_placement(), inference_context():
                generated_sequence = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generate_kwargs
                )
            out_b = generated_sequence.shape[0]
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
            return {"generated_sequence": generated_sequence, "input_ids": origin_input_ids, "prompt_text": prompt_text}

        input_ids = input_ids[:, -self.prefix_length:]
        output_ids = None
        while max_tokens_to_generate > 0:
            generate_kwargs["min_new_tokens"] = min(self.stride_length, min_tokens_to_generate)
            generate_kwargs["max_new_tokens"] = min(self.stride_length, max_tokens_to_generate)

            with self.device_placement(), inference_context():
                intermediate_output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=None,
                    **generate_kwargs
                )
            if output_ids is None:
                output_ids = intermediate_output_ids
            else:
                output_ids = torch.cat((output_ids, intermediate_output_ids), dim=1)
            if intermediate_output_ids[0, -1] == self.tokenizer.eos_token_id:
                break
            self._update_temp_lora(torch.cat((input_ids, intermediate_output_ids), dim=1), _SimpleDataset)
            # TODO: using kv cache
            input_ids = torch.cat((input_ids, intermediate_output_ids), dim=1)[:, -self.prefix_length:]

            output_length = intermediate_output_ids.shape[-1]
            min_tokens_to_generate -= output_length
            max_tokens_to_generate -= output_length
        return {"generated_sequence": output_ids, "input_ids": origin_input_ids, "prompt_text": prompt_text}

    def _forward(self, model_inputs, **generate_kwargs):
        if "max_new_tokens" not in generate_kwargs:
            raise ValueError("must specify max_new_tokens")
        if "min_new_tokens" not in generate_kwargs:
            raise ValueError("must specify min_new_tokens")
        if generate_kwargs.get("num_return_sequences", 1) != 1:
            raise ValueError("num_return_sequences can only be 1")

        self.model: PreTrainedModel

        if self.initial_lora_model_path:
            self._create_temp_lora()

        input_ids = model_inputs["input_ids"]
        input_length = input_ids.shape[-1]
        model_max_length = self.model.config.max_position_embeddings
        if input_length >= model_max_length:  # encode prompt to lora weights before generate
            self._create_temp_lora()
            self._update_temp_lora(input_ids[:, :-self.prefix_length], _Dataset)
            model_inputs["input_ids"] = input_ids[:, -self.prefix_length:]
            if model_inputs["attention_mask"] is not None:
                model_inputs["attention_mask"] = model_inputs["attention_mask"][:, -self.prefix_length:]

        outputs = self._inference_with_lora_update(model_inputs, input_ids, **generate_kwargs)

        self._delete_temp_lora()
        return outputs

    def preprocess(
        self,
        prompt_text,
        prefix="",
        handle_long_generation=None,
        add_special_tokens=False,
        truncation=None,
        padding=False,
        max_length=None,
        **generate_kwargs,
    ):
        inputs = self.tokenizer(
            prefix + prompt_text,
            return_tensors=self.framework,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        inputs["prompt_text"] = prompt_text

        return inputs

    def train_initial_lora(
        self,
        text: str,
        save_directory: str,
        safe_serialization: bool = True,
        save_embedding_layers: str | bool = "auto",
        lora_training_config: Optional[TempLoraTuningConfig] = None,
    ):
        model_inputs = self.preprocess(text)
        self._delete_temp_lora()
        self._create_temp_lora()
        self._update_temp_lora(model_inputs["input_ids"], _Dataset, lora_training_config)
        self.model.save_pretrained(
            save_directory=save_directory,
            safe_serialization=safe_serialization,
            save_embedding_layers=save_embedding_layers
        )
        self._delete_temp_lora()


__all__ = [
    "TempLoraTuningConfig",
    "TempLoraCompletionPipeline"
]
