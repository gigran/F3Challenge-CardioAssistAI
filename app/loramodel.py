from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


# Adaptador LoRA versionado junto do código, em data/loramodel.
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "loramodel"


class LoraChatModel(BaseChatModel):
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    max_new_tokens: int = 512

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._model = model
        self._tokenizer = tokenizer

    @property
    def _llm_type(self) -> str:
        return "llama3-lora-unsloth"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        import torch

        system_content = "" 
        human_content = ""

        for message in messages: 
            if isinstance(message, SystemMessage): 
                system_content = str(message.content) 
            elif isinstance(message, HumanMessage): 
                human_content = str(message.content)

        instruction = "" 
        question = "" 
        context = ""

        if "Instrução:" in human_content: 
            instruction = human_content.split( "Instrução:", 1 )[1]

        if "Pergunta:" in instruction: 
            instruction, question_part = instruction.split( "Pergunta:", 1 ) 
            question = question_part

        if "Contexto recuperado:" in question: 
            question, context = question.split( "Contexto recuperado:", 1 )    

        instruction = instruction.strip() 
        question = question.strip() 
        context = context.strip()  

        prompt = f"""{system_content}

                    ### Instrução: 
                    {instruction}

                    ### Pergunta:
                    {question}

                    ### Context recuperado:
                    {context}                    

                    ### Resposta:
                    """

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Remove os tokens correspondentes ao prompt.
        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[0][input_length:]

        response = self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=response)
                )
            ]
        )

@lru_cache(maxsize=1)
def load_lora_model():
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    from peft import PeftModel

    model = PeftModel.from_pretrained(
        model,
        str(MODEL_PATH),
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer