import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Union

@dataclass
class Conversation:
    system: str
    conversations: list
    bos_token: str
    sep_system: str
    sep_question: str
    sep_answer: str
    place_image_token: Callable
    image_token: str = "<|image|>"
    pre_system: str = ""
    pre_question: str = ""
    pre_answers: str = ""
    eos_token: str = ""

    def get_conversation_dict_list(
        self, num_images: int = 1, num_patches: int = 144, media_type: str = "image"
    ) -> List[Dict]:
        conv_dict_list = []
        sys_text = self.pre_system + self.system + self.sep_system
        is_first = True

        if media_type == "multi_image":
            for conversation in self.conversations:
                if conversation["from"] == "human":
                    conversation["value"] = conversation["value"].replace(
                        "<image>", self.image_token * num_patches
                    )
        
        else:
            self.conversations[0]["value"] = (
                self.conversations[0]["value"]
                .replace("<image>\n", "")
                .replace("\n<image>", "")
                .replace("<image>", "")
                .replace("<video>\n", "")
                .replace("\n<video>", "")
                .replace("<video>", "")
            )

            self.conversations[0]["value"] = self.place_image_token(
                self.conversations[0]["value"],
                self.image_token,
                num_images * num_patches,
            )
        
        for conv in self.conversations:
            if is_first and conv["from"] == "assistant":
                continue
            if conv["from"] == "human":
                conv_text = ""
                if is_first:
                    conv_text += sys_text
                conv_text += self.pre_question + conv["value"] + self.sep_question
                conv_dict = {"user" : conv_text}
                is_first = False
            elif conv["from"] == "assistant":
                conv_text = self.pre_answers + conv["value"] + self.sep_answer
                conv_dict.update({"assistant": conv_text})
                conv_dict_list.append(conv_dict)
            else:
                raise ValueError(
                    f"conv['from'] must be human or assistant, but got {conv['from']}."
                    "Please fix your jsonl file."
                )
            conv_dict_list[0]["user"] = f"{self.bos_token}{conv_dict_list[0]['user']}"
            conv_dict_list[-1][
                "assistant"
            ] = f"{conv_dict_list[-1]['assistant']}{self.eos_token}"

            return conv_dict_list
            

    def get_generation_prompt(
        self, prompt: str, num_images: int = 1, num_patches: int = 144
    ):
        if prompt.count("<image>") == num_images:
            prompt = prompt.replace("<image>", self.image_token * num_patches)
        else:
            prompt = (
                prompt.replace("<image>\n", "")
                .replace("\n<image>", "")
                .replace("<image>", "")
                .replace("<video>\n", "")
                .replace("\n<video>", "")
                .replace("<video>", "")
            )
            prompt = self.place_image_token(
                prompt,
                self.image_token,
                num_images* num_patches,
            )
        
        sys_text = self.bos_token + self.pre_system + self.system + self.sep_system
        return (
            sys_text + self.pre_question + prompt + self.sep_question + self.pre_answers
        )


    def add_conv(self, conv: Union[List, Dict]):
        if isinstance(conv, list):
            self.conversations.extend(conv)
        elif isinstance(conv, dict):
            self.conversations.append(conv)
        else:
            raise ValueError("Conversation must be a list or a dict.")
    
    def copy(self):
        return Conversation(
            system=self.system,
            conversations=copy.deepcopy(self.conversations),
            place_image_token=self.place_image_token,
            bos_token=self.bos_token,
            sep_system=self.sep_system,
            sep_question=self.sep_question,
            sep_answer=self.sep_answer,
            pre_system=self.pre_system,
            pre_question=self.pre_question,
            pre_answer=self.pre_answer,
            image_token=self.image_token,
            eos_token=self.eos_token,
        )

conv_warmup = Conversation(
    system="",
    conversations=[],
    place_image_token=lambda text, image_token, num_image_tokens: image_token
    * num_image_tokens,  # [warmup] ignores question entirely
    bos_token="",
    pre_system="",
    pre_question="",
    pre_answer="",
    sep_system="",
    sep_question="",
    sep_answer="\n",
    eos_token="",
    image_token="<|image|>",
)

conv_plm_sft = Conversation(
    system="You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    conversations=[],
    place_image_token=lambda text, image_token, num_image_tokens: (
        image_token * num_image_tokens
    )
    + text,
    bos_token="<|begin_of_text|>",
    pre_system="<|start_header_id|>system<|end_header_id|>\n\n",
    pre_question="<|start_header_id|>user<|end_header_id|>\n\n",
    pre_answer="<|start_header_id|>assistant<|end_header_id|>\n\n",
    sep_system="<|eot_id|>",
    sep_question="<|eot_id|>",
    sep_answer="<|eot_id|>",
    eos_token="<|end_of_text|>",
    image_token="<|image|>",
)


REGISTERED_CONVS = {
    "warmup": conv_warmup,
    "plm_sft": conv_plm_sft,
}