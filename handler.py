
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ts.torch_handler.base_handler import BaseHandler

class EmailReplyHandler(BaseHandler):

    def initialize(self, context):
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        item = data[0].get("body") or data[0].get("data")

        if isinstance(item, bytes):
            item = item.decode("utf-8")

        if isinstance(item, str):
            import json
            item = json.loads(item)

        email = item.get("email", "")
        tone = item.get("tone", "Formal")

        prompt = f"Email: {email[:500]}\nWrite a {tone} reply:\n"
        return self.tokenizer(prompt, return_tensors="pt")

    def inference(self, inputs):
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return output, inputs["input_ids"].shape[1]

    def postprocess(self, inference_output):
        output, input_length = inference_output
        reply = self.tokenizer.decode(
            output[0][input_length:],
            skip_special_tokens=True
        )
        return [{"reply": reply.strip()}]