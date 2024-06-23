import json
import math
import re
import warnings
from collections import defaultdict
from typing import List, Optional

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

import time
import numpy as np
import os

INFILL_MODE = False
INSTRUCTION_MODE = False


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instructin-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        num_devices,
        max_length,
        limit_start=0,
        n_tasks=None,
        n_copies=1,
        prefix="",
        has_encoder=False,
        instruction_tokens=None,
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_devices = num_devices
        self.max_length = max_length
        self.limit_start = limit_start
        self.n_tasks = n_tasks
        self.n_copies = n_copies
        self.prefix = prefix
        self.has_encoder = has_encoder
        self.instruction_tokens = instruction_tokens

    def __iter__(self):
        prompts = []
        prompts_encoder = []
        infill = []
        instruction = []
        for sample in range(self.limit_start, self.limit_start + self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                # Normal code completion mode
                infill.append(False)
                instruction.append(False)
                prompt = self.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                if set(prompt_contents.keys()) == {"prefix", "suffix"}:
                    # Infilling mode
                    infill.append(True)
                    instruction.append(False)
                    prompt = self._make_infill_prompt(
                        **prompt_contents, preprefix=self.prefix
                    )
                elif set(prompt_contents.keys()) == {"instruction", "context"}:
                    # Instruction-tuning mode
                    instruction.append(True)
                    infill.append(False)
                    prompt = self._make_instruction_prompt(
                        **prompt_contents, prefix=self.prefix
                    )
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)
            if self.has_encoder:
                prompt_encoder = self.task.get_prompt_encoder(self.dataset[sample])
                if isinstance(prompt_encoder, str):
                    prompt_encoder = self.prefix + prompt_encoder
                prompts_encoder.append(prompt_encoder)

        if not len(set(infill)) == 1 or not len(set(instruction)) == 1:
            raise ValueError(
                "Mixed infill/instruction and completion prompts are not supported."
            )
        global INFILL_MODE
        global INSTRUCTION_MODE
        INFILL_MODE = infill[0]
        INSTRUCTION_MODE = instruction[0]
        if INFILL_MODE:
            return_token_type_ids = False
        else:
            return_token_type_ids = None  # default

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=return_token_type_ids,
        )
        if self.has_encoder:
            outputs_encoder = self.tokenizer(
                prompts_encoder,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
                return_token_type_ids=return_token_type_ids,
            )

        if self.n_copies == 1 and self.n_tasks % self.num_devices != 0:
            self.n_copies = 2
            warnings.warn(
                "n_copies (n_samples/batch_size) was changed from 1 to 2 because n_tasks isn't proportional to num devices"
            )

        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                if self.has_encoder:
                    yield {
                        "ids": outputs.input_ids[sample],
                        "ids_encoder": outputs_encoder.input_ids[sample],
                        "task_id": sample,
                        "input_len": outputs.attention_mask[sample].sum(),
                        "input_len_encoder": outputs_encoder.attention_mask[
                            sample
                        ].sum(),
                    }
                else:
                    yield {
                        "ids": outputs.input_ids[sample],
                        "task_id": sample,
                        "input_len": outputs.attention_mask[sample].sum(),
                    }

    def _make_infill_prompt(self, prefix, suffix, preprefix=""):
        """Make a prompt for infilling.
        Currently supported only for official InCoder and SantaCoder implementations.
        """
        model_id = self.tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            return f"{preprefix}{prefix}<|mask:0|>{suffix}<|mask:0|>"
        elif model_id in ["bigcode/santacoder"]:
            return f"<fim-prefix>{preprefix}{prefix}<fim-suffix>{suffix}<fim-middle>"
        elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
            return f"<fim_prefix>{preprefix}{prefix}<fim_suffix>{suffix}<fim_middle>"
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")

    def _make_instruction_prompt(self, instruction, context, prefix=""):
        """Make a prompt for instruction-tuning. Delimit instruction and context with specific tokens if provided."""
        if not self.instruction_tokens:
            warnings.warn(
                "Instruction-tuning tokens are not provided for an instruction-tuning task, we will leave them empty."
            )
            user_token, end_token, assistant_token = "", "", "\n"
        else:
            user_token, end_token, assistant_token = self.instruction_tokens
            if not user_token or not assistant_token or not end_token:
                warnings.warn(
                    "Instruction-tuning tokens provided but one or more are empty. Ignore warning if this was intended"
                )
        prompt = (
            prefix + user_token + instruction + end_token + assistant_token + context
        )

        return prompt


def _parse_infill(code, tokenizer):
    """Reorder infill code and remove remaining special tokens."""
    model_id = tokenizer.name_or_path
    if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
        prefix, suffix, infill = code.split("<|mask:0|>", 2)
        infill = infill.split("<|endofmask|>")[0]
    elif model_id in ["bigcode/santacoder"]:
        prefix, rest = code.split("<fim-suffix>", 1)
        suffix, infill = rest.split("<fim-middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
        prefix, rest = code.split("<fim_suffix>", 1)
        suffix, infill = rest.split("<fim_middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    else:
        raise ValueError(f"Infilling not yet supported for: {model_id}")
    for k, v in tokenizer.special_tokens_map.items():
        if k == "additional_special_tokens":
            for t in v:
                infill = infill.replace(t, "")
        else:
            infill = infill.replace(v, "")
    return infill


def _parse_instruction(code, instruction_tokens):
    """Return code block after assistant_token/end_token"""
    _, end_token, assistant_token = instruction_tokens
    if not assistant_token and end_token:
        assistant_token = end_token
    elif not assistant_token and not end_token:
        return code

    idx = code.find(assistant_token)
    shift = len(assistant_token)
    if idx == -1:
        warnings.warn(
            "The assistant token was not detected in the generation, this might disrupt the post-processing and lead to lower evaluation scores"
        )
        return code

    if "```python" in assistant_token:
        idx = code.find("```python", idx)
        shift = len("```python")
    return code[idx + shift :]


def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    limit_start=0,
    batch_size=20,
    prefix="",
    instruction_tokens=None,
    postprocess=True,
    is_wrapped=False,
    save_every_k_tasks: int = -1,
    intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
    intermediate_save_generations_path: Optional[str] = None,
    model_name=None,
    **gen_kwargs,
):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """
    # keep track of the list of generated codes
    # where len(code_gens) = n_tasks and len(code_gens[0]) = number of generated code samples
    code_gens: List[List[Optional[str]]] = [[] for _ in range(n_tasks)]
    generations = [] if not intermediate_generations else intermediate_generations
    gen_token_dict = defaultdict(list)  # dict of list of generated tokens

    timer_results = list()

    # import torchao

    # from torch import nn
    # def layers_to_quantize(layer, x):
    #     if isinstance(layer, nn.Linear):
    #         return True
    #     return False
    
    # model = torchao.autoquant(model, manual=True, filter_fn=layers_to_quantize)
    # # model = torchao.autoquant(model, manual=True)

    # print("Autoquant shape calibration start") 

    # warmup_input = torch.tensor([[    1,   515, 19229,  1053,  2391,    13,    13,    13,  1753,   756,
    #         29918,  5358, 29918, 17664, 29898, 20326, 29901,  2391, 29961,  7411,
    #         1402, 16897, 29901,  5785, 29897,  1599,  6120, 29901,    13,  1678,
    #         9995,  5399,   565,   297,  2183,  1051,   310,  3694, 29892,   526,
    #         738,  1023,  3694, 17649,   304,  1269,   916,  1135,    13,  1678,
    #         2183, 16897, 29889,    13,  1678,  8653,   756, 29918,  5358, 29918,
    #         17664,  4197, 29896, 29889, 29900, 29892, 29871, 29906, 29889, 29900,
    #         29892, 29871, 29941, 29889, 29900,  1402, 29871, 29900, 29889, 29945,
    #         29897,    13,  1678,  7700,    13,  1678,  8653,   756, 29918,  5358,
    #         29918, 17664,  4197, 29896, 29889, 29900, 29892, 29871, 29906, 29889,
    #         29947, 29892, 29871, 29941, 29889, 29900, 29892, 29871, 29946, 29889,
    #         29900, 29892, 29871, 29945, 29889, 29900, 29892, 29871, 29906, 29889,
    #         29900,  1402, 29871, 29900, 29889, 29941, 29897,    13,  1678,  5852,
    #             13,  1678,  9995]]).cuda()
    
    # model.generate(
    #     input_ids=warmup_input,
    #     num_return_sequences=batch_size,
    #     **gen_kwargs,
    # )

    # print("Autoquant benchmarking Start") 
    # # do autoquantization
    # model.do_autoquant()

    torch._dynamo.config.cache_size_limit = 32
    compile = os.environ.get('TORCH_COMPILE', False)
    if compile:
        print("Torch.compile Start")
        # model = torch.compile(model, mode='max-autotune', fullgraph=True) # -> It's not fast
        print("Torch.compile - Prefill")
        model.model = torch.compile(model.model, mode='max-autotune', dynamic=True)#, fullgraph=True)
        print("Torch.compile - Decoding")
        model.model.forward2 = torch.compile(model.model.forward2, mode='max-autotune')#, fullgraph=True)

    # # # Warmup
    # # print("Starting Warmup")    
    # warmup_input = torch.tensor([[    1,   515, 19229,  1053,  2391,    13,    13,    13,  1753,   756,
    #         29918,  5358, 29918, 17664, 29898, 20326, 29901,  2391, 29961,  7411,
    #         1402, 16897, 29901,  5785, 29897,  1599,  6120, 29901,    13,  1678,
    #         9995,  5399,   565,   297,  2183,  1051,   310,  3694, 29892,   526,
    #         738,  1023,  3694, 17649,   304,  1269,   916,  1135,    13,  1678,
    #         2183, 16897, 29889,    13,  1678,  8653,   756, 29918,  5358, 29918,
    #         17664,  4197, 29896, 29889, 29900, 29892, 29871, 29906, 29889, 29900,
    #         29892, 29871, 29941, 29889, 29900,  1402, 29871, 29900, 29889, 29945,
    #         29897,    13,  1678,  7700,    13,  1678,  8653,   756, 29918,  5358,
    #         29918, 17664,  4197, 29896, 29889, 29900, 29892, 29871, 29906, 29889,
    #         29947, 29892, 29871, 29941, 29889, 29900, 29892, 29871, 29946, 29889,
    #         29900, 29892, 29871, 29945, 29889, 29900, 29892, 29871, 29906, 29889,
    #         29900,  1402, 29871, 29900, 29889, 29941, 29897,    13,  1678,  5852,
    #             13,  1678,  9995]], device='cuda:0')
    # # gen_kwargs = {'do_sample': True, 'temperature': 0.2, 'top_p': 0.95, 'top_k': 0, 'max_length': 10000}
    # gen_kwargs["stopping_criteria"][0].start_length = 133

    # for i in range(15):
    #     _, runtime = model.generate(
    #         input_ids=warmup_input,
    #         num_return_sequences=batch_size,
    #         **gen_kwargs,
    #     )
    #     print(runtime)

    # print("Ending Warmup")
    # # Warmup
    # exit(0)
    # 430 for humaneval, 1748 for mbpp


    for step, batch in tqdm(
        enumerate(dataloader),
        total=math.ceil(
            n_tasks * dataloader.dataset.n_copies / accelerator.num_processes
        ),
    ):
        with torch.no_grad():
            if task.stop_words:
                # Set the start_length after which to check for stopping to be the longest input ignoring padding
                max_len = batch["input_len"].max().item()
                if "ids_encoder" in batch:
                    max_len += 1  # Add 1 for decoder_start_token_id
                gen_kwargs["stopping_criteria"][0].start_length = max_len
            if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
                idx = 1 if task.stop_words else 0
                gen_kwargs["stopping_criteria"][idx].input_length = (
                    batch["input_len"].max().item()
                )

            inputs = batch["ids"][:, : batch["input_len"]] if tokenizer.padding_side == "right" else batch["ids"]
            if "ids_encoder" in batch:
                if is_wrapped:
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        decoder_input_ids=inputs,
                        input_ids=batch["ids_encoder"][:, : batch["input_len_encoder"]],
                        num_return_sequences=batch_size,
                        decoder_start_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **gen_kwargs,
                    )
                else:
                    generated_tokens = model.generate(
                        decoder_input_ids=inputs,
                        input_ids=batch["ids_encoder"][:, : batch["input_len_encoder"]],
                        num_return_sequences=batch_size,
                        decoder_start_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **gen_kwargs,
                    )
            else:
                if is_wrapped:
                    # 8bit and 4bit models are wrapped in accelerator
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        input_ids=inputs,
                        num_return_sequences=batch_size,
                        **gen_kwargs,
                    )
                else:
                    # new_inputs = torch.zeros([inputs.shape[0], 500], dtype=inputs.dtype).cuda()
                    # new_inputs[:, :inputs.shape[1]] = inputs
                    # padding_mask = torch.zeros([inputs.shape[0], 500], dtype=torch.bool).cuda()
                    # padding_mask[:, :inputs.shape[1]] = True
                    # for l in model.model.layers:
                    #     l.self_attn.padding_mask = padding_mask

                    # generated_tokens, timer_result = model.generate(
                    #     input_ids=new_inputs,
                    #     num_return_sequences=batch_size,
                    #     **gen_kwargs,
                    # )
                    generated_tokens, timer_result = model.generate(
                        input_ids=inputs,
                        num_return_sequences=batch_size,
                        **gen_kwargs,
                    )

            # each task is generated batch_size times
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather(
                (generated_tokens, generated_tasks)
            )
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for sample, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[sample].append(generated_tokens)

            if save_every_k_tasks >= 1 and (step + 1) % save_every_k_tasks == 0:
                if not intermediate_save_generations_path:
                    raise ValueError(
                        "intermediate_save_generations_path cannot be empty!"
                    )
                torch.cuda.synchronize()
                start_time = time.time()
                code_gens = update_code_gens(
                    task,
                    tokenizer,
                    limit_start,
                    prefix,
                    instruction_tokens,
                    postprocess,
                    code_gens,
                    gen_token_dict,
                )
                torch.cuda.synchronize()
                timer_result["TextDecode"] = (time.time()-start_time)*1000
                # with open(intermediate_save_generations_path, "w") as fp:
                #     json.dump(generations + code_gens, fp)
                #     print(
                #         f"intermediate generations were saved at {intermediate_save_generations_path}"
                #     )
                # reset gen_token_dict - prevent redundant decoding
                gen_token_dict = defaultdict(list)
        if timer_result["Total"] != -1:
            timer_results.append(timer_result)
            print(timer_result)
        else:
            print("Skipping this example")

    dump_dir = "/fsx-atom/yejinlee/paper_submission_results/torch_compile/1gpu_1node/"+task.__class__.__name__+"_codellama/"+str(model_name)+"/batch_size_"+str(batch_size) if compile \
        else "/fsx-atom/yejinlee/paper_submission_results/torch_compile_baseline/1gpu_1node/"+task.__class__.__name__+"_codellama/"+str(model_name)+"/batch_size_"+str(batch_size)
    os.makedirs(dump_dir, exist_ok=True)
    with open(dump_dir+"/timer_result.txt", "w") as f:
        f.write("\t".join(list(timer_results[0].keys()))+"\n")
        write_str=""
        for tr in  timer_results:
            write_str += "\t".join([str(t) for t in list(tr.values())]) + "\n"
        f.write(write_str)
    print("Written to : ", dump_dir+"/timer_result.txt")

    exit(0)

    code_gens = update_code_gens(
        task,
        tokenizer,
        limit_start,
        prefix,
        instruction_tokens,
        postprocess,
        code_gens,
        gen_token_dict,
    )

    generations.extend(code_gens)
    return generations


def update_code_gens(
    task,
    tokenizer,
    limit_start,
    prefix,
    instruction_tokens,
    postprocess,
    code_gens,
    gen_token_dict,
):  
    for sample, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            if INFILL_MODE or tokenizer.eos_token in task.stop_words:
                if s[0] == tokenizer.bos_token_id:
                    s = s[1:]
                # Treat eos token as a regular stop word not removing it from the output
                # If it's removed it may have the effect of removing it in the middle of a
                # longer generation in case a batch size > 1 is used, which will result in
                # a wrong generation as it won't be used for splitting lateron
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                try:
                    # some tokenizers add a multi-token prefix to the generation (e.g ChatGLM)
                    tokenizer_prefix = tokenizer.decode(tokenizer.get_prefix_tokens())
                    if gen_code.startswith(f"{tokenizer_prefix}"):
                        gen_code = gen_code[len(tokenizer_prefix):].lstrip()
                except:
                    pass
                if INFILL_MODE:
                    gen_code = _parse_infill(gen_code, tokenizer)
                if INSTRUCTION_MODE:
                    gen_code = _parse_instruction(gen_code, instruction_tokens)
            else:
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            if not INFILL_MODE:
                gen_code = gen_code[len(prefix) :]
            if postprocess:
                code_gens[sample].append(
                    task.postprocess_generation(gen_code, int(sample) + limit_start)
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens[sample].append(gen_code)
    return code_gens


def remove_after_return(code):
    """
    Takes as input a code, and removes everything that is after the return.
    That is, the first line that does not start with a space character
    """
    pattern = r"[^\n]+(\n|$)"
    end_last_match = None
    # Go trough the regex to match any sequence of characters ending with a \n
    for match in re.finditer(pattern, code):
        start_match, end_match = match.span()
        # Search for the first line which does not start by a space character
        if (
            end_last_match is not None
            and start_match < len(code)
            and code[start_match].strip() != ""
        ):
            return code[0: start_match]
        end_last_match = end_match
    return code
