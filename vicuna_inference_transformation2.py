#
#! This code is used to generate positives. Follow up with vicuna_postprocessing.py to further cleanup some more generations.

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template, add_model_args
import pandas as pd
import time
from transformers import set_seed
from old_vicuna_inference_ppfsnegfs_examples import pos0_template, pos0_template_examples, pos1_template, pos2_template, pos3_template, pos4_template, pos1_template_examples, pos2_template_examples, pos3_template_examples, pos4_template_examples
import numpy as np




def get_prompt(version, input_text):
    if(version==0):
        pos_template = pos0_template
        pos_template_examples = pos0_template_examples
    elif(version==1):
        pos_template = pos1_template
        pos_template_examples = pos1_template_examples
    elif(version==2):
        pos_template = pos2_template
        pos_template_examples = pos2_template_examples
    elif(version==3):
        pos_template = pos3_template
        pos_template_examples = pos3_template_examples
    elif(version==4):
        pos_template = pos4_template
        pos_template_examples = pos4_template_examples
    else:
        assert False, ("version is", version)

    if(len(pos_template_examples)==10):
        demos_ids = np.random.choice(10, 5, replace=False)
        text_list = [each for demo_id in demos_ids for each in pos_template_examples[demo_id]]+[input_text]
        # import ipdb; ipdb.set_trace()
        llm_input_prompt = pos_template[0].format(*text_list)
    else:
        text_list = [input_text]
        # import ipdb; ipdb.set_trace()
        llm_input_prompt = pos_template[0].format(*text_list)

    return llm_input_prompt, pos_template[1]



def postprocess(input_text):
    return input_text.split("Explanation:")[0]




@torch.inference_mode()
def main(args):
    dataset = pd.read_csv(args.input_path, header=None)
    # dataset.columns = ["sent0", "sent1", "hard_neg"]

    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )



    def query_llm(msg, agent_pre):
        conv = get_conversation_template(args.model_path)
        # print(conv)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        prompt = prompt + agent_pre
        # print(prompt)
        # bad_words_ids = [[10567],[4290], [13]]

        input_ids = tokenizer([prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            # bad_words_ids=bad_words_ids,
            # top_p=0.9
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        return outputs
    
    CL_dataset = []
    start_time = time.time()
    for idx, row in dataset.iterrows():
        print(idx, flush=True)
        if(idx%10==0):
            print(time.time()-start_time)
        # if(idx==20):
        #     break;

        # import ipdb; ipdb.set_trace()

        transformedrow = query_llm(*get_prompt(args.transformation, row[1])).replace("\n", "  ")

        ranked_generations = [row[0], transformedrow]

        # print(ranked_generations)
        CL_dataset.append(ranked_generations)

    CL_dataframe = pd.DataFrame(CL_dataset)
    CL_dataframe.to_csv(args.output_path, header=None, index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--top_p", type=float, default=1)

    parser.add_argument("--transformation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    set_seed(args.seed)
    print("Seed", args.seed)
    print("transformation", args.transformation)


    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)
