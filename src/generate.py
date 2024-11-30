import argparse
from datasets import load_dataset
import json

from models import end_to_end, vision_grounding_code, ideal_grounding_code, vision_code,few_shot_vision_grounding_code,gpt_vision_grounding_code
import os
from tqdm import tqdm



def model_query(data, vision_model, code_model, temperature, max_new_tokens, top_p, num_samples,type):
    if type == "E2E":
        generated_answers = end_to_end(data, vision_model, temperature, max_new_tokens, top_p, num_samples,type)
    elif type == "VGC":
        generated_answers = vision_grounding_code(data, vision_model, code_model, temperature, max_new_tokens, top_p, num_samples,type)
    elif type == "IGC":
        generated_answers = ideal_grounding_code(data, code_model, temperature, max_new_tokens, top_p, num_samples,type)
    elif type == "VC":
        generated_answers = vision_code(data, vision_model, temperature, max_new_tokens, top_p, num_samples,type)
    elif type == "FSVGC":
        generated_answers = few_shot_vision_grounding_code(data, vision_model, code_model, temperature, max_new_tokens, top_p, num_samples,type)
    elif type == "GPTVGC":
        generated_answers = gpt_vision_grounding_code(data, vision_model, code_model, temperature, max_new_tokens, top_p, num_samples,type)
    return generated_answers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/tasks.json", help="dataset")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature")
    parser.add_argument("--vision_model", type=str, default="Qwen/Qwen-VL-Chat", help="model")
    parser.add_argument("--code_model", type=str, default="WizardLM/WizardCoder-15B-V1.0", help="model")
    parser.add_argument("--max_new_tokens", type=int, default=3096, help="max length of tokens")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p")
    parser.add_argument("--num_samples", type=int, default=1, help="number of samples")
    parser.add_argument("--type", type=str, default="E2E", help="number of samples")
    args = parser.parse_args()
    #data = load_dataset('json', data_files=args.data, split = 'train')
    data = eval(open(args.data).read())
    generated_answers = model_query(data, args.vision_model,args.code_model, args.temperature, args.max_new_tokens, args.top_p, args.num_samples,args.type)
    #path = f"generations/{args.model}
    #if not os.path.exists(path):
    #    os.makedirs(path)
    #with open(f'{path}/generated_outputs.jsonl', 'w') as f:
    #    json.dump(generated_answers, f,indent = 4)