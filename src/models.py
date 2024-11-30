import openai,time,re,os
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
from transformers.generation import GenerationConfig
import torch,transformers
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
import datetime
from datetime import date
import json
import config_gpt


def prompt_template(file):
    path = "prompts"
    environment = Environment(loader=FileSystemLoader(path))
    template = environment.get_template(file)
    return environment, template

def readme_write(prompt,model,temperature = 0,max_new_tokens = 5192,top_p = 0.95,num_samples = 1,type = "",path = ""):
    l={"Model":model, "Prompt":prompt,"Temperature":temperature,"Max new tokens":max_new_tokens, "Top_P":top_p, "Samples":num_samples, "Timestamp":datetime.datetime.now()}
    PATH='generations/{}/{}/{}'.format(type,model.split('/')[-1],str(date.today()))
    if path != "":
        PATH = path
    if path == "" and os.path.exists(PATH):
        cnt = 1
        while(os.path.exists(PATH+f"-{cnt}")):
            cnt += 1
        PATH = PATH+f"-{cnt}"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    f=open(os.path.join(PATH, "readme.txt"),"a+")
    f.write(f"{str(l)}\n")
    f.close()
    return PATH

def gpt_query(prompt,model, temperature, max_new_tokens, top_p, num_samples):
    messages=[{"role": "user", "content": prompt}]
    attempt = 1
    max_retries = 100

    if(model == "gpt-4"):
        engine = "gpt-4"
    else:
        engine = "gpt-35-tunro"

    while attempt < max_retries:
        try:
            generated_answers = []
            openai.api_key  =  config_gpt.openai_api_key
            openai.api_base = config_gpt.openai_api_base
            openai.api_type = config_gpt.openai_api_type
            openai.api_version = config_gpt.openai_api_version
            

            # Make the request to the remote LLM API with retries.
            response = openai.ChatCompletion.create(
            engine = engine,
            messages = messages,
            temperature = temperature,
            max_tokens = max_new_tokens,
            top_p = top_p,
            n = num_samples,
            frequency_penalty = 0,
            presence_penalty = 0,
            stop = None)
            #print(response)
            for sample in response['choices']:
                generated_answers.append(sample['message']['content'])
            return generated_answers[0]

        except Exception as e:
            print(e)
            attempt += 1
            seconds = re.search(r"retry after (\d+) seconds", str(e))
            if seconds:
                time.sleep(int(seconds.group(1)))
            else:
                time.sleep(2 * attempt)
            continue

def os_query(pipeline,tokenizer,prompt,temperature,max_new_tokens,top_p,num_samples):
    results = pipeline(
            prompt,
            do_sample = False,
            num_return_sequences=num_samples,
            eos_token_id=tokenizer.eos_token_id,
            max_length= max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            return_full_text=False
        )
    return results[0]["generated_text"]

def llava_query(prompt,image_file,temp,max_new_tokens,top_p,num_samples):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.eval.run_llava import eval_model

    model_path = "liuhaotian/llava-v1.5-7b"
        
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature":temp,
        "top_p":top_p,
        "num_beams":1,
        "max_new_tokens":max_new_tokens})()

    return eval_model(args)

def Qwen_query(model,tokenizer,prompt,image_file,temp,max_new_tokens,top_p,num_samples):
    query = tokenizer.from_list_format([
            {'image': image_file},
            {'text': prompt},
        ])
    if num_samples > 1:
        do_sample = True
    else:
        do_sample = False
    response = model.chat(tokenizer,query = query, history = None, temperature = temp,max_new_tokens = max_new_tokens, top_p = top_p,do_sample = do_sample)
    return response[0]

def fuyu_query(processor,model,prompt,image_file,temp,max_new_tokens,top_p,num_samples):
    image = Image.open(image_file).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")

    # autoregressively generate text
    generation_output = model.generate(**inputs, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temp, num_return_sequences=num_samples)
    generation_text = processor.batch_decode(generation_output[:, -max_new_tokens:], skip_special_tokens=True)
    try:
        generation_text = generation_text[0].split("Response:")[1]
    except:
        generation_text = generation_text[0]
    return generation_text

def cogvlm_query(model,tokenizer,prompt,image_file,temp,max_new_tokens,top_p,num_samples):
    image = Image.open(image_file).convert('RGB')
    inputs = model.build_conversation_input_ids(tokenizer, query=prompt, history=[], images=[image])  # chat mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": max_new_tokens, "do_sample": False, "temperature": temp, "top_p": top_p, "num_return_sequences": num_samples}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0])

def vision_code(data, vision_model, temperature, max_new_tokens, top_p, num_samples, type):

    environment, template = prompt_template("vision_code.txt")
    
    path = readme_write(template.render(),vision_model,temperature,max_new_tokens,top_p,num_samples,type)
    if vision_model.startswith("Qwen"):
        tokenizer = AutoTokenizer.from_pretrained(vision_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(vision_model, device_map="auto", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(vision_model, trust_remote_code=True)
    elif "fuyu" in vision_model:
        processor = FuyuProcessor.from_pretrained(vision_model)
        model = FuyuForCausalLM.from_pretrained(vision_model, device_map="cuda:0")
    elif "cogvlm" in vision_model:
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            vision_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()
    generations = []
    for i in tqdm(range(len(data)),desc = "Generating outputs"):
        messages = data[i]
        image_file = os.path.join("data/images",messages["image"])
        prompt = template.render(prompt = messages["instruction"])
        start = time.time()
        if vision_model.startswith("Qwen"):
            response = Qwen_query(model,tokenizer,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif vision_model.startswith("llava"):
            response = llava_query(prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif "fuyu" in vision_model:
            response = fuyu_query(processor,model,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif "cogvlm" in vision_model:
            response = cogvlm_query(model,tokenizer,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        end = time.time()
        generations.append(dict(instruction = messages['instruction'], output = response,target = messages["target"], inference_time = end - start))
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/generated_outputs.jsonl', 'w') as f:
        json.dump(generations, f,indent = 4)
    return generations

def end_to_end(data, vision_model, temperature, max_new_tokens, top_p, num_samples, type):
    environment, template = prompt_template("e2e.txt")
    path = readme_write(template.render(),vision_model,temperature,max_new_tokens,top_p,num_samples,type)
    if vision_model.startswith("Qwen"):
        tokenizer = AutoTokenizer.from_pretrained(vision_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(vision_model, device_map="auto", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(vision_model, trust_remote_code=True)
    elif "fuyu" in vision_model:
        processor = FuyuProcessor.from_pretrained(vision_model)
        model = FuyuForCausalLM.from_pretrained(vision_model, device_map="cuda:0")
    elif "cogvlm" in vision_model:
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            vision_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()
    generations = []
    for i in tqdm(range(len(data)),desc = "Generating outputs"):
        messages = data[i]
        image_file = os.path.join("data/images",messages["image"])
        prompt = template.render(prompt = messages["instruction"])
        start = time.time()
        if vision_model.startswith("Qwen"):
            response = Qwen_query(model,tokenizer,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif vision_model.startswith("llava"):
            response = llava_query(prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif "fuyu" in vision_model:
            response = fuyu_query(processor,model,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif "cogvlm" in vision_model:
            response = cogvlm_query(model,tokenizer,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        end = time.time()
        generations.append(dict(instruction = messages['instruction'], output = response,target = messages["target"], inference_time = end - start))
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/generated_outputs.jsonl', 'w') as f:
        json.dump(generations, f,indent = 4)
    return generations

def gpt_vision_grounding_code(data, vision_model, code_model, temperature, max_new_tokens, top_p, num_samples, type):
    env,temp = prompt_template("question.txt")

    environment, template = prompt_template("vision_question.txt")
    path = readme_write(template.render(),vision_model.split('/')[-1]+'_'+code_model.split('/')[-1],temperature,max_new_tokens,top_p,num_samples,type)
    if vision_model.startswith("Qwen"):
        tokenizer = AutoTokenizer.from_pretrained(vision_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(vision_model, device_map="auto", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(vision_model, trust_remote_code=True)
    elif "fuyu" in vision_model:
        processor = FuyuProcessor.from_pretrained(vision_model)
        model = FuyuForCausalLM.from_pretrained(vision_model, device_map="cuda:0")
    elif "cogvlm" in vision_model:
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            vision_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()
    if code_model != "gpt-4":
        tokenizer1 = AutoTokenizer.from_pretrained(code_model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=code_model,
            torch_dtype=torch.float16,
            device_map="auto",
            use_fast = True
        )
    environment1, template1 = prompt_template("vgc.txt")
    path = readme_write(template1.render(),vision_model.split('/')[-1]+'_'+code_model.split('/')[-1],temperature,max_new_tokens,top_p,num_samples,type,path)
    generations = []
    for i in tqdm(range(len(data)),desc = "Generating outputs"):
        start = time.time()
        messages = data[i]
        questions = gpt_query(temp.render(instruction = messages["instruction"]),"gpt-4",temperature,max_new_tokens,top_p,num_samples)
        image_file = os.path.join("data/images",messages["image"])
        prompt = template.render(questions = questions) 
        if vision_model.startswith("Qwen"):
            response = Qwen_query(model,tokenizer,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif vision_model.startswith("llava"):
            response = llava_query(prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif "fuyu" in vision_model:
            response = fuyu_query(processor,model,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif "cogvlm" in vision_model:
            response = cogvlm_query(model,tokenizer,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        prompt = template1.render(prompt = messages["instruction"],info = response)
        if code_model == "gpt-4":
            results = gpt_query(prompt,code_model,temperature,max_new_tokens,top_p,num_samples)
        else:
            results = os_query(pipeline,tokenizer1,prompt,temperature,max_new_tokens,top_p,num_samples)
        end = time.time()
        generations.append(dict(instruction = messages['instruction'], questions = questions,extracted_information = response, output = results,target = messages["target"], inference_time = end - start))
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/generated_outputs.jsonl', 'w') as f:
        json.dump(generations, f,indent = 4)
    return generations

def few_shot_vision_grounding_code(data, vision_model, code_model, temperature, max_new_tokens, top_p, num_samples, type):
    
    environment, template = prompt_template("few-shot.txt")
    path = readme_write(template.render(),vision_model.split('/')[-1]+'_'+code_model.split('/')[-1],temperature,max_new_tokens,top_p,num_samples,type)
    if vision_model.startswith("Qwen"):
        tokenizer = AutoTokenizer.from_pretrained(vision_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(vision_model, device_map="auto", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(vision_model, trust_remote_code=True)
    elif "fuyu" in vision_model:
        processor = FuyuProcessor.from_pretrained(vision_model)
        model = FuyuForCausalLM.from_pretrained(vision_model, device_map="cuda:0")
    elif "cogvlm" in vision_model:
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            vision_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()
    if code_model != "gpt-4":
        tokenizer1 = AutoTokenizer.from_pretrained(code_model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=code_model,
            torch_dtype=torch.float16,
            device_map="auto",
            use_fast = True
        )
    environment1, template1 = prompt_template("vgc.txt")
    path = readme_write(template1.render(),vision_model.split('/')[-1]+'_'+code_model.split('/')[-1],temperature,max_new_tokens,top_p,num_samples,type,path)
    generations = []
    for i in tqdm(range(len(data)),desc = "Generating outputs"):
        start = time.time()
        messages = data[i]
        image_file = os.path.join("data/images",messages["image"])
        prompt = template.render(instruction = messages["instruction"]) 
        if vision_model.startswith("Qwen"):
            response = Qwen_query(model,tokenizer,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif vision_model.startswith("llava"):
            response = llava_query(prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif "fuyu" in vision_model:
            response = fuyu_query(processor,model,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif "cogvlm" in vision_model:
            response = cogvlm_query(model,tokenizer,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        prompt = template1.render(prompt = messages["instruction"],info = response)
        if code_model == "gpt-4":
            results = gpt_query(prompt,code_model,temperature,max_new_tokens,top_p,num_samples)
        else:
            results = os_query(pipeline,tokenizer1,prompt,temperature,max_new_tokens,top_p,num_samples)
        end = time.time()
        generations.append(dict(instruction = messages['instruction'], extracted_information = response, output = results,target = messages["target"], inference_time = end - start))
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/generated_outputs.jsonl', 'w') as f:
        json.dump(generations, f,indent = 4)
    return generations

def vision_grounding_code(data, vision_model, code_model, temperature, max_new_tokens, top_p, num_samples, type):
    
    environment, template = prompt_template("grounding.txt")
    path = readme_write(template.render(),vision_model.split('/')[-1]+'_'+code_model.split('/')[-1],temperature,max_new_tokens,top_p,num_samples,type)
    if vision_model.startswith("Qwen"):
        tokenizer = AutoTokenizer.from_pretrained(vision_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(vision_model, device_map="auto", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(vision_model, trust_remote_code=True)
    elif "fuyu" in vision_model:
        processor = FuyuProcessor.from_pretrained(vision_model)
        model = FuyuForCausalLM.from_pretrained(vision_model, device_map="cuda:0")
    elif "cogvlm" in vision_model:
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            vision_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()
    if code_model != "gpt-4":
        tokenizer1 = AutoTokenizer.from_pretrained(code_model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=code_model,
            torch_dtype=torch.float16,
            device_map="auto",
            use_fast = True
        )
    environment1, template1 = prompt_template("vgc.txt")
    path = readme_write(template1.render(),vision_model.split('/')[-1]+'_'+code_model.split('/')[-1],temperature,max_new_tokens,top_p,num_samples,type,path)
    generations = []
    for i in tqdm(range(len(data)),desc = "Generating outputs"):
        start = time.time()
        messages = data[i]
        image_file = os.path.join("data/images",messages["image"])
        prompt = template.render() 
        if vision_model.startswith("Qwen"):
            response = Qwen_query(model,tokenizer,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif vision_model.startswith("llava"):
            response = llava_query(prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif "fuyu" in vision_model:
            response = fuyu_query(processor,model,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        elif "cogvlm" in vision_model:
            response = cogvlm_query(model,tokenizer,prompt,image_file,temperature,max_new_tokens,top_p,num_samples)
        prompt = template1.render(prompt = messages["instruction"],info = response)
        if code_model == "gpt-4":
            results = gpt_query(prompt,code_model,temperature,max_new_tokens,top_p,num_samples)
        else:
            results = os_query(pipeline,tokenizer1,prompt,temperature,max_new_tokens,top_p,num_samples)
        end = time.time()
        generations.append(dict(instruction = messages['instruction'], extracted_information = response, output = results,target = messages["target"], inference_time = end - start))
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/generated_outputs.jsonl', 'w') as f:
        json.dump(generations, f,indent = 4)
    return generations

def ideal_grounding_code(data, code_model, temperature, max_new_tokens, top_p, num_samples, type):
    if code_model != "gpt-4":
        tokenizer = AutoTokenizer.from_pretrained(code_model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=code_model,
            torch_dtype=torch.float16,
            device_map="auto",
            use_fast = True
        )
    environment, template = prompt_template("igc.txt")
    path = readme_write(template.render(),code_model.split('/')[-1],temperature,max_new_tokens,top_p,num_samples,type)
    generations = []
    for i in tqdm(range(len(data)),desc = "Generating outputs"):
        start = time.time()
        messages = data[i]  
        prompt = template.render(prompt = messages["instruction"],info = messages["ideal_grounding"]['content'],comment = messages["ideal_grounding"].get("comment","comment"))
        if code_model == "gpt-4":
            results = gpt_query(prompt,code_model,temperature,max_new_tokens,top_p,num_samples)
        else:
            results = os_query(pipeline,tokenizer,prompt,temperature,max_new_tokens,top_p,num_samples)
        end = time.time()
        generations.append(dict(instruction = messages['instruction'], ideal_information = messages["ideal_grounding"], output = results,target = messages["target"], inference_time = end - start))
    with open(f'{path}/generated_outputs.jsonl', 'w') as f:
        json.dump(generations, f,indent = 4)
    return generations