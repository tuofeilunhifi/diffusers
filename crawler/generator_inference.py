import torch
from diffusers import StableDiffusionPipeline

model_path = "/mnt/workspace/workgroup/yunji.cjy/projects/text-to-image/sd1.5-wallhaven-science_fiction_all-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

# # prompt = "Concept art of a hostile alien planet with unbreathable purple air and toxic clouds, sinister atmosphere, deep shadows, sharp details"
# # prompt = "Concept art of a hostile alien planet with unbreathable purple air and toxic clouds, sinister atmosphere, deep shadows, sharp details, 8k, photography"
# # prompt = "cyberpunk locomotive on railroad through cyberpunk industrial site. cyberpunk factories. cyberpunk kowloon. rail tracks. cyberpunk industrial area. Digital render. digital painting. Beeple. Noah Bradley. Cyril Roland. Ross Tran. trending on artstation."
# # prompt = "Ironman as a businessman, realistic, dramatic light"
# # prompt = "Synthwave halloween formula 1 car racing on a night road in Singapore"
prompt = "a dog sitting on a blue wall"
image = pipe(prompt=prompt).images[0]

image.save("science_fiction_all_dog.png")

# from diffusers import StableDiffusionPipeline
# import torch

# model_path = "/mnt/workspace/workgroup/yunji.cjy/projects/text-to-image/sd1.5-wallhaven-science_fiction_all-model_lora/"
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipe.unet.load_attn_procs(model_path)
# pipe.to("cuda")

# # prompt = "Concept art of a hostile alien planet with unbreathable purple air and toxic clouds, sinister atmosphere, deep shadows, sharp details"
# # prompt = "Concept art of a hostile alien planet with unbreathable purple air and toxic clouds, sinister atmosphere, deep shadows, sharp details, 8k, photography"
# # prompt = "cyberpunk locomotive on railroad through cyberpunk industrial site. cyberpunk factories. cyberpunk kowloon. rail tracks. cyberpunk industrial area. Digital render. digital painting. Beeple. Noah Bradley. Cyril Roland. Ross Tran. trending on artstation."
# # prompt = "Ironman as a businessman, realistic, dramatic light"
# # prompt = "Synthwave halloween formula 1 car racing on a night road in Singapore"
# prompt = "a dog sitting on a blue wall"
# image = pipe(prompt).images[0]
# image.save("science_fiction_all_lora_dog.png")