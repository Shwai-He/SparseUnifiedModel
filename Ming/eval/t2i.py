import os
import argparse
from safetensors.torch import load_file

import torch
from data.data_utils import add_special_tokens
from modeling.causalfusion_navit import CausalFusionConfig, CausalFusion, Qwen2Config, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

from PIL import Image
from data.transforms import ImageTransform
from modeling.causalfusion_navit.qwen2_navit import NaiveCache
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Generate images using CausalFusion model.")
parser.add_argument("--resume_from", type=str, required=True, help="Path to the checkpoint directory to resume from.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
parser.add_argument("--llm_path", type=str, default="/mnt/bn/seed-aws-va/likunchang/hf/Qwen2.5-7B-Instruct/")
parser.add_argument("--vae_path", type=str, default="/mnt/bn/seed-aws-va/chaorui/flux/vae/ae.safetensors")
parser.add_argument("--no-ema", action="store_true")
args = parser.parse_args()

resume_from = args.resume_from
os.makedirs(args.output_dir, exist_ok=True)
print(f"Output images are saved in {args.output_dir}")

llm_config = Qwen2Config.from_pretrained(args.llm_path)
llm_config.qk_norm = True
llm_config.layer_module = "Qwen2MoTDecoderLayer"
llm_config.tie_word_embeddings = False
language_model = Qwen2ForCausalLM.from_pretrained(args.llm_path, config=llm_config)
vae_model, vae_config = load_ae(local_path=args.vae_path)
config = CausalFusionConfig(
    visual_gen=True,
    visual_und=False,
    llm_config=llm_config, 
    vae_config=vae_config,
    latent_patch_size=2,
    max_latent_size=32,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
)
model = CausalFusion(language_model, None, config)

tokenizer = Qwen2Tokenizer.from_pretrained(args.llm_path)
tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
if num_new_tokens > 0:
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.config.llm_config.vocab_size = len(tokenizer)
    model.language_model.config.vocab_size = len(tokenizer)

if args.no_ema:
    state_dict_path = os.path.join(resume_from, f"model.safetensors")
else:
    state_dict_path = os.path.join(resume_from, f"ema.safetensors")
state_dict = load_file(state_dict_path, device="cpu")
state_dict.pop("latent_pos_embed.pos_embed")
msg = model.load_state_dict(state_dict, strict=False)
print(msg)

model = model.cuda().eval()
vae_model = vae_model.cuda().eval()
vae_transform = ImageTransform(512, 256, 16)
gen_model = model


def generate_image(prompt, num_timesteps=50, cfg_scale=10.0, cfg_interval=None, timestep_shift=1.0, seed=42):
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(512, 512)], 
        new_token_ids=new_token_ids,
    )

    generation_input_cfg = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=[0],
        curr_rope=[0], 
        image_sizes=[(512, 512)], 
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
            past_key_values=past_key_values,
            cfg_past_key_values=None,
            num_timesteps=num_timesteps,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            timestep_shift=timestep_shift,
            **generation_input,
            **generation_input_cfg,
        )

    latent0 = unpacked_latent[0]
    latent0 = latent0.reshape(1, 32, 32, 2, 2, 16)
    latent0 = torch.einsum("nhwpqc->nchpwq", latent0)
    latent0 = latent0.reshape(1, 16, 64, 64)
    image = vae_model.decode(latent0)
    tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmpimage = Image.fromarray(tmpimage)

    return tmpimage


prompt_list = [
    "Photorealistic closeup image of two pirate ships battling each other as they sail inside a cup of coffee.",
    "An elephant entirely composed of various and colorful leaves with clear - visible veins. It stands on a vibrant forest meadow, surrounded by trees of different heights. Sunlight filters through the gaps in the branches and leaves, casting light on the elephant and the ground, creating a dreamy and natural atmosphere.",
    "未来城市中，高耸入云的金属建筑表面闪烁着银光，磁悬浮列车在透明轨道上高速行驶，街边有机器人在进行清洁工作。",
    "A young woman with long, flowing hair, wearing a vintage sundress, standing in a field of wildflowers. She has a gentle smile, and the sunlight creates a soft, ethereal glow around her. The image is styled in a watercolor painting format, with delicate brushstrokes and pastel colors.",
    "A young Asian woman with fair skin has shoulder-length wavy curly dark brown hair, and the ends of the hair are slightly turned inwards. Under the curved willow-leaf eyebrows are a pair of bright almond-shaped eyes, with the outer corners of the eyes slightly raised, and her eyes are as bright as black gemstones. Her nose is straight and tall, and her lips are rosy and plump, revealing a gentle smile. She is wearing a light blue chiffon dress embroidered with delicate white small flower patterns, and the skirt is fluffy. She has a pair of white flat sandals on her feet, and a thin golden anklet is tied around her ankle. Her hands are naturally drooping, and one hand is gently holding a beige woven handbag. She is standing in a park filled with large patches of pink cherry blossoms. Behind her is an ancient wooden pavilion, and light cherry blossom petals are floating around.",
    "An afrofuturist lady wearing gold jewelry.",
    "A cute golden retriever is sitting on a wooden bench near a clear lake. The lake has calm, shimmering blue - green water, with small ripples created by a gentle breeze. In the background, there are tall, lush trees with bright green leaves swaying slightly. The sky above is a clear, vivid blue, dotted with a few fluffy white clouds. The dog has a friendly expression on its face, its tongue slightly out, looking out at the beautiful lake scenery. ",
    "A majestic Siberian tiger stands gracefully on a rocky outcrop in a dense forest. Its fur is a rich orange hue with bold black stripes, glistening under the dappled sunlight filtering through the tall trees. The tiger's eyes are a piercing yellow, exuding an air of strength and authority. Around it, the forest is alive with greenery, including lush ferns and towering conifers. The ground is covered with a mix of fallen leaves and patches of moss. In the background, a gentle stream meanders through the woods, adding to the serene and natural setting.",
    "A chubby giant panda sits under a leafy bamboo tree in a peaceful bamboo forest. Its fur is a distinct combination of pure white and deep black, with round, black - rimmed eyes that make it look incredibly endearing. The panda has a bamboo shoot in its paws, nibbling on it contentedly, and bits of bamboo leaves are scattered around. The sunlight filters through the thick bamboo canopy, creating dappled shadows on the forest floor. The ground is covered with fresh green bamboo grass, and the air is filled with the gentle rustling of bamboo leaves in the soft breeze. It's a serene and idyllic scene, perfectly capturing the panda's laid - back and charming nature.",
    "A wooden desk by the window. Its surface is smooth, with a stack of books on one side and a lamp with a soft - glowing bulb on the other. Beside the lamp, there's a half - filled coffee mug and a pen. A small potted plant adds a bit of greenery.",
    "Beautiful oil painting of a steamboat in a river.",
    "A bustling ancient Chinese street scene during a festival. Red lanterns of various shapes and sizes are strung up, casting a warm glow. People in traditional Hanfu are strolling, chatting, and engaging in different activities. Some are buying handicrafts from street vendors, while others are enjoying snacks. There are also acrobatic performers in the middle of the street, drawing a crowd. The buildings on both sides are in classic Chinese architectural style with upturned eaves and carved wooden doors. The sky is filled with floating colorful kites, adding to the lively atmosphere.",
    "A Gothic cathedral stands tall against the blue sky. Its spires are topped with crosses, reaching towards the clouds. Pointed arches and stained - glass windows adorn the facade. Gargoyles sit on the edges, looking fierce. In the courtyard, cobblestones form paths, and flower beds are filled with colorful flowers.",
    "在夜晚，一座现代化的城市灯火辉煌。高楼大厦的窗户透出温暖的灯光，霓虹灯闪烁着五彩斑斓的光芒。街道上车水马龙，车灯形成一条条流动的光带。远处可以看到城市的标志性建筑，在灯光的映照下显得格外壮观。天空中挂着一轮明月，为这座繁华的城市增添了一份宁静。",

    '''A movie poster for a film titled "Conductor." The poster features a person in a dark suit, holding a conductor's baton, with their left hand raised in a gesture that suggests they are leading or guiding. The background is dark and somewhat abstract, with a hint of a stage or performance setting. The title "CONDUCOR" is prominently displayed at the top in bold, white capital letters. Below the title, the subtitle "Music for the body" is written in a smaller, white font. The overall design is sleek and professional, with a focus on the conductor's role and the theme of music and performance''',
    '''A mathematical equation written in a simple, clean font. The equation is "1 + 1 = 2". The numbers "1" and "1" are placed on the left side of the equals sign, and the number "2" is placed on the right side of the equals sign. The equation is centered on the image, with a white background that provides a clear contrast to the black text. The overall appearance is minimalistic and straightforward, designed to convey a basic arithmetic operation.''',
    '''A bottle of juice with an orange-colored liquid, lying on a white, slightly wrinkled fabric. The bottle is tall and narrow with a simple, white, flip-top lid. The label on the bottle is predominantly orange and is canceled. At the top of the label, there is a circular logo with some text inside it, but the text is too small to read clearly. Below the logo, the word "ROOTS" is prominently displayed in large, white, uppercase letters, followed by the number "5" in smaller white font. Underneath, the ingredients are listed: "Carrot, Pineapple, Ginger," each item separated by a comma and followed by a period. At the bottom of the label, it states "Cold Pressed & HPP," indicating the juice is cold-pressed and high-pressure pasteurized. The volume of the juice is indicated as "473ml += 10cc" on the label. The overall appearance of the bottle suggests it is a healthy, organic, and possibly fresh-rendered juice mix designed for a daily diet.''',
    '''A close-up of a handwritten note on a piece of paper. The text reads ".:I Love You:. XOX". The handwriting is done with a marker, and the content is framed by decorative elements: small heart shapes in pink, red, and black scattered around the paper. The word "The Crazy Daisies" is printed in the bottom left corner of the image. A pencil is resting on the paper, partially erasing the already written '.XOX'. The overall setting and text suggest a romantic or affectionate context, likely intended as a love note.''',
    '''A book cover. It features a woman standing in an urban setting, likely in London, given the theme of the book. The woman is wearing a red top and a green skirt, and she is holding a pink shopping bag, suggesting a relatable, modern-day scenario. In the background, there are other people, including children, which adds to the everyday city life theme. The title of the book, "Made in London," is prominently displayed in a large, elegant, cursive font. The author's name, "CLARE LYDON," is placed above the title. The overall design is vibrant and eye-catching, with a mix of warm and cool tones to create a sense of contrast and appeal.''',
    '''A set of motorcycle parts, specifically a headlight fairing and its installation tools. The fairing is a black, curved plastic piece designed to cover the headlight of a motorcycle. It has a sleek, aerodynamic shape with a clear windshield element in the upper portion, which likely helps in reducing wind resistance at higher speeds. The fairing appears to have four round mounting points, two at the top and two near the bottom, for securing it to the motorcycle's frame. To the right of the fairing, there are various installation components laid out. These include: -Two black L-shaped brackets, which are likely used to attach the fairing to the motorcycle. -A Phillips or Torx screwdriver, which is necessary for assembling the fairing. -A set of screws and washers, all neatly arranged, which are used to fasten the fairing securely to the motorcycle's frame. The components are displayed against a plain white background, which highlights the individual parts and their arrangement, making it easier to identify each piece.''',
    '''A wooden box with a hinged lid that is open. The lid is propped up at a slight angle, revealing the interior of the box. The wood appears to have a weathered, possibly antique, finish with visible grain and some distressing, suggesting it may be old or used. The interior of the box is lined with a softer material, likely fabric or a similar substance, which is inward-facing and has a lighter color compared to the wood. The box itself has a rectangular shape with a smooth exterior. The edges are slightly worn, and there are no noticeable engravings, cliffs, or additional features on the exterior. The background is a flat, textured surface, possibly made of wood or a similar material, which is relatively neutral in color.''',
    '''A beige-colored leather organizer or wallet that is being held open by a person's hand. The organizer has multiple compartments and a zipped closure on the right side. The zippers are silver in color. The interior is lined with a texture that appears to be a combination of leather and a woven fabric, providing some form of insulation or padding. There are several large, open slots for cards and notes, and a smaller compartment at the bottom, likely intended for a phone or smaller items. The organizer is being held over a white tissue or paper bag with some text on it. The text "SHOPONLINE@RELUZZO.COM" is visible, indicating a branding or sales message. The background includes a wooden surface and some white fabric, possibly a cloth or a bag.''',
    '''A close-up of a red and black portable device, likely a vacuum cleaner, being used. Two hands are visible; one is holding the handle of the device, while the other is inserting a black, mesh-filtered attachment into the device. The attachment has a handle and a flexible hose that appears to be connected to the main body of the device, which has a tray or compartment at the bottom. The device looks compact and designed for ease of use, possibly for cleaning in tight spaces. The background is white, which helps in focusing attention on the device and the hands interacting with it.''',
    '''A compact electronic module with multiple wired connections attached via white connectors. The colored wires (red, black, yellow, and others) indicate power, signal, or data transmission. A metallic shield covers a key component, likely for RF or EMI protection. The board features multiple ports, allowing extensive wired interfacing. Two small push buttons and surface-mount components are visible. The design suggests it is a wired communication or control module, possibly for IoT, telemetry, or embedded systems, relying on physical connections for data and power transmission.''',
    '''The engine bay of a vintage car with a clean and well-maintained engine. The components include a large air cleaner housing, a green-painted engine block, and various hoses and wiring neatly arranged. A modern air conditioning compressor is visible, suggesting some upgrades. The black radiator and coolant reservoir are positioned at the front. The firewall and inner fenders are painted in red, matching the exterior. The setup reflects a classic design, likely from an older American vehicle, with a mix of original and updated mechanical parts.''',
    '''A 3D vector field visualization over a red, semi-transparent irregular polyhedral surface. The surface is constructed using triangular mesh elements, with black edges outlining its structure. Black arrows originate from various points on the surface, pointing outward, indicating a vector field, possibly representing normal vectors or force directions. A three-dimensional coordinate system with labeled X, Y, and Z axes is present, providing spatial orientation. The visualization suggests a scientific or engineering application, such as fluid dynamics, electromagnetism, or geometric modeling, analyzing directional properties on a complex 3D shape.''',
    '''An architectural visualization of a residential property in an aerial perspective. The main building and surrounding structures are represented in a simplified white model, while the landscape includes green grass, hedges, and trees with detailed foliage. A swimming pool is integrated into the backyard, surrounded by modern extensions and outdoor spaces. The background consists of a site plan with outlined property boundaries, roads, and neighboring structures. The rendering highlights the spatial arrangement and landscape design, illustrating the relationship between built and natural elements in an urban or suburban setting.''',
    '''A technical isometric drawing of an exploded mechanical or structural assembly. It consists of rectangular and square components, likely representing metal or wooden beams, plates, and fasteners. Several panels with circular cutouts are present, suggesting possible openings for wiring, ventilation, or mechanical components. The drawing uses a black background with white outlines, giving it a blueprint or CAD-rendered appearance. The perspective and arrangement suggest a modular design, possibly for a machine frame, electronic enclosure, or structural support system. The detailed layout highlights assembly steps and individual part connections.''',
    '''A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.''',
    '''An extreme close-up of an gray-haired man with a beard in his 60s, he is deep in thought pondering the history of the universe as he sits at a cafe in Paris, his eyes focus on people offscreen as they walk as he sits mostly motionless, he is dressed in a wool coat suit coat with a button-down shirt , he wears a brown beret and glasses and has a very professorial appearance, and the end he offers a subtle closed-mouth smile as if he found the answer to the mystery of life, the lighting is very cinematic with the golden light and the Parisian streets and city in the background, depth of field, cinematic 35mm film.''',
    '''Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.''',
    '''A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect.''',
    '''A white and orange tabby cat is seen happily darting through a dense garden, as if chasing something. Its eyes are wide and happy as it jogs forward, scanning the branches, flowers, and leaves as it walks. The path is narrow as it makes its way between all the plants. the scene is captured from a ground-level angle, following the cat closely, giving a low and intimate perspective. The image is cinematic with warm tones and a grainy texture. The scattered daylight between the leaves and plants above creates a warm contrast, accentuating the cat’s orange fur. The shot is clear and sharp, with a shallow depth of field.''',
    '''3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest.''',
    '''Tracking shot. Cinematic scene. A 19th century scuba diver runs down a busy street in New York City. The light is natural and warm, glinting off of the diver's suit. The diver's suit is burnished and old, held together with rusted bolts. The diver's helmet is round, with a black round glass porthole in the front. All around the diver, people walk down the street in period specific attire, such as large corset dresses with sweeping skirts, tailored suits, and top hats. The scene should feel joyful and amusing, heightening the thrill of the running diver.''',
    '''Camera tracking shot. A gigantic flying monster flies through midcentury new york city skyscrapers breathing and spewing fire from its open mouth. The light is overly-saturated and intense, making the monster glow with intensity. The monster darts through the sky, shooting enormous flames from its open mouth that engulf the entire scene. the flames are huge and are directed at buildings an the ground. The monster has the face of a dragon, the claws of an eagle, and huge leathery wings that are frayed and scarred. The footage should feel cinematic and premium, like an action movie. The scene should convey a fast-paced action and thrill.''',


    # "A female cosplayer dressed as a seductive bunny girl, featuring oversized, fluffy white ears and a sleek, form-fitting outfit in pastel pink and white. She has a playful expression with sparkling eyes and a mischievous smile. Her outfit includes a short skirt, a bow-tie, and thigh-high stockings. She holds a small, decorative carrot. The background is a whimsical carnival setting with colorful lights and balloons.",
    # "A female cosplayer embodying a popular anime character, wearing an intricately designed outfit with vibrant colors and detailed patterns. She has large, expressive eyes, long flowing hair, and a dynamic pose that captures the essence of the anime. Her outfit includes a custom-made dress with intricate embroidery and accessories like a wand or sword. The background is a futuristic cityscape with neon lights and holographic elements.",
    # "A female cosplayer dressed as a medieval knight, wearing a full suit of shining armor with intricate engravings. She has a strong, determined expression and stands in a heroic pose, holding a sword. Her outfit includes a helmet, chainmail, and a flowing cape. The background is a castle courtyard with stone walls, banners fluttering in the wind, and a sense of medieval grandeur.",
    # "A female cosplayer dressed as a powerful superheroine, wearing a sleek, high-tech costume with a bold color scheme and intricate designs. She has a confident, determined expression with a mask covering her eyes. Her outfit includes a cape, a utility belt, and various gadgets. The background is a city skyline at night with dramatic lighting highlighting her heroic presence.",
    # "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere.",
    # "A female cosplayer dressed in a steampunk-inspired outfit, featuring a corseted dress with brass gears and intricate metalwork. She has a daring expression with goggles on her head and a steampunk-themed weapon like a mechanical crossbow. The background is an industrial setting with steam pipes, gears, and a vintage airship.",
    # "A female cosplayer dressed as a classic Disney princess, wearing a flowing gown in a signature color like Cinderella's blue or Aurora's green. She has a graceful, elegant pose with a tiara on her head and a gentle smile. Her outfit includes a sparkling dress with intricate embroidery and a scepter. The background is a fairytale castle with lush gardens and a magical, enchanting atmosphere.",
    # "A female cosplayer dressed in a dark, gothic-inspired outfit with a corseted dress in black and deep red. She has a mysterious, brooding expression with dark makeup and a crown of thorns. Her outfit includes lace, velvet, and intricate jewelry. The background is a haunting, abandoned mansion with flickering candles and shadows.",
    # "A male character dressed in a high-tech cyberpunk outfit, featuring neon accents and glowing elements. His expression is rugged and intense, with short, spiky hair and cybernetic enhancements such as glowing eyes or robotic limbs. His attire includes a leather trench coat, a futuristic mask, and various gadgets like a wrist-mounted laser or a cybernetic arm. The background is a neon-lit futuristic cityscape with towering skyscrapers, holographic advertisements, and an atmosphere of gritty urban energy.",
    # "A male warrior in a full suit of medieval armor, intricately engraved with battle scars and signs of wear, stands with a stern and determined expression, exuding bravery. He holds a battle-axe firmly, his posture powerful and commanding. His armor includes a visored helmet, chainmail, and a tattered, flowing cape. Behind him lies a medieval castle courtyard, surrounded by ancient stone walls and fluttering banners, evoking a sense of timeless grandeur.",
    # "A male superhero in a sleek, high-tech costume with bold colors and intricate patterns, exuding confidence and heroism. His eyes are hidden behind a mask, and he wears a flowing cape, a utility belt, and advanced gadgets like a wrist-mounted device and a high-tech helmet. The background is a city skyline at night, illuminated by dramatic lighting that highlights his heroic stance, with distant explosions hinting at an ongoing battle.",
    # "A male character stands in a sleek, high-tech suit of armor that blends traditional African motifs with advanced design, featuring dark, metallic material adorned with intricate patterns and glowing energy lines. His expression is stoic and determined, with sharp eyes conveying strength and wisdom. Claw-like gauntlets made from durable metal and a utility belt equipped with glowing symbols complete his attire. The background is a futuristic cityscape combining modern skyscrapers and traditional African architecture, with flying vehicles and a massive glowing structure symbolizing technological prowess. Set during twilight, the scene captures the character on a high vantage point, surveying his domain with pride and responsibility.",

    # "In a modern urban café, four young people are sitting around a wooden round table. Sunlight streams through large windows, casting a warm glow over them. On the table are freshly brewed lattes and a piece of chocolate cake. The boy on the left, dressed in a casual suit, is smiling and showing photos on his phone. The girl in the center, with wavy long hair and retro glasses, is listening attentively. The boy on the right, wearing a hoodie, is leaning on his hand with a relaxed expression. The last girl, in a dress, is holding an open book and smiling as she joins the conversation. In the background, you can see the minimalist café decor with art paintings on the walls, creating a cozy and inviting atmosphere.",
    # "In an elegant Jiangnan garden, five literati are gathered in a pavilion surrounded by bamboo groves and rockeries, with lotus flowers blooming in the pond. The central figure is a middle-aged scholar in traditional robes, holding a brush and writing on rice paper. To his left, a scholar in a blue robe holds a folding fan and is conversing with another in a long robe. On the right, two young scholars are engaged in different activities—one is admiring a painting while the other plays the guzheng, filling the air with melodious music. Behind them, a screen painted with landscapes adds to the classical elegance of the scene.",
    # "On a sunny beach, six friends are enjoying a lively party. Colorful umbrellas and beach chairs are scattered around, with the azure sea and white sailboats in the distance. At the front, a boy in a surfboard shorts and sunglasses is tossing a beach ball into the air. Next to him, a girl in a bikini and flower crown is lounging on a beach chair, soaking up the sun. In the center, three friends are building a sandcastle—one girl in a one-piece swimsuit is focused on sculpting with a small shovel. The last boy, in shorts and a T-shirt, is returning from surfing, holding his surfboard and grinning with excitement. The scene is filled with joy and energy.",
    # "In a grand medieval castle banquet hall, a lavish ball is in full swing. The hall is brightly lit with chandeliers hanging from the ceiling and tapestries adorning the walls. In the center of the dance floor, a couple in elegant attire is dancing gracefully—a man in a black tailcoat and a woman in a red gown whose skirt swirls with every step. Around them, other guests in noble and knightly attire are either chatting or watching the dancers. A band in the corner plays classical music, adding to the romantic and opulent atmosphere of the scene.",
    # "In a futuristic sci-fi city street, a team of five is on a mission. The street is filled with neon lights and hovering vehicles, with towering skyscrapers that exude a sense of the future. The team members are dressed in high-tech combat suits with transparent displays on their helmets. The leader, a woman, is holding an energy weapon and scanning the area ahead. Behind her are two men—one carrying a high-tech backpack and the other holding a shield. In the center is a robot, glowing blue as it analyzes the surroundings. The last member is a covert agent in an invisibility cloak, preparing to infiltrate a target area. The scene is tense and filled with a sense of advanced technology."
]


# cfg_scale = [4.0] * len(prompt_list)
# cfg_scale = [6.0] * len(prompt_list)
cfg_scale = [8.0] * len(prompt_list)
cfg_interval = (0, 1.0)
timestep_shift = 3.0
seed = 42
num_timesteps = 50
# num_timesteps = 250

for index in tqdm(range(len(prompt_list))):
    tmpimage = generate_image(
        prompt=prompt_list[index],
        cfg_scale=cfg_scale[index], 
        cfg_interval=cfg_interval, 
        timestep_shift=timestep_shift, 
        seed=seed, 
        num_timesteps=num_timesteps
    )

    tmpimage = tmpimage.crop(tmpimage.getbbox())
    output_path = os.path.join(args.output_dir, f"{index}.png")
    tmpimage.save(output_path)
    with open(os.path.join(args.output_dir, "prompt.txt"), "a") as f:
        f.writelines(prompt_list[index] + "\n")
