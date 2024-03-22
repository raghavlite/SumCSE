"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template, add_model_args
import pandas as pd
import time
from transformers import set_seed


### PPFS0
pos0_template = (
#Summary
"""Summarize the input sentence in seven words.
 The input sentence is - {}
 What is your generated sentence?""",
" The output sentence is:")
pos0_template_examples=[]

#### PPFS
pos1_template =(
""" Paraphrase the input sentence or phrase, providing an alternative expression with the same meaning. Please do not provide any alternative or reasoning or explanation.
ASSISTANT: Alright.
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
""", 
"""""")

pos1_template_examples=[
("A woman and the kid are walking along with the dog.", "A woman, a child, and a dog are walking together."),
("Girl holding a box of crayons, with a notebook in front of her.", "A girl is holding a box of crayons and there is a notebook in front of her."),
("The man is pedaling the tiny bike down the steps.", "The man is riding a small bike down the stairs."),
("A group of people cleaning a beach.", "Several individuals are cleaning the beach."),
("Most of them seemed to be dead or unconscious.", "Majority of them appeared lifeless or unconsciousness."),
("An older gentleman wearing a dark coat and hat leaning over a green object.", "A man, who appears to be elderly, is leaning over a green object while dressed in a dark coat and hat."),
("a man wearing white cleans dirt from the ground with a hose.", "A man in white clothing uses a hose to clean dirt off the ground."),
("An older woman with a nose ring, silver necklace, and decorative scarf is quietly sitting.", "A woman with a nose ring, silver necklace, and decorative scarf is sitting quietly."),
("Don't say it!", "Do not utter it!"),
("Woman making sushi from home.", "A woman is creating sushi in her own home."),
]



#### PPFS2
pos2_template =(
""" Rewrite the input sentence or phrase using different sentence structure and different words while preserving its original meaning. Please do not provide any alternative or reasoning or explanation.
ASSISTANT: Alright.
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
""", 
"""""")


pos2_template_examples=[
("A person is hanging upside down from power lines.", "Someone is hanging from power lines, but upside down."),
("a woman points with her left hand wearing navy blue clothes as a man wearing a visor backwards stands looking at her.", "A man wearing a visor backwards is standing and looking at a woman wearing navy blue clothes who is pointing with her left hand."),
("A young child is crying in the hands of an elderly male.", "An elderly male is holding a young child who is crying."),
("A little girl playing in a water jet fountain, with multiple children playing in the background.", "In the background, multiple children are playing while a little girl is playing in a water jet fountain."),
("A motorcycle rider leans in during a turn", "During a turn, the motorcycle rider leans in."),
("The data reliability process begins with two relatively simple steps.", "Two simple steps start data reliability process."),
("A woman is walking a blue bike across a road.", "Across the street, a lady is pushing a blue bicycle."),
("An older woman is sitting outside drawing with a brick building behind her.", "A brick building can be seen behind an older woman who is sitting outside and drawing."),
("You're right, said Tommy slowly.", "Tommy said slowly that you were right."),
("People are walking through an outdoor market.", "An outdoor market is bustling with people walking through it."),
]




#### PPFS3
pos3_template =(
""" Create a sentence or phrase that is also true, assuming the provided input sentence or phrase is true. Please do not provide any alternative or reasoning or explanation.
ASSISTANT: Alright.
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
""", 
"""""")


pos3_template_examples=[
("An older gentleman wearing a dark coat and hat leaning over a green object.", "An older man observing an object."),
("a man wearing white cleans dirt from the ground with a hose.", "A man is clearing dirt with a hose."),
("One baseball player is on the ground with his mouth open while another jumps above him.", "A baseball player with mouth agape is under another who is jumping."),
("Several people are posing for a photo with the naked cowboy in NYC.", "People are taking pictures with an unclothed individual."),
("A middle-aged woman is getting her hair done in a barber shop with polka-dotted walls.", "She was having her hair done."),
("A brown dog running on the beach near the ocean.", "The dog is running outside."),
("Woman making sushi from home.", "There were some people making homemade sushi."),
("An Asian skateboarder in a black shirt and fitted jeans shows off a trick.", "A skateboarder in black shows off a trick."),
("A woman and the kid are walking along with the dog.","The people are walking."),
("A shirtless child plays with an adult indoors.","The people are inside.")
]





#### PPFS4
pos4_template =(
""" Provide a concise paraphrase of the input sentence or phrase, maintaining the core meaning while altering the words and sentence structure. Feel free to omit some of the non-essential details like adjectives or adverbs. Please do not provide any alternative or reasoning or explanation.
ASSISTANT: Alright.
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
""", 
"""""")


pos4_template_examples=[
("The data reliability process begins with two relatively simple steps.", "Two simple steps start data reliability process."),
("Bald white men stands in a shop with many shirts.", "A white bald man is in a shirt store."),
("People hang upside down as a roller coaster executes a spiral loop-the-loop.", "People are on a roller coaster."),
("I heard nothing. Mrs. Vandemeyer gazed round her fearfully.", "Mrs. Vandemeyer looked around fearfully as I heard nothing."),
("Dave paid no attention to where his feet were leading him, only vaguely aware that he was heading down a gully below the current construction job.", "Dave walked absentmindedly down a gully."),
("and that's been uh uh a problem you know to to the merchant people that that fish and stuff up here they run into that ice stuff in the winter and it breaks away", "The ice breaks away in the winter when merchants fish."),
("These changes provide EPA with approximate targets so that each of the scenarios can be mapped into the AMIGA model.", "EPA has created approximate targets to map each scenario into the AMIGA model."),
("An audience of senior citizens is sitting for some kind of presentation.", "A presentation is being given to an audience consisting of senior citizens."),
("A group of people are discussing what they need to buy from Walmart", "A Walmart shopping list is being discussed by a group of people."),
("A woman is on her hands and knees cleaning a rug.", "A lady is scrubbing a rug on her knees."),
]





#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################





#### NEGFS1
neg1_template =(
""" Revise the provided sentence by swapping, changing, or contradicting some details in order to express a different meaning, while maintaining the general context and structure.
ASSISTANT: Alright.
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
""", 
"""""")


# NEGFS1 and NEGFS2 have the same demonstrations(changes in actions, items, entities)
neg1_template_examples=[
("A woman is walking a blue bike across a road.", "A blue bike is being carried by a woman across a road."),
("A motorcycle rider leans in during a turn.", "A motorcycle rider leans out during a turn."),
("A little girl playing in a water jet fountain, with multiple children playing in the background.", "A little girl playing in a water jet fountain, with several children sitting in the background."),
("While everybody else is having fun on a roller coaster ride, one lady at the very front seat seems too scared.", "While everybody else is thrilled on a roller coaster ride, one lady at the very front seat looks bored."),
("A soccer team celebrates.", "The soccer team sulks in defeat."),
("Two children sit side by side while eating a treat.", "Two children stand facing each other while eating a treat."),
("An older woman with a nose ring, silver necklace, and decorative scarf is quietly sitting.", "A young woman with multiple piercings, tattoos, and bright hair dye is dancing loudly to the music."),
("An Asian skateboarder in a black shirt and fitted jeans shows off a trick.", "A European skateboarder in a red shirt and loose jeans displays a stunt."),
("A group of girls who are gothically dressed standing together", "A group of boys who are elegantly dressed standing together."),
("Many Asian people waiting at a train stop.", "Few African people waiting at a train stop."),
]





#### NEGFS2
neg2_template =(
""" Generate a slightly modified version of the provided sentence to express an opposing or alternate meaning by changing one or two specific elements, while maintaining the overall context and sentence structure.
ASSISTANT: Alright.
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
""", 
"""""")

# NEGFS1 and NEGFS2 have the same demonstrations(changes in actions, items, entities)

neg2_template_examples=[
("A motorcycle rider leans in during a turn.", "A motorcycle rider leans out during a turn."),
("A woman is walking a blue bike across a road.", "A blue bike is being carried by a woman across a road."),
("A little girl playing in a water jet fountain, with multiple children playing in the background.", "A little girl playing in a water jet fountain, with several children sitting in the background."),
("While everybody else is having fun on a roller coaster ride, one lady at the very front seat seems too scared.", "While everybody else is thrilled on a roller coaster ride, one lady at the very front seat looks bored."),
("A soccer team celebrates.", "The soccer team sulks in defeat."),
("Two children sit side by side while eating a treat.", "Two children stand facing each other while eating a treat."),
("An older woman with a nose ring, silver necklace, and decorative scarf is quietly sitting.", "A young woman with multiple piercings, tattoos, and bright hair dye is dancing loudly to the music."),
("An Asian skateboarder in a black shirt and fitted jeans shows off a trick.", "A European skateboarder in a red shirt and loose jeans displays a stunt."),
("A group of girls who are gothically dressed standing together", "A group of boys who are elegantly dressed standing together."),
("Many Asian people waiting at a train stop.", "Few African people waiting at a train stop."),
]




#### NEGFS3
neg3_template =(
""" Transform the input sentence by adjusting, altering, or contradicting its original meaning to create a logical and sensible output sentence with a different meaning from the input sentence.
ASSISTANT: Alright.
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
""", 
"""""")

# opposite(opposite) meaning examples
neg3_template_examples=[
("a woman points with her left hand wearing navy blue clothes as a man wearing a visor backwards stands looking at her.", "The man wearing a visor backwards points with his right hand wearing red clothes as a woman wearing a hat stands looking at him."),
("they're on weekdays at eleven o'clock everyday Monday through Friday", "They're on weekends at eleven o'clock everyday Saturday and Sunday."),
("Most of them seemed to be dead or unconscious.", "Few of them appeared to be lifeless and unconsciousness."),
("People hang upside down as a roller coaster executes a spiral loop-the-loop.", "People remain upright as a roller coaster maneuvers a straight path."),
("Cybercitizens will be hard-wired, and entertainment as we know it will be in the form of IPOs.", "Humans will continue to enjoy entertainment in traditional forms such as movies and TV shows, while technology continues to advance."),
("A street vendor sitting on a stool, on the sidewalk with his wares, reading his newspaper.", "A street vendor standing with his wares on the sidewalk, yelling to draw attention to his goods."),
("Eva was put under . . . pressure to purchase the car immediately.", "Eva was given the option to take her time in considering the purchase of the car."),
("i got sort of hooked on that uh Dark Shadows", "I never really got into the Dark Shadows TV series, despite its popularity."),
("A group of people cleaning a beach.", "A gathering of individuals littering on the beach."),
("The data reliability process begins with two relatively simple steps.", "A complex and lengthy series of steps begin the data reliability process."),
]





#### NEGFS4
neg4_template =(
""" Generate a sentence that conveys a altering, contrasting or opposite idea to the given input sentence, while ensuring the new sentence is logical, realistic, and grounded in common sense.
ASSISTANT: Alright.
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
ASSISTANT: {}
USER: {}
""", 
"""""")

# contrasting(different) meaning examples
neg4_template_examples=[

("right it's it's not a problem either way like i said my only complaint is where they mixed them on the same car", "My complaints are too many to count."),
("a man wearing white cleans dirt from the ground with a hose.", "A man spills oil on the ground with a hose while wearing black."),
("One baseball player is on the ground with his mouth open while another jumps above him.", "The baseball players are calmly standing on the field, waiting for the game to start."),
("An egret stretches his neck to eat things out of the still water.", "The egret flies away, ignoring the murky pond below."),
("People are walking through an outdoor market.", "People are running through an indoor shopping mall."),
("A man trying to go home on his bicycle after long day of entertainment in the city suburbs.", "A man cycling to the suburbs after a long day of work in the city center."),
("Two men putting a cover over a makeshift boat with a couch on it.", "Two men were busy dismantling the makeshift boat with a couch on it."),
("A woman in park with a more bag.", "A woman is in a park with a big blue box."),
("Two white girls with brown hair are standing in front of a van.", "The van is parked in the garage next to the brown-haired driver's SUV."),
("A hiker standing high on a bluff overlooking the mountains below.", "The hiker was afraid of heights and refused to climb the bluff.")
]