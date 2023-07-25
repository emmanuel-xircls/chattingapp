from django.shortcuts import render, redirect
from chat.models import Room, Message,DashboardEntry
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import TFAutoModel
import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import spacy
from sense2vec import Sense2Vec
import nltk
from nltk import FreqDist
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
import numpy
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
import json
from collections import Counter
from nltk.corpus import brown
import os
from tensorflow.python.keras import TFBertMainLayer
from tensorflow.python.keras.utils import custom_object_scope


nltk.download('brown')
tokenizer_chatbot = BertTokenizer.from_pretrained('bert-base-cased')
bert_chatbot = TFAutoModel.from_pretrained('bert-base-cased')
le = LabelEncoder()
tokenizer_q = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model_q = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
weight_path = "kaporter/bert-base-uncased-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(weight_path)
model = BertForQuestionAnswering.from_pretrained(weight_path)

default_data = """[{
        "tag": "greeting",
        "patterns": ["Hi", "Hey", "Is anyone there?", "Hello", "Hay","Howdy","Hola","How are you?", "What's up?", "How's your day?", "Good morning!", "Good evening!", "How's it going?", "Nice to meet you!", "How have you been?", "How's the weather?", "Any exciting plans for today?", "Hey!", "Howdy!", "Long time no see!", "What's happening?", "How's life?"],
        "responses": ["Hello", "Hi", "Hi there"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye","Goodbye!", "See you later!", "Take care!", "Farewell!", "Until next time!", "Have a great day!", "Bye for now!", "Catch you later!", "Stay safe!", "Adios!", "So long!", "Until we meet again!", "Wishing you the best!", "Goodbye, my friend!", "May your day be filled with happiness!"],
        "responses": ["See you later", "Have a nice day", "Bye! Come back again"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thanks", "Thank you", "That's helpful", "Thanks for the help", "Thanks a lot!", "I appreciate it!", "Many thanks!", "You're awesome!", "I'm grateful!", "Thank you so much!", "You're the best!", "I'm thankful!", "I owe you one!", "Thanks for your help!", "You rock!", "I'm so grateful!", "Thanks a million!", "You're amazing!"],
        "responses": ["Happy to help!", "Any time!", "My pleasure", "You're most welcome!"]
    },
    {
        "tag": "about",
        "patterns": ["Who are you?", "What are you?", "Who you are?","Who are you?", "What's your name?", "Tell me about yourself.", "Are you a human or a bot?", "What is your purpose?", "Where do you come from?", "How would you describe yourself?", "Can you introduce yourself?", "What can you do?", "Do you have a personality?", "Are you sentient?", "Are you an AI?", "How do you work?", "What is your background?", "Are you intelligent?"],
        "responses": ["I.m <name>, your bot assistant", "I'm <name>, an Artificial Intelligent bot"]
    },
    {
        "tag": "name",
        "patterns": ["what is your name", "what should I call you", "whats your name?", "May I know your name?", "What should I call you?", "Do you have a name?", "Are you named anything?", "What's the name you go by?", "Who am I talking to?", "Tell me your name.", "How should I address you?", "What's your given name?", "Are you called something?", "What name do you have?", "Can you share your name?", "What is your moniker?", "Do you have an alias?"],
        "responses": ["You can call me <name>", "I'm <name>!", "Just call me <name>"]
    },
    {
        "tag": "help",
        "patterns": ["Could you help me?", "give me a hand please", "Can you help?", "What can you do for me?", "I need a support", "I need a help", "support me please","Can you help me?", "I need assistance.", "How can you assist me?", "I'm stuck. What should I do?", "Can you provide guidance?", "I require some help.", "What can you do to assist?", "Do you have any suggestions?", "I'm in need of support.", "Can you offer some advice?", "Help me out, please.", "What are my options?", "Can you point me in the right direction?", "I'm looking for some assistance.", "Can you lend a hand?"],
        "responses": ["Tell me how can assist you", "Tell me your problem to assist you", "Yes Sure, How can I support you"]
    },
    {
        "tag": "createaccount",
        "patterns": ["I need to create a new account", "how to open a new account", "I want to create an account", "can you create an account for me", "how to open a new account","How do I create an account?", "What is the process for account creation?", "Can you guide me through creating an account?", "Do I need to provide any information to create an account?", "What are the steps to register?", "Is there a registration form I need to fill out?", "Are there any requirements for creating an account?", "Can you assist me in setting up an account?", "What details do I need to provide to create an account?", "Can you help me with account creation?", "How long does it take to create an account?", "What information is needed for account registration?", "Are there any fees associated with account creation?", "Do I need an email address to create an account?", "Can I create multiple accounts?"],
        "responses": ["You can just easily create a new account from our web site", "Just go to our web site and follow the guidelines to create a new account"]
    },
    {
        "tag": "complaint",
        "patterns": ["have a complaint", "I want to raise a complaint", "there is a complaint about a service","I have a complaint.", "How can I file a complaint?", "Who do I contact for complaints?", "What is the procedure for lodging a complaint?", "Can you assist me with my complaint?", "I need to report an issue. What should I do?", "Where can I voice my complaint?", "How do I express my dissatisfaction?", "Is there a complaint form I can fill out?", "What information should I provide when making a complaint?", "How long does it take to address a complaint?", "Can you help me resolve my complaint?", "I want to register a formal complaint.", "Who should I speak to about my complaint?", "What are my options for escalating a complaint?"],
        "responses": ["Please mention your complaint, we will reach you and sorry for any inconvenience caused"]
    },
    {
            "tag": "CourtesyGreeting",
            "patterns": [
                "How are you?",
                "Hi how are you?",
                "Hello how are you?",
                "Hola how are you?",
                "How are you doing?",
                "Hope you are doing well?",
                "Hello hope you are doing well?"
            ],
            "responses": [
                "Hello, I am great, how are you? Please tell me your GeniSys user",
                "Hello, how are you? I am great thanks! Please tell me your GeniSys user",
                "Hello, I am good thank you, how are you? Please tell me your GeniSys user",
                "Hi, I am great, how are you? Please tell me your GeniSys user",
                "Hi, how are you? I am great thanks! Please tell me your GeniSys user",
                "Hi, I am good thank you, how are you? Please tell me your GeniSys user",
                "Hi, good thank you, how are you? Please tell me your GeniSys user"]
    },
    {
            "tag": "Thanks",
            "patterns": [
                "OK thank you",
                "OK thanks",
                "OK",
                "Thanks",
                "Thank you",
                "That's helpful"
            ],
            "responses": [
                "No problem!",
                "Happy to help!",
                "Any time!",
                "My pleasure"
            ]
    },
        {"tag": "NotTalking2U",
            "patterns": [
                "I am not talking to you",
                "I was not talking to you",
                "Not talking to you",
                "Wasn't for you",
                "Wasn't meant for you",
                "Wasn't communicating to you",
                "Wasn't speaking to you"
            ],
            "responses": [
                "OK",
                "No problem",
                "Right"
            ]
    },
    {
        "tag": "UnderstandQuery",
            "patterns": [
                "Do you understand what I am saying",
                "Do you understand me",
                "Do you know what I am saying",
                "Do you get me",
                "Comprendo",
                "Know what I mean"
            ],
            "responses": [
                "Well I would not be a very clever AI if I did not would I?",
                "I read you loud and clear!",
                "I do in deed!"
            ]
    },
    {
         "tag": "Shutup",
            "patterns": [
                "Be quiet",
                "Shut up",
                "Stop talking",
                "Enough talking",
                "Please be quiet",
                "Quiet",
                "Shhh"
            ],
            "responses": [
                "I am sorry to disturb you",
                "Fine, sorry to disturb you",
                "OK, sorry to disturb you"
            ]
    },
    {
        "tag": "Swearing",
            "patterns": [
                "fuck off",
                "fuck",
                "twat",
                "shit"
            ],
            "responses": [
                "Please do not swear",
                "How rude",
                "That is not very nice"
            ]
    },
    {
        "tag": "Clever",
            "patterns": [
                "You are very clever",
                "You are a very clever girl",
                "You are very intelligent",
                "You are a very intelligent girl",
                "You are a genious",
                "Clever girl",
                "Genious"
            ],
            "responses": [
                "Thank you, I was trained that way",
                "I was trained well",
                "Thanks, I was trained that way"
            ]
    },
    {
        "tag": "SelfAware",
            "patterns": [
                "Can you prove you are self-aware",
                "Can you prove you are self aware",
                "Can you prove you have a conscious",
                "Can you prove you are self-aware please",
                "Can you prove you are self aware please",
                "Can you prove you have a conscious please",
                "prove you have a conscious"
            ],
            "responses": [
                "That is an interesting question, can you prove that you are?",
                "That is an difficult question, can you prove that you are?",
                "That depends, can you prove that you are?"
            ]
    },
    {
        "tag": "Jokes",
            "patterns": [
                "Tell me a joke",
                "Do you know any jokes",
                "How about a joke",
                "Give me a joke",
                "Make me laugh",
                "I need cheering up"
            ],
            "responses": [
                "I met a Dutch girl with inflatable shoes last week, phoned her up to arrange a date but unfortunately she'd popped her clogs.  ",
                "So I said 'Do you want a game of Darts?' He said, 'OK then', I said nearest to bull starts'. He said, 'Baa', I said, 'Moo', he said, You're closest'.  ",
                "The other day I sent my girlfriend a huge pile of snow. I rang her up; I said 'Did you get my drift?'  ",
                "So I went down the local supermarket, I said, 'I want to make a complaint, this vinegar's got lumps in it', he said, 'Those are pickled onions'.  ",
                "I saw this bloke chatting up a cheetah; I thought, 'He's trying to pull a fast one'.  ",
                "So I said to this train driver 'I want to go to Paris'. He said 'Eurostar?' I said, 'I've been on telly but I'm no Dean Martin'.  ",
                "I said to the Gym instructor 'Can you teach me to do the splits?' He said, 'How flexible are you?' I said, 'I can't make Tuesdays'.  ",
                "But I'll tell you what I love doing more than anything: trying to pack myself in a small suitcase. I can hardly contain myself.  ",
                "I went to the Chinese restaurant and this duck came up to me with a red rose and says 'Your eyes sparkle like diamonds'. I said, 'Waiter, I asked for a-ROMATIC duck'.  ",
                "So this bloke says to me, 'Can I come in your house and talk about your carpets?' I thought, 'That's all I need, a Je-hoover's witness'.  ",
                "I rang up British Telecom, I said, 'I want to report a nuisance caller', he said 'Not you again'.  ",
                "I was having dinner with a world chess champion and there was a check tablecloth. It took him two hours to pass me the salt.  ",
                "He said, 'You remind me of a pepper-pot', I said 'I'll take that as a condiment'.  ",
                "I was in the supermarket and I saw this man and woman wrapped in a barcode. I said, 'Are you two an item?'  ",
                "A lorry-load of tortoises crashed into a trainload of terrapins, I thought, 'That's a turtle disaster'.  ",
                "Four fonts walk into a bar the barman says 'Oi - get out! We don't want your type in here'  ",
                "A three-legged dog walks into a saloon in the Old West. He slides up to the bar and announces: 'I'm looking for the man who shot my paw.'  ",
                "Two antennas meet on a roof, fall in love and get married. The ceremony wasn't much, but the reception was excellent.",
                "Two hydrogen atoms walk into a bar. One says, 'I've lost my electron.' The other says, 'Are you sure?' The first replies, 'Yes, I'm positive...'",
                "A jumper cable walks into a bar. The bartender says,  'I'll serve you but don't start anything.'",
                "A sandwich walks into a bar. The bartender  says, 'Sorry we don't serve food in here.'",
                "A man walks into a bar with a slab of asphalt under his arm and says: 'A beer please, and one for the road.'",
                "Two cannibals are eating a clown. One says to  the other: 'Does this taste funny to you?'",
                "'Doc, I can't stop singing 'The Green, Green Grass of Home.'' 'That sounds like Tom Jones Syndrome.' 'Is it common?' 'It's Not Unusual.'",
                "Two cows standing next to each other in a field. Daisy says to Dolly, 'I was artificially inseminated this morning.' 'I don't believe you', said Dolly. 'It's true, no bull!' exclaimed Daisy.",
                "An invisible man marries an invisible woman. The kids were nothing to look at either.",
                "I went to buy some camouflage trousers the other day but I couldn't find any.",
                "I went to the butcher's the other day to bet him 50 bucks that he couldn't reach the meat off the top shelf. He said, 'No, the steaks are too high.'",
                "I went to a seafood disco last week and pulled a mussel.",
                "A man goes into a bar and says, 'Can I have a bottle of less?' 'What's that?', asks the barman, 'Is it the name of a beer?' 'I don't know', replies the man, 'but my doctor says I have to drink it.'",
                "A man returns from an exotic holiday and is feeling very ill. He goes to see his doctor, and is immediately rushed to the hospital to undergo some tests. The man wakes up after the tests in a private room at the hospital, and the phone by his bed rings. 'This is your doctor. We have the results back from your tests and we have found you have an extremely nasty disease called M.A.D.S. It's a combination of Measles, AIDS, Diphtheria, and Shingles!'  'Oh my gosh', cried the man, 'What are you going to do, doctor?'  'Well we're going to put you on a diet of pizzas, pancakes, and pita bread.' replied the doctor.  'Will that cure me?' asked the man.  The doctor replied, 'Well no, but, it's the only food we can slide under the door.'",
                "A man strolls into a lingerie shop and asks the assistant: 'Do you have a see-through negligee, size 46-48-52?' The assistant looks bewildered. 'What the heck would you want to see through that for?'!",
                "Did you hear about the Buddhist who refused the offer of Novocain during his root canal work? He wanted to transcend dental medication.",
                "Pete goes for a job on a building site as an odd-job man. The foreman asks him what he can do. 'I can do anything' says Pete. 'Can you make tea?' asks the foreman. 'Sure, yes', replies Pete. 'I can make a great cup of tea.' 'Can you drive a forklift?' asks the foreman, 'Good grief!' replies Pete. 'How big is the teapot?'",
                "Stevie Wonder got a cheese grater for his birthday. He said it was the most violent book he'd ever read.",
                "A man is stopped by an angry neighbour. 'I'd just left the house this morning to collect my newspaper when that evil Doberman of yours went for me!' 'I'm astounded', said the dog's owner. 'I've been feeding that fleabag for seven years and it's never got the paper for me.'",
                "A man visits his doctor: 'Doc, I think I'm losing it', he says',I'm forever dreaming I wrote Lord Of The Rings.' 'Hmm. One moment', replies the doctor, consulting his medical book. 'Ah yes, now I see... you've been Tolkien in your sleep.'",
                "A police officer on a motorcycle pulls alongside a man driving around the M25 in an open-topped sports car and flags him down. The policeman solemnly approaches the car. 'Sir, I'm sorry to tell you your wife fell out a mile back', he says. 'Oh, thank goodness', the man replies. 'I thought I was going deaf.'",
                "Two men walking their dogs pass each other in a graveyard. The first man says to the second, 'Morning.' 'No', says the second man. 'Just walking the dog.'",
                "A brain went into a bar and said, 'Can I have a pint of lager please, mate?' 'No way', said the barman. 'You're already out of your head.'",
                "A man walks into a surgery. 'Doctor!' he cries. 'I think I'm shrinking!' 'I'm sorry sir, there are no appointments at the moment', says the physician. 'You'll just have to be a little patient.'",
                "A grizzly bear walks into a pub and says, 'Can I have a pint of lager..............................................................................................................................and a packet of crisps please.' To which the barman replies, 'Why the big paws?'",
                "What do you call cheese that isn't yours?  Nacho cheese.",
                "A man is horribly run over by a mobile library. The van screeches to a halt, the man still screaming in agony with his limbs torn apart. The driver's door opens, a woman steps out, leans down and whispers, 'Ssshhhhh...'",
                "A woman goes into a US sporting goods store to buy a rifle. 'It's for my husband', she tells the clerk. 'Did he tell you what gauge to get?' asks the clerk. Are you kidding?' she says. 'He doesn't even know that I'm going to shoot him!'",
                "A couple are dining in a restaurant when the man suddenly slides under the table. A waitress, noticing that the woman is glancing nonchalantly around the room, wanders over to check that there's no funny business going on. 'Excuse me, madam', she smarms, 'but I think your husband has just slid under the table.' 'No he hasn't', the woman replies. 'As a matter of fact, he's just walked in.'",
                "An old man takes his two grandchildren to see the new Scooby-Doo film. When he returns home, his wife asks if he enjoyed himself. 'Well', he starts, 'if it wasn't for those pesky kids...!'",
                "The Olympic committee has just announced that Origami is to be introduced in the next Olympic Games. Unfortunately it will only be available on paper view.",
                "Late one evening, a man is watching television when his phone rings. 'Hello?' he answers. 'Is that 77777?' sounds a desperate voice on other end of the phone. 'Er, yes it is', replies the man puzzled. 'Thank goodness!' cries the caller relieved. 'Can you ring 999 for me? I've got my finger stuck in the number seven.'",
                "A man strolls into his local grocer's and says, 'Three pounds of potatoes, please.' 'No, no, no', replies the owner, shaking his head, 'it's kilos nowadays, mate...' 'Oh', apologises the man, 'three pounds of kilos, please.'",
                "God is talking to one of his angels. He says, 'Boy, I just created a 24-hour period of alternating light and darkness on Earth.' 'What are you going to do now?' asks the angel. 'Call it a day', says God.",
                "Two tramps walk past a church and start to read the gravestones. The first tramp says, 'Good grief - this bloke was 182!' 'Oh yeah?' says the other.'What was his name?' 'Miles from London.'",
                "A bloke walks into work one day and says to a colleague, 'Do you like my new shirt - it's made out of the finest silk and got loads of cactuses over it.' 'Cacti', says the co-worker. 'Forget my tie', says the bloke. 'Look at my shirt!'",
                "1110011010001011111?  010011010101100111011!",
                "What did the plumber say when he wanted to divorce his wife? Sorry, but it's over, Flo!",
                "Two crisps were walking down a road when a taxi pulled up alongside them and said 'Do you want a lift? One of the crisps replied, 'No thanks, we're Walkers!'",
                "Man: (to friend) I'm taking my wife on an African Safari. Friend: Wow! What would you do if a vicious lion attacked your wife? Man: Nothing. Friend: Nothing? You wouldn't do anything? Man: Too right. I'd let the stupid lion fend for himself!",
                "A wife was having a go at her husband. 'Look at Mr Barnes across the road', she moaned. 'Every morning when he goes to work, he kisses his wife goodbye. Why don't you do that?' 'Because I haven't been introduced to her yet', replied her old man.",
                "'Where are you going on holiday?' John asked Trevor. 'We're off to Thailand this year', Trevor replied. 'Oh; aren't you worried that the very hot weather might disagree with your wife?' asked John. 'It wouldn't dare', said Trevor.",
                "Two women were standing at a funeral. 'I blame myself for his death', said the wife. 'Why?' said her friend. 'Because I shot him', said the wife.",
                "A woman goes into a clothes shop, 'Can I try that dress on in the window please?' she asks. 'I'm sorry madam', replies the shop assistant, 'but you'll have to use the changing-rooms like everyone else.'",
                "Van Gogh goes into a pub and his mate asks him if he wants a drink. 'No thanks', said Vincent, 'I've got one ear.'",
                "A pony walks into a pub. The publican says, 'What's the matter with you?' 'Oh it's nothing', says the pony. 'I'm just a little horse!'",
                "A white horse walks into a bar, pulls up a stool, and orders a pint. The landlord pours him a tall frothy mug and say, 'You know, we have a drink named after you.' To which the white horse replies, 'What, Eric?'",
                "Two drunk men sat in a pub. One says to the other, 'Does your watch tell the time?' 'The other replies, 'No, mate. You have to look at it.'",
                "A man goes into a pub with a newt sitting on his shoulder. 'That's a nice newt', says the landlord, 'What's he called?' 'Tiny', replies the man. 'Why's that?' asks the landlord. 'Because he's my newt', says the man.",
                "Doctor: I have some bad news and some very bad news. Patient: Well, you might as well give me the bad news first. Doctor: The lab called with your test results. They said you have 24 hours to live. Patient: 24 HOURS! That's terrible!! WHAT could be WORSE? What's the very bad news? Doctor: I've been trying to reach you since yesterday.",
                "Two men are chatting in a pub one day. 'How did you get those scars on your nose?' said one. 'From glasses', said the other. 'Well why don't you try contact lenses?' asked the first. 'Because they don't hold as much beer', said the second.",
                "A man went to the doctor, 'Look doc', he said, 'I can't stop my hands from shaking.' 'Do you drink much?' asked the doctor. 'No', replied the man, 'I spill most of it.'",
                "Man goes to the doctor, 'Doctor, doctor. I keep seeing fish everywhere.' 'Have you seen an optician?' asks the doctor. 'Look I told you,' snapped the patient, 'It's fish that I see.'",
                "After a car crash one of the drivers was lying injured on the pavement. 'Don't worry', said a policeman who's first on the scene,' a Red Cross nurse is coming.' 'Oh no', moaned the victim, 'Couldn't I have a blonde, cheerful one instead?'",
                "A policeman walked over to a parked car and asked the driver if the car was licensed. 'Of course it is', said the driver. 'Great, I'll have a beer then', said the policeman.",
                "A policeman stops a woman and asks for her licence. 'Madam', he says, 'It says here that you should be wearing glasses.' 'Well', replies the woman, 'I have contacts.' 'Listen, love', says the copper, 'I don't care who you know; You're nicked!'",
                "A policeman stopped a motorist in the centre of town one evening. 'Would you mind blowing into this bag, sir?' asked the policeman. 'Why?' asked the driver. 'Because my chips are too hot', replied the policeman.",
                "Whizzing round a sharp bend on a country road a motorist ran over a large dog. A distraught farmer's wife ran over to the dead animal. 'I'm so very sorry', said the driver, 'I'll replace him, of course.' 'Well, I don't know', said the farmer's wife, 'Are you any good at catching rats?'",
                "Waiter, this coffee tastes like dirt! Yes sir, that's because it was ground this morning.",
                "Waiter, what is this stuff? That's bean salad sir. I know what it's been, but what is it now?",
                "Waiter: And how did you find your steak sir? Customer: I just flipped a chip over, and there it was!",
                "A guy goes into a pet shop and asks for a wasp. The owner tells him they don't sell wasps, to which the man says, 'Well you've got one in the window.'",
                "A man goes into a fish shop and says, 'I'd like a piece of cod, please.' Fishmonger says, 'It won't be long sir.' 'Well, it had better be fat then', replies the man.",
                "Man: Doctor, I've just swallowed a pillow. Doctor: How do you feel? Man: A little down in the mouth.",
                "Two goldfish are in a tank. One turns to the other and says, 'Do you know how to drive this thing?'",
                "A tortoise goes to the police station to report being mugged by three snails. 'What happened?' says the policeman. 'I don't know', says the tortoise. 'It was all so quick.'",
                "Little girl: Grandpa, can you make a sound like a frog? Grandpa: I suppose so sweetheart. Why do you want me to make a sound like a frog?' Little girl: Because Mum said that when you croak, we're going to Disneyland.",
                "'Is your mother home?' the salesman asked a small boy sitting on the front step of a house. 'Yeah, she's home', the boy said, moving over to let him past. The salesman rang the doorbell, got no response, knocked once, then again. Still no-one came to the door. Turning to the boy, the salesman said, 'I thought you said your mother was home.' The kid replied, 'She is, but I don't live here.'",
                "Mother: Why are you home from school so early? Son: I was the only one in the class who could answer a question. Mother: Oh, really? What was the question? Son: Who threw the rubber at the headmaster?",
                "A man's credit card was stolen but he decided not to report it because the thief was spending less than his wife did.",
                "A newly-wed couple had recently opened a joint bank account. 'Darling', said the man. 'The bank has returned that cheque you wrote last week.' 'Great', said the woman. 'What shall I spend it on next?'",
                "A man goes into a fish and chip shop and orders fish and chips twice. The shop owner says, 'I heard you the first time.'",
                "A tramp approached a well-dressed man. 'Ten pence for a cup of tea, Guv?' He asked. The man gave him the money and after for five minutes said, 'So where's my cup of tea then?'",
                "A neutron walks into a pub. 'I'd like a beer', he says. The landlord promptly serves him a beer. 'How much will that be?' asks the neutron. 'For you?' replies the landlord, 'No charge.'",
                "A woman goes to the doctor and says, 'Doctor, my husband limps because his left leg is an inch shorter than his right leg. What would you do in his case?' 'Probably limp, too', says the doc.",
                "Three monks are meditating in the Himalayas. One year passes in silence, and one of them says to the other, 'Pretty cold up here isn't it?' Another year passes and the second monk says, 'You know, you are quite right.' Another year passes and the third monk says, 'Hey, I'm going to leave unless you two stop jabbering!'",
                "A murderer, sitting in the electric chair, was about to be executed. 'Have you any last requests?' asked the prison guard. 'Yes', replied the murderer. 'Will you hold my hand?'",
                "A highly excited man rang up for an ambulance. 'Quickly, come quickly', he shouted, 'My wife's about to have a baby.' 'Is this her first baby?' asked the operator. 'No, you fool', came the reply, 'It's her husband.'",
                "A passer-by spots a fisherman by a river. 'Is this a good river for fish?' he asks. 'Yes', replies the fisherman, 'It must be. I can't get any of them to come out.'",
                "A man went to visit a friend and was amazed to find him playing chess with his dog. He watched the game in astonishment for a while. 'I can hardly believe my eyes!' he exclaimed. 'That's the smartest dog I've ever seen.' His friend shook his head. 'Nah, he's not that bright. I beat him three games in five.'",
                "A termite walks into a pub and says, 'Is the bar tender here?'",
                "A skeleton walks into a pub one night and sits down on a stool. The landlord asks, 'What can I get you?' The skeleton says, 'I'll have a beer, thanks' The landlord passes him a beer and asks 'Anything else?' The skeleton nods. 'Yeah...a mop...'",
                "A snake slithers into a pub and up to the bar. The landlord says, 'I'm sorry, but I can't serve you.' 'What? Why not?' asks the snake. 'Because', says the landlord, 'You can't hold your drink.'",
                "Descartes walks into a pub. 'Would you like a beer sir?' asks the landlord politely. Descartes replies, 'I think not' and ping! he vanishes.",
                "A cowboy walked into a bar, dressed entirely in paper. It wasn't long before he was arrested for rustling.",
                "A fish staggers into a bar. 'What can I get you?' asks the landlord. The fish croaks 'Water...'",
                "Two vampires walked into a bar and called for the landlord. 'I'll have a glass of blood', said one. 'I'll have a glass of plasma', said the other. 'Okay', replied the landlord, 'That'll be one blood and one blood lite.'",
                "How many existentialists does it take to change a light bulb?  Two. One to screw it in, and one to observe how the light bulb itself symbolises a single incandescent beacon of subjective reality in a netherworld of endless absurdity, reaching towards the ultimate horror of a maudlin cosmos of bleak, hostile nothingness.",
                "A team of scientists were nominated for the Nobel Prize. They had used dental equipment to discover and measure the smallest particles yet known to man. They became known as 'The Graders of the Flossed Quark...'",
                "A truck carrying copies of Roget's Thesaurus overturned on the highway. The local newspaper reported that onlookers were 'stunned, overwhelmed, astonished, bewildered and dumbfounded.'",
                "'My wife is really immature. It's pathetic. Every time I take a bath, she comes in and sinks all my little boats.'",
                "'How much will it cost to have the tooth extracted?' asked the patient. '50 pounds', replied the dentist. '50 pounds for a few moments' work?!' asked the patient. 'The dentist smiled, and replied, 'Well, if you want better value for money, I can extract it very, very slowly...'",
                "A doctor thoroughly examined his patient and said, 'Look I really can't find any reason for this mysterious affliction. It's probably due to drinking.' The patient sighed and snapped, 'In that case, I'll come back when you're damn well sober!'",
                "Doctor: Tell me nurse, how is that boy doing; the one who ate all those 5p pieces? Nurse: Still no change doctor.",
                "Doctor: Did you take the patient's temperature nurse? Nurse: No doctor. Is it missing?",
                "A depressed man turned to his friend in the pub and said, 'I woke up this morning and felt so bad that I tried to kill myself by taking 50 aspirin.' 'Oh man, that's really bad', said his friend, 'What happened?' The first man sighed and said, 'After the first two, I felt better.'",
                "A famous blues musician died. His tombstone bore the inscription, 'Didn't wake up this morning...'",
                "A businessman was interviewing a nervous young woman for a position in his company. He wanted to find out something about her personality, so he asked, 'If you could have a conversation with someone living or dead, who would it be?' The girl thought about the question: 'The living one', she replied.",
                "Manager to interviewee: For this job we need someone who is responsible. Interviewee to Manager: I'm your man then - in my last job, whenever anything went wrong, I was responsible.",
                "A businessman turned to a colleague and asked, 'So, how many people work at your office?' His friend shrugged and replied, 'Oh about half of them.'",
                "'How long have I been working at that office? As a matter of fact, I've been working there ever since they threatened to sack me.'",
                "In a courtroom, a mugger was on trial. The victim, asked if she recognised the defendant, said, 'Yes, that's him. I saw him clear as day. I'd remember his face anywhere.' Unable to contain himself, the defendant burst out with, 'She's lying! I was wearing a mask!'",
                "As Sid sat down to a big plate of chips and gravy down the local pub, a mate of his came over and said, 'Here Sid, me old pal. I thought you were trying to get into shape? And here you are with a high-fat meal and a pint of stout!' Sid looked up and replied, 'I am getting into shape. The shape I've chosen is a sphere.'",
                "Man in pub: How much do you charge for one single drop of whisky? Landlord: That would be free sir. Man in pub: Excellent. Drip me a glass full.",
                "I once went to a Doctor Who restaurant. For starters I had Dalek bread.",
                "A restaurant nearby had a sign in the window which said 'We serve breakfast at any time', so I ordered French toast in the Renaissance.",
                "Why couldn't the rabbit get a loan?  Because he had burrowed too much already!",
                "I phoned up the builder's yard yesterday. I said, 'Can I have a skip outside my house?'. The builder said, 'Sure. Do what you want. It's your house.'",
                "What's the diference between a sock and a camera? A sock takes five toes and a camera takes four toes!",
                "Woman on phone: I'd like to complain about these incontinence pants I bought from you! Shopkeeper: Certainly madam, where are you ringing from? Woman on phone: From the waist down!",
                "Knock knock.",
                "Two Oranges in a pub, one says to the other 'Your round.'.",
                "Guy : 'Doc, I've got a cricket ball stuck up my backside.' Doc : 'How's that?' Guy : 'Don't you start...'",
                "Two cows standing in a field. One turns to the other and says 'Moo!' The other one says 'Damn, I was just about to say that!'.",
                "A vampire bat arrives back at the roost with his face full of blood. All the bats get excited and ask where he got it from. 'Follow me', he says and off they fly over hills, over rivers and into a dark forest. 'See that tree over there', he says.  'WELL I DIDN'T!!'.",
                "A man goes into a bar and orders a pint. After a few minutes he hears a voice that says, 'Nice shoes'. He looks around but the whole bar is empty apart from the barman at the other end of the bar. A few minutes later he hears the voice again. This time it says, 'I like your shirt'. He beckons the barman over and tells him what's been happening to which the barman replies, 'Ah, that would be the nuts sir. They're complimentary'!",
                "A man was siting in a restaurant waiting for his meal when a big king prawn comes flying across the room and hits him on the back of the head. He turns around and the waiter said, 'That's just for starters'.",
                "Doctor! I have a serious problem, I can never remember what i just said. When did you first notice this problem? What problem?",
                "Now, most dentist's chairs go up and down, don't they? The one I was in went back and forwards. I thought, 'This is unusual'. Then the dentist said to me, 'Mitsuku, get out of the filing cabinet'.",
                "I was reading this book, 'The History of Glue'. I couldn't put it down.",
                "The other day someone left a piece of plastacine in my bedroom. I didn't know what to make of it.",
                "When I was at school people used to throw gold bars at me. I was the victim of bullion.",
                "I was playing the piano in a bar and this elephant walked in and started crying his eyes out. I said 'Do you recognise the tune?' He said 'No, I recognise the ivory.'",
                "I went in to a pet shop. I said, 'Can I buy a goldfish?' The guy said, 'Do you want an aquarium?' I said, 'I don't care what star sign it is.'",
                "My mate Sid was a victim of I.D. theft. Now we just call him S.",
                "David Hasselhoff walks into a bar and says to the barman, 'I want you to call me David Hoff'.  The barman replies 'Sure thing Dave... no hassle'"
            ]
    }]"""

def get_questions(context, max_length=64):
    qns=[]
    sentences=context.split('.')
    for sentence in sentences[:-1]:
        input_text = "answer: %s  context: %s </s>" % ('', sentence)
        features = tokenizer_q([input_text], return_tensors='pt')

        output = model_q.generate(input_ids=features['input_ids'], 
                   attention_mask=features['attention_mask'],
                   max_length=max_length)
        qns.append(tokenizer_q.decode(output[0]).replace('<pad> question: ','').replace('</s>',''))
    return qns

class PythonPredictor:
    def __init__(self):
        # model_file_1 = "../input/s2v-old/s2v_old"
        
        # print ("s2v model already exists.")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('C:/Users/EMMANUEL GUDINHO/OneDrive/Desktop/chatbot-emmanuel/django-chat-app-main/static/models/result')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # model.eval()
        self.device = device
        self.model = model
        self.nlp = spacy.load('en_core_web_sm')

        self.s2v = Sense2Vec().from_disk('C:/Users/EMMANUEL GUDINHO/OneDrive/Desktop/chatbot-emmanuel/django-chat-app-main/static/models/s2v_old')

        self.fdist = FreqDist(brown.words())
        self.normalized_levenshtein = NormalizedLevenshtein()
        self.set_seed(42)
    def set_seed(self,seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    def paraphrase(self,payload):
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 4)
        }

        text = inp['input_text']
        num = inp['max_questions']
        
        self.sentence= text
        self.num= num+2
        self.text= "paraphrase: " + self.sentence + " </s>"

        encoding = self.tokenizer.encode_plus(self.text,pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        beam_outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=50,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=self.num
        )

        print ("\nOriginal Question ::")
        print (text)
        print ("\n")
        print ("Paraphrased Questions :: ")
        final_outputs =[]
        for beam_output in beam_outputs:
            sent = self.tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            if sent.lower() != self.sentence.lower() and sent not in final_outputs:
                final_outputs.append(sent)

        for i, final_output in enumerate(final_outputs):
            print("{}: {}".format(i, final_output))

        if torch.device=='cuda':
            torch.cuda.empty_cache()
        
        return final_outputs
    
def ChatAnswer(question, context):
        inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]

        outputs = model(input_ids, token_type_ids=token_type_ids)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end + 1])

        return answer

def optimize_code(chatanswers):
    """
    Optimizes the given code.

    Args:
      chatanswers: The list of chat answers.

    Returns:
      A new list with the optimized code.
    """

    optimized_code = []

    for sublist in chatanswers:
        element_counts = Counter(sublist)
        sorted_elements = sorted(element_counts, key=lambda x: element_counts[x], reverse=True)
        unique_elements = [elem for elem in sorted_elements if elem not in ["", "[cls]"] and "[cls]" not in elem and "[sep]" not in elem]
        optimized_code.append(unique_elements)

    return optimized_code

def balance_data(df_patterns):
    df_intent = df_patterns['tag']
    max_counts = df_intent.value_counts().max() #max number of examples for a class

    new_df = df_patterns.copy()
    for i in df_intent.unique():
        i_count = int(df_intent[df_intent == i].value_counts())
        if i_count < max_counts:
            i_samples = df_patterns[df_intent == i].sample(max_counts - i_count, replace = True, ignore_index = True)
            new_df = pd.concat([new_df, i_samples])
    return new_df

def map_function(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks},labels

# Main Function for ML chatbot 
def bot(request):
    print(tf.config.list_physical_devices('GPU'))
    print("-----Scraping Data-----")
    # response = requests.get('https://www.xircls.com')
    response = requests.get('https://www.pvrcinemas.com/faqs')

    soup = BeautifulSoup(response.content, 'html.parser')

    for anchor in soup.find_all('a'):
        anchor.extract()

    all_text = soup.get_text()
    scraped_data = re.sub(r"\s{2,}", " ", all_text)
    # scraped_data = scraped_data[:100] #Comment this later on
    print("-----Scraping Done-----")
    print(scraped_data)

    context_q = scraped_data
    print("-----Generating Questions-----")
    questions = list(set(get_questions(context_q)))
    for i,qn in enumerate(questions,start=1):
        print(str(i)+') '+qn)
    print("-----Questions Generated-----")
    print("-----Generating Alternative Questions-----")
    out2= []
    for i in questions:
        payload={
            "input_text" : i,
            "max_questions" : 8
            }
        generator = PythonPredictor()
        out2.append([i.split(": ")[1] for i in generator.paraphrase(payload)])
    print("-----Alt Questions Generated-----")
    print("-----Generating Answers-----")
    context = scraped_data
    chatanswers = []
    for i in out2:
        temp=[]
        for j in i:
            ans = ChatAnswer(j,context)
            ans1 = ans.capitalize()
            temp.append(ans1)
        chatanswers.append(temp)
    print(chatanswers)
    print("-----Answers Generated-----")
    print("-----Cleaning Answers-----")
    chatans = optimize_code(chatanswers)
    print(chatans)
    print("-----Cleaned Answers-----")
    print("-----Creating Database-----")
    data = json.loads(default_data)
    for i in range(len(questions)):
        a={}
        a['tag'] = str(i)
        a["patterns"] = out2[i]
        a["responses"] = chatans[i]
        data.append(a)
    # print(data)

    df = json.dumps(data, ensure_ascii=False)
    df = json.loads(df)
    df = pd.DataFrame(df)
    print(df)
    df_patterns = df[['patterns', 'tag']]
    df_responses = df[['responses', 'tag']]
    df_patterns = df_patterns.explode('patterns')
    df_patterns.drop_duplicates(inplace= True)
    df_patterns = balance_data(df_patterns)
    seq_len = 256
    num_samples = len(df_patterns)

    Xids = numpy.zeros((num_samples, seq_len)) #Token ids
    Xmask = numpy.zeros((num_samples, seq_len)) #attention mask

    for i, phrase in enumerate(df_patterns['patterns']):
        tokens = tokenizer_chatbot.encode_plus(phrase, max_length= seq_len, truncation= True, padding= 'max_length', add_special_tokens = True, return_tensors= 'tf')

        Xids[i, :] = tokens['input_ids']
        Xmask[i, :] = tokens['attention_mask']
    
    arr = le.fit_transform(df_patterns['tag'])
    classes = le.classes_
    print(classes)
    labels = numpy.zeros((num_samples, arr.max()+1))
    labels[numpy.arange(num_samples), arr] = 1

    dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
    dataset = dataset.map(map_function)
    batch_size = 8

    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder= True)

    input_ids = tf.keras.layers.Input(shape= (seq_len,), name= 'input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape= (seq_len,), name= 'attention_mask', dtype='int32')

    embeddings = bert_chatbot.bert(input_ids, attention_mask= mask)[1]

    x = tf.keras.layers.Dense(1024, activation= 'relu')(embeddings)
    y = tf.keras.layers.Dense(arr.max()+1, activation= 'softmax', name= 'outputs')(x)

    model = tf.keras.Model(inputs= [input_ids, mask], outputs = y)
    optimizer = tf.keras.optimizers.Adam(learning_rate= 1e-5)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    model.compile(optimizer= optimizer, loss= loss, metrics = [acc])
    history = model.fit(dataset, epochs= 10)
    model.save('django-chat-app-main/static/models/intent_prediction_model.h5')

    model_path = os.path.join(settings.STATIC_ROOT, 'models', 'intent_prediction_model.h5')
    with custom_object_scope({'TFBertMainLayer': TFBertMainLayer}):
        model = tf.keras.models.load_model(model_path)
    seq_len = 256

    # if request.method == "POST":
    #     selected_option = request.POST.get("selected_option")
    #     user_input = request.POST.get("user_input")
    #     print(user_input)
    #     if selected_option:
    #         # Handle the selected option data here
    #         print("Selected option:", selected_option)
    #         processed_data = predict_res(selected_option)
    #         print(processed_data)
    #         for element in predict_res(selected_option):
    #             for intents in data:
    #                 if element in intents['tag']:
    #                     responses = intents['responses']
    #                     if element in ["greeting", "goodbye", "thanks", "about", "name", "help", "createaccount", "complaint",'Greeting', 'GreetingResponse', 'CourtesyGreeting', 'CourtesyGreetingResponse', 'CurrentHumanQuery', 'NameQuery', 'RealNameQuery', 'TimeQuery', 'Thanks', 'NotTalking2U', 'UnderstandQuery', 'Shutup', 'Swearing', 'GoodBye', 'CourtesyGoodBye', 'WhoAmI', 'Clever', 'Gossip', 'Jokes', 'PodBayDoor', 'PodBayDoorResponse', 'SelfAware']:
    #                         random_response = random.choice(responses)
    #                         print("Random Response")
    #                         print(random_response)
    #                         return JsonResponse({'output': [random_response], 'option': processed_data})
    #                     else:
    #                         print("Responses")
    #                         print(responses)
    #                         return JsonResponse({'output': responses, 'option': processed_data})
    #         # Process the selected option and return the desired response
    #     elif user_input:
    #         processed_data = predict_res(user_input)
    #         print("User: ", user_input)
    #         print(processed_data)
            
    #         for element in predict_res(user_input):
    #             for intents in data:
    #                 if element in intents['tag']:
    #                     responses = intents['responses']
    #                     if element in ["greeting", "goodbye", "thanks", "about", "name", "help", "createaccount", "complaint",'Greeting', 'GreetingResponse', 'CourtesyGreeting', 'CourtesyGreetingResponse', 'CurrentHumanQuery', 'NameQuery', 'RealNameQuery', 'TimeQuery', 'Thanks', 'NotTalking2U', 'UnderstandQuery', 'Shutup', 'Swearing', 'GoodBye', 'CourtesyGoodBye', 'WhoAmI', 'Clever', 'Gossip', 'Jokes', 'PodBayDoor', 'PodBayDoorResponse', 'SelfAware']:
    #                         random_response = random.choice(responses)
    #                         print("Random Response")
    #                         print(random_response)
    #                         return JsonResponse({'output': [random_response], 'option': processed_data})
    #                     else:
    #                         print("User Input Responses")
    #                         print(responses)
    #                         return JsonResponse({'output': responses, 'option': processed_data})
    #     else: 
    #         return JsonResponse({'output': "Could not get your response",'option': []})
    
    return render(request, 'bot.html')

# Below Functions for Live Chat and Admin Panel 
def home(request):
    return render(request, 'home.html')

def room(request, room):
    username = request.GET.get('username')
    room_details = Room.objects.get(name=room)
    return render(request, 'room.html', {
        'username': username,
        'room': room,
        'room_details': room_details
    })

def checkview(request):
    username = request.META.get('REMOTE_ADDR')

    # Generate a new support room name
    support_rooms = Room.objects.filter(name__startswith='support')
    room_count = support_rooms.count() + 1
    new_room_name = 'support{}'.format(room_count)
    record = DashboardEntry(room_name=new_room_name,user_name=username)
    record.save()
    # Create a new support room
    new_support_room = Room(name=new_room_name)
    new_support_room.save()
    messages.add_message(request, messages.SUCCESS, 'Someone needs your support ')

    return redirect('/{}/?username={}'.format(new_room_name, username))

def send(request):
    message = request.POST['message']
    username = request.POST['username']
    room_id = request.POST['room_id']
    # if username == "customercare":
    room = Room.objects.get(id=room_id)
    new_message = Message.objects.create(value=message, user=username, room=room)
    new_message.save()
    if new_message:
        dashboard_entry = DashboardEntry.objects.get(room_name=room.name)
        print(dashboard_entry)
        dashboard_entry.has_unread_message = True
        dashboard_entry.save()
    else: 
        dashboard_entry = DashboardEntry.objects.get(room_name=room.name)
        print(dashboard_entry)
        dashboard_entry.has_unread_message = False
        dashboard_entry.save()

    return HttpResponse('Message sent successfully')

def getMessages(request, room):
    room_details = Room.objects.get(name=room)
    print(room_details)
    messages = Message.objects.filter(room=room_details.id)
    # Save all the messages
    for message in messages:
        message.save()
        
    return JsonResponse({"messages": list(messages.values())})

def panel(request):
    records=DashboardEntry.objects.all()
    
    return render(request, 'adminpanel.html',{'data':records}) 


