"""Probe 7: generation-quality assessment across 10 Indic languages.

Two prompt types per language (factual QA + instruction following), greedy decode,
instruct model, thinking off. Outputs are judged qualitatively in the report.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "XHToken/Spark-X2.6-1.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    REPO, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()

PROMPTS = [
    # Hindi
    ("hi_qa", "ताज महल कहाँ स्थित है? एक वाक्य में उत्तर दीजिए।"),
    ("hi_task", "एक छोटा अनुच्छेद लिखिए जिसमें बताइए कि बारिश क्यों होती है।"),
    # Bengali
    ("bn_qa", "রবীন্দ্রনাথ ঠাকুর কে? এক বাক্যে উত্তর দিন।"),
    ("bn_task", "স্বাস্থ্যের জন্য হাঁটার দুটি উপকারিতা লিখুন।"),
    # Tamil
    ("ta_qa", "தமிழ்நாட்டின் தலைநகரம் எது? ஒரு வாக்கியத்தில் பதிலளிக்கவும்."),
    ("ta_task", "கல்வி ஏன் முக்கியம் என்பதை இரண்டு வாக்கியங்களில் விளக்கவும்."),
    # Telugu
    ("te_qa", "చార్మినార్ ఏ నగరంలో ఉంది? ఒక వాక్యంలో సమాధానం ఇవ్వండి."),
    ("te_task", "పుస్తకాలు చదవడం వల్ల రెండు ప్రయోజనాలను తెలుగులో వివరించండి."),
    # Marathi
    ("mr_qa", "महाराष्ट्राची राजधानी कोणती? एका वाक्यात उत्तर द्या."),
    ("mr_task", "पाण्याची बचत का महत्त्वाची आहे ते दोन वाक्यांत सांगा."),
    # Gujarati
    ("gu_qa", "સ્ટેચ્યુ ઓફ યુનિટી ક્યાં આવેલું છે? એક વાક્યમાં જવાબ આપો."),
    ("gu_task", "સ્વસ્થ જીવન માટે વ્યાયામના બે ફાયદા ગુજરાતીમાં લખો."),
    # Kannada
    ("kn_qa", "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು? ಒಂದು ವಾಕ್ಯದಲ್ಲಿ ಉತ್ತರಿಸಿ."),
    ("kn_task", "ಮಳೆ ನೀರು ಸಂಗ್ರಹಣೆಯ ಎರಡು ಪ್ರಯೋಜನಗಳನ್ನು ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ."),
    # Malayalam
    ("ml_qa", "കേരളത്തിന്റെ തലസ്ഥാനം ഏത്? ഒരു വാക്യത്തിൽ ഉത്തരം പറയൂ."),
    ("ml_task", "വൃക്ഷത്തൈകൾ നടുന്നതിന്റെ രണ്ട് ഗുണങ്ങൾ മലയാളത്തിൽ എഴുതൂ."),
    # Punjabi
    ("pa_qa", "ਗੋਲਡਨ ਟੈਂਪਲ ਕਿੱਥੇ ਹੈ? ਇੱਕ ਵਾਕ ਵਿੱਚ ਜਵਾਬ ਦਿਓ।"),
    ("pa_task", "ਸਵੇਰੇ ਉੱਠਣ ਦੇ ਦੋ ਫਾਇਦੇ ਪੰਜਾਬੀ ਵਿੱਚ ਲਿਖੋ।"),
    # Odia
    ("or_qa", "ଜଗନ୍ନାଥ ମନ୍ଦିର କେଉଁଠାରେ ଅଛି? ଗୋଟିଏ ବାକ୍ୟରେ ଉତ୍ତର ଦିଅନ୍ତୁ।"),
    ("or_task", "ପରିଷ୍କାର ପାଣି ପିଇବାର ଦୁଇଟି ଲାଭ ଓଡ଼ିଆରେ ଲେଖ।"),
    # Urdu
    ("ur_qa", "پاکستان کا دارالحکومت کیا ہے؟ ایک جملے میں جواب دیں۔"),
    ("ur_task", "پودے لگانے کے دو فائدے اردو میں لکھیں۔"),
]

report = {}
for name, text in PROMPTS:
    msgs = [{"role": "user", "content": text}]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", reasoning_effort="low").to(device)
    except TypeError:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=180, do_sample=False,
                             pad_token_id=tok.pad_token_id or 2)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    report[name] = {"prompt": text, "output": gen}
    print(f"\n===== {name} =====\nPROMPT: {text}\nOUTPUT: {gen[:500]}")

os.makedirs("results", exist_ok=True)
with open("results/probe7_decode.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\nDONE probe7")
