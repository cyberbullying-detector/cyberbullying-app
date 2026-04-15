from flask import Flask, render_template, request
import joblib
import re
import string
import nltk
import demoji
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

model_pipeline = joblib.load("cyberbullying_model_pipeline.pkl")

# ── Preprocessing ──────────────────────────────────────────────
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've',
                  'u', 'ur', 'don', 'p', 't', 's', 'aren', 'kp', 'o', 'kat',
                  'de', 're', 'amp', 'will', 'wa', 'e', 'like'])
stemmer = SnowballStemmer('english')

def clean_text(text):
    pattern = re.compile(
        r"(#[A-Za-z0-9]+|@[A-Za-z0-9]+|https?://\S+|www\.\S+|\S+\.[a-z]+|RT @)"
    )
    text = pattern.sub('', text)
    text = " ".join(text.split())
    text = text.lower()

    # Emoji → text (safe replacement)
    try:
        emoji_found = demoji.findall(text)
        for emot, name in emoji_found.items():
            text = text.replace(emot, "_".join(name.split()))
    except Exception:
        pass

    # Remove punctuation BEFORE stemming
    remove_punc = re.compile(r"[%s]" % re.escape(string.punctuation))
    text = remove_punc.sub('', text)

    # Stem after punctuation is gone
    text = " ".join([stemmer.stem(word) for word in text.split()])

    # Remove stopwords
    text = " ".join([w for w in text.split() if w not in STOPWORDS])

    return text

# ── Keyword Rules (built from your real dataset) ───────────────
KEYWORD_RULES = {
    'age': [
        'bullied at school', 'bullying at school', 'bully at school',
        'middle school', 'high school bully', 'school bully',
        'kids are mean', 'mean kids', 'kids bully',
        'old people', 'old man', 'old woman', 'too old',
        'elder', 'boomer', 'senile', 'elderly', 'aging',
        'grandpa', 'grandma', 'old folk', 'past their prime',
        'young people', 'young kids', 'little kids',
    ],
    'gender': [
        'rape joke', 'rape jokes', 'making rape', 'rape is',
        'sexist', 'sexism', 'feminist', 'feminazi',
        'girls are', 'women are', 'woman are', 'females are',
        'dumb girl', 'weak female', 'women should', 'girls should',
        'woman should', 'ladies are', 'women belong', 'girls belong',
        'gay joke', 'gay jokes', 'calling gay', 'call him gay',
        'bitch ass', 'dumb bitch', 'stupid bitch',
        'female are', 'girl are', 'woman belongs',
    ],
    'ethnicity': [
        # racial slurs from your dataset
        'nigger', 'niggers', 'negro', 'colored people',
        'dumb nigger', 'fuck obama', 'dumb ass nigger',
        'black people', 'white people', 'brown people',
        'racist', 'racism', 'racial',
        # asian stereotypes
        'all asians', 'asians look', 'asians cannot drive',
        'asians cant drive', 'asian people', 'asian students',
        'asians cheat', 'chinese people', 'these chinese',
        'taking over', 'taking over our', 'taking over everything',
        # white stereotypes
        'white people have', 'white culture', 'whites have no',
        # general
        'go back to', 'your kind', 'dirty immigrant',
        'these people', 'those people', 'your race',
        'not from here', 'foreigner',
        'all blacks', 'all whites', 'all asians', 'all arabs',
        'all mexicans', 'all indians', 'all chinese',
        'these blacks', 'these whites', 'these arabs',
        'these mexicans', 'these indians', 'these immigrants',
        'black people are', 'white people are', 'asian people are',
        'chinese people are', 'arab people are', 'indian people are',
    ],
    'religion': [
        'muslims are', 'muslim are', 'islam is', 'islamic terror',
        'radical islam', 'radical muslim', 'muslim terrorist',
        'isis', 'quran', 'terrorist muslim', 'dirty christian',
        'christians are', 'hindus are', 'jews are',
        'infidel', 'kafir', 'that religion', 'your religion',
        'your god', 'your prophet', 'their religion',
        'anti-semit', 'antisemit', 'islamophob',
    ],
    'other_cyberbullying': [
        'kill yourself', 'kys', 'go die', 'end yourself',
        'nobody likes you', 'everyone hates you',
        'you should die', 'you deserve to die',
    ],
}

# ── Standalone Abuse (no group target needed) ──────────────────
STANDALONE_ABUSE = [
    'fuck you', 'fuck off', 'fuckyou', 'fck you',
    'asshole', 'ass hole', 'piece of shit', 'you piece',
    'go to hell', 'screw you', 'piss off', 'shove it',
    'you bastard', 'bastard', 'son of a bitch',
    'motherfucker', 'motherfucking',
    'shut the fuck', 'what the fuck',
    'bitch ass', 'cry baby', 'crybaby',
    'piece of crap', 'eat shit', 'drop dead',
    'die already', 'i hate you', 'we hate you',
    'kill yourself', 'go die', 'kys',
    'nobody likes you', 'everyone hates you',
    'you are worthless', 'you are useless',
    'you are pathetic', 'you are disgusting',
    'stupid idiot', 'dumb idiot', 'complete idiot',
    'ugly bastard', 'ugly bitch', 'fat bitch',
    'shut up idiot', 'shut up stupid',
]

# ── Bully Words (with group = targeted bullying) ───────────────
BULLY_WORDS = [
    'useless', 'worthless', 'pathetic', 'disgusting',
    'stupid', 'idiot', 'idiots', 'loser', 'ugly', 'dumb',
    'hate you', 'kill yourself', 'nobody likes', 'go die',
    'shut up', 'you suck', 'trash', 'garbage', 'waste',
    'inferior', 'filthy', 'nasty', 'horrible', 'terrible',
    'awful', 'hate them', 'hate those', 'all look the same',
    'cannot drive', 'cant drive', 'no culture', 'boring and bland',
    'they cheat', 'taking over', 'succeed because',
    'fucking idiot', 'dumb ass', 'dumb fuck',
    'fuck you', 'asshole', 'piece of shit', 'bastard',
    'son of a bitch', 'motherfucker', 'screw you',
    'piece of crap', 'eat shit', 'drop dead', 'die already',
    'i hate you', 'we hate you',
]

# ── Group Words ────────────────────────────────────────────────
GROUP_WORDS = {
    'age': [
        'old', 'elder', 'elderly', 'boomer', 'grandpa', 'grandma',
        'senior', 'aged', 'kids', 'children', 'school', 'young',
    ],
    'gender': [
        'woman', 'women', 'girl', 'girls', 'female', 'females',
        'lady', 'ladies', 'gay', 'lesbian', 'male', 'men', 'man',
    ],
    'ethnicity': [
        'black', 'white', 'brown', 'asian', 'asians', 'arab', 'arabs',
        'immigrant', 'foreign', 'race', 'chinese', 'mexican', 'indian',
        'japanese', 'korean', 'hispanic', 'latino', 'latina', 'jewish',
        'negro', 'colored', 'nigger', 'niggers',
    ],
    'religion': [
        'muslim', 'muslims', 'christian', 'christians', 'hindu', 'hindus',
        'jewish', 'jew', 'jews', 'islam', 'islamic', 'religion',
        'god', 'allah', 'quran', 'bible', 'church', 'mosque', 'temple',
    ],
}

# ── Smarter Keyword Fallback ───────────────────────────────────
def keyword_fallback(original_text):
    text_lower = original_text.lower()

    # 1. Check category-specific phrase rules first
    for category, keywords in KEYWORD_RULES.items():
        if any(kw in text_lower for kw in keywords):
            return category

    # 2. Standalone abuse — cyberbullying regardless of target
    if any(word in text_lower for word in STANDALONE_ABUSE):
        return 'other_cyberbullying'

    # 3. Bully word + group word together = targeted bullying
    has_bully = any(word in text_lower for word in BULLY_WORDS)
    if has_bully:
        for category, group_words in GROUP_WORDS.items():
            if any(gw in text_lower for gw in group_words):
                return category
        return 'other_cyberbullying'

    return 'not_cyberbullying'

# ── Labels ─────────────────────────────────────────────────────
labels = {
    "age":                "🧒 Age-based Cyberbullying",
    "gender":             "⚧ Gender-based Cyberbullying",
    "ethnicity":          "🌍 Ethnicity-based Cyberbullying",
    "religion":           "🕌 Religion-based Cyberbullying",
    "other_cyberbullying":"⚠️ Other Cyberbullying",
    "not_cyberbullying":  "✅ Safe Message",
}

CONFIDENCE_THRESHOLD = 0.45

# ── Routes ─────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    original_text = request.form.get("message", "").strip()

    if not original_text:
        return render_template("index.html", error="Please enter a message.")

    cleaned = clean_text(original_text)

    proba   = model_pipeline.predict_proba([cleaned])[0]
    classes = model_pipeline.classes_
    max_prob = float(max(proba))
    raw_pred = classes[proba.argmax()]

    print(f"\n--- PREDICTION DEBUG ---")
    print(f"Original  : {original_text}")
    print(f"Cleaned   : {cleaned}")
    print(f"Raw Result: {raw_pred} ({max_prob:.1%} confidence)")

    # Use fallback if model is not confident enough
    if max_prob < CONFIDENCE_THRESHOLD:
        final_pred = keyword_fallback(original_text)
        print(f"Low confidence → fallback → {final_pred}")
    else:
        final_pred = raw_pred

    print(f"Final     : {final_pred}\n")

    result = labels.get(final_pred, f"⚠️ Detected: {final_pred}")

    return render_template(
        "index.html",
        message=original_text,
        prediction=result,
        confidence=f"{max_prob:.1%}"
    )

if __name__ == "__main__":
    app.run(debug=True)