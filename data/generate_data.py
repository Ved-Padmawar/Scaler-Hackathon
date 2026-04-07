import json
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SENDERS = ["Rahul", "Priya", "Amit", "Sneha", "Karan", "Meera"]


def topic_sequence(labels):
    return [
        labels[0],
        labels[4],
        labels[1],
        labels[2],
        labels[3],
        labels[0],
        labels[2],
        labels[4],
        labels[1],
        labels[3],
    ] * 5


def build_messages(labels, triads, start_time):
    label_counts = {label: 0 for label in labels}
    sequence = topic_sequence(labels)
    timestamp = datetime.fromisoformat(start_time)
    messages = []
    sender_index = 0

    for topic in sequence:
        triad_index = label_counts[topic] // 3
        texts = triads[topic][triad_index]
        label_counts[topic] += 3

        for offset, text in enumerate(texts):
            messages.append(
                {
                    "id": f"msg_{len(messages) + 1:03d}",
                    "sender": SENDERS[(sender_index + offset) % len(SENDERS)],
                    "text": text,
                    "timestamp": timestamp.isoformat(timespec="minutes"),
                    "ground_truth_label": topic,
                }
            )
            timestamp += timedelta(minutes=1)

        sender_index += 1
        timestamp += timedelta(minutes=2 if len(messages) % 9 else 3)

    return messages


electronics_triads = {
    "product_inquiry": [
        ("air fryer 4L black bacha hai kya", "which brand wala?", "TechChef ka tha shyd, pic bhejo na"),
        ("noise buds x7 white ya blue any left?", "white maybe, store2 check karo", "haan wohi, customer wait krra"),
        ("HP 15s i5 16gb wala abhi live hai?", "silver only i think", "ok send exact model once"),
        ("samsung 43 smart tv with wall mount mil jayega kya", "mount alag hoga maybe", "acha stock confirm kar do"),
        ("type c charger fast one 25 pcs possible?", "which one? pd ya normal", "bhai dono ka qty bolo"),
        ("macbook air m2 512 midnight avail? outlet 2", "midnight nahi maybe starlight", "haan chalega but sealed chahiye"),
        ("insta360/ action cam wala piece demo me hai?", "demo hai but battery low lol", "customer bas dekhna chah raha"),
        ("router ax3000 under 6k kuch hai?", "tp link tha kal", "acha same invoice pe add karna ho sakta"),
        ("sony soundbar with woofer kaunsa ready stock me", "which budget bro", "20 ke under ideally"),
        ("washing machine 8kg front load grey left or gone?", "1 maybe hold pe", "hold kis naam pe?"),
    ],
    "pricing": [
        ("best kya dena 3 iph 14 128 on gst", "prepaid pe aur niche ja sakta", "final bolo na, banda call pe hai"),
        ("same mouse online 799 store 850 kyun bolu usko", "cash pe adjust kar de?", "invoice today hi chahiye"),
        ("6 mixer grinder combo final kitna", "with gst ya without?", "gst incl bolna usne"),
        ("bluetooth speaker 12 pcs dealer rate asap", "shipping extra rakhna", "hmm but round figure chahiye"),
        ("inverter+battery 4 sets kya close karu", "old rate mat bhejna", "haan fresh quote bhej"),
        ("student bill pe mac air lowest kitna jaega", "card pe diff h", "sir ko just headline num chahiye"),
        ("ps5 + extra ctrl final final", "which controller color 🤦", "koi bhi, rate bata pehle"),
        ("air cooler bulk 8 pcs no install final?", "transport alag hai", "acha all incl bhi send"),
        ("monitor + kb combo same as last time de skte?", "qty kitni?", "5 abhi, maybe 7 later"),
        ("dealer bolra earbuds 999 me de do warna leave", "999 tough hai yaar", "thik 1049 last?"),
    ],
    "complaint": [
        ("earbuds one side dead in 2 din 😑", "invoice pic bhejo", "bheja tha upar, check karo"),
        ("fan making tik tik sound after install", "video manga?", "haan but low audio hai"),
        ("monitor stand hi nahi nikla box se", "sealed tha kya", "customer bolra yes but trust nhi"),
        ("smartwatch strap day1 me toot gaya", "replacement ya refund puchha?", "dono bolra lol"),
        ("soundbar remote missing in box", "which order?", "same morning wala blue dart"),
        ("seal open jaisa lag raha customer shouting", "send pic fast", "bhej diya grp me"),
        ("mouse scroll jam after 1 day use", "brand?", "hp wala, grey"),
        ("mixer overheating bolke 3rd call aa gaya", "service center pin bheju?", "pehle soothe karo usko"),
        ("tv panel pe scratch nikla delivery ke baad", "unbox video hai?", "partial hai bas"),
        ("adapter garam ho rha bohot, banda dara hua", "load kitna tha?", "bola normal ही था"),
    ],
    "order_update": [
        ("ORD 458 packed, rider abhi nahi", "awb bana kya?", "haan but not picked"),
        ("payment recd for 471, move kar do", "ok done", "dispatch before lunch pls"),
        ("455 delivered late, 8:55 pm only", "customer ok tha?", "thoda crib kia but fine"),
        ("462 scanned but not packed yet", "which shelf me pada?", "C rack maybe"),
        ("awb live for 510, tracking in 20", "send number", "grp me drop karta"),
        ("cancel mat karo, address corrected now", "re-opened shipment?", "haan hub se niklega"),
        ("pickup customer aa raha 6 bje", "item side me rakhna", "charger bhi include hai"),
        ("return reached hub, qc kal hoga", "refund hold then", "yes till qc"),
        ("same day promise mat karo jaipur pls", "noted", "app pe eta edit bhi karo"),
        ("replacement approved, dispatch tomo", "old unit reverse booked?", "haan ho gaya"),
    ],
    "general": [
        ("haan", "ok done", "pin new sheet pls"),
        ("which one?", "HP wala", "acha"),
        ("bhai check karo", "checking", "👍"),
        ("send pic", "sent", "not clear yaar"),
        ("kal call karte", "haan remind me", "12 ke baad"),
        ("bro outlet 2 khul gaya?", "haan abhi", "ok"),
        ("lunch ke baad dekhte?", "fine", "mat bhoolna bas"),
        ("pls dont overpromise today", "noted boss", "traffic weird hai"),
        ("same wala doc kisko bhejna tha", "meera ko", "done"),
        ("acha hold on", "network gaya tha", "ab bolo"),
    ],
}

restaurant_triads = {
    "menu_query": [
        ("white sauce pasta without garlic ho skta?", "chef se puchu?", "haan table wait kr rha"),
        ("biryani combo me raita included ya extra?", "depends on outlet maybe", "cp wale me kya hai"),
        ("jain gravy for paneer lababdar doable?", "onion garlic full hata denge", "ok note kar"),
        ("brownie with icecream abhi available?", "icecream low hai", "which flavor bacha"),
        ("sesame salad me exact allergen kya kya", "nuts nahi hai shyd", "shyd nahi pls confirm 😅"),
        ("kids meal with fries still same?", "juice bhi aata maybe", "customer pooch rha bas"),
        ("millet khichdi after 7 bhi serve hota?", "if prep bacha then yes", "acha tentative bol deta"),
        ("hakka noodles less spicy kar skte?", "haan but mention karna", "done"),
        ("ramen veg fully ho jayega kya", "broth swap karna padega", "chef ok bola kya"),
        ("pesto pasta me nuts hai na?", "pine nuts nahi, sauce premix wala", "send proper line pls"),
    ],
    "reservation": [
        ("table for 8 at 8:30 cp confirm fast", "name?", "Sharma ji"),
        ("2 tables together rakh sakte 9pm?", "which branch?", "noida sec18"),
        ("anniversary couple wants corner if poss", "cake aa rha unka?", "haan small one"),
        ("6 pax tentative only mark karna", "ok till what time hold?", "7:15"),
        ("7:45 booking ke liye 2 kids chair side me", "done", "window mila?"),
        ("tomorrow lunch 10 pax corp group block", "advance liya?", "nahi abhi"),
        ("high chair with 7 pm booking pls", "name bhejo", "aarav family"),
        ("window seat chahiye but walkin jaisa scene", "possible if no show", "thik waitlist pe daalo"),
        ("8 pm ko 8:30 shift kar do guest call aya", "same table?", "haan better"),
        ("birthday setup wala booking 5 pax note kia?", "balloons bhi?", "minimal only"),
    ],
    "complaint": [
        ("soup cold gaya again", "which table?", "14A maybe"),
        ("mint chutney bitter bol rahe 2 tables", "batch check kia?", "abhi kara"),
        ("zomato order leaked full 😓", "rider wait krra?", "haan downstairs"),
        ("naan rubbery tha, guest not happy", "replace kara?", "haan but mood off"),
        ("delay hua 35 min callback pls", "which token?", "takeaway 221"),
        ("brownie portion chhota laga unko", "old pic compare hai?", "nah bas verbal"),
        ("ac bohot thanda near table 14", "food cold ho gaya?", "yep same complaint"),
        ("one butter naan missing in pickup", "bill pic bhejo", "sent"),
        ("washroom dirty bola sunday dinner me", "housekeeping informed?", "haan late hua"),
        ("sizzler plate chipped nikli yaar", "stack check karo sab", "done 3 aur nikle"),
    ],
    "staff_update": [
        ("Rakesh absent, fever", "cash desk kaun lega?", "Amit maybe"),
        ("service guy 30 min late traffic", "noted", "host ko bata diya"),
        ("who swapped off with Manoj?", "me", "ok sheet update"),
        ("dishwasher guy post 12 ayega", "tab tak manually chalao", "haan"),
        ("tandoor helper leaving early today", "replacement mil gaya?", "Sneha dekh rhi"),
        ("new steward Priya evening se join", "brief POS pls", "haan kar denge"),
        ("pre shift briefing 5 sharp", "everyone pls on floor", "late mat aana"),
        ("2 trainees kitchen prep se aa rahe", "uniform hai?", "one missing cap"),
        ("Anita host desk le legi first half", "billing kaun?", "Rahul"),
        ("chef Manoj mood off lol careful", "again kya", "supplier fight"),
    ],
    "inventory": [
        ("paneer low bro, 6kg max", "lunch nikal jayega?", "tight hai"),
        ("3 crates cola chahiye before rush", "warehouse se mangau?", "haan asap"),
        ("tomato puree almost khatam", "kitna bacha", "1 pouch bas"),
        ("mushroom final qty bolo vendor waiting", "4 kg add", "noted"),
        ("vanilla tub 1 hi bacha", "brownie combo hold kare?", "maybe"),
        ("lettuce kal tak aayega vendor bola", "aaj kaise manage", "cabbage mix lol"),
        ("takeaway containers only 1 carton", "evening tak chalega?", "doubt"),
        ("basil count karo pasta ka load hai", "ok after lunch", "pls mat bhoolna"),
        ("salmon fillet 6 only", "86 kar dein after that?", "haan maybe"),
        ("curd + green chilli add next order", "done", "napkin rolls bhi"),
    ],
}

real_estate_triads = {
    "property_inquiry": [
        ("Tower B me 2bhk high floor kuch bacha?", "which side chahiye?", "garden if possible"),
        ("west facing villa avail kya abhi", "phase 1 ya 2?", "either chalega"),
        ("ready to move near clubhouse hai koi", "2bhk only", "haan for family"),
        ("blue brochure wala unit sold ya hold?", "hold shyd", "kis naam pe?"),
        ("3bhk + servant lower floor left?", "16 ya 21 only maybe", "18th not?"),
        ("garden facing duplex final release me hai?", "1 corner maybe", "send stack pls"),
        ("compact 1bhk under 900 sqft mil raha?", "very less", "which tower me"),
        ("lake facing balcony option chahiye client ko", "premium tower only", "inventory bachi?"),
        ("co working space wale block ke paas unit hai?", "tower c side", "acha"),
        ("corner 4bhk servant room included na?", "most yes but sheet check", "send exact unit"),
    ],
    "pricing": [
        ("1540 sqft all incl final bolo no surprises", "parking incl?", "haan sab"),
        ("floor rise + parking wala revised quote bhejo", "old sheet mat use krna", "yes pls"),
        ("1.42 cr if token today close karu?", "which unit?", "C-1104"),
        ("sir wants just headline num no breakup", "bsp only ya all in", "all in"),
        ("investor ko BSP only chahiye taxes baad me", "ok fresh sheet bhejta", "jaldi"),
        ("brochure rates old hain btw", "haan new cost sheet use", "send here"),
        ("mod kitchen offer add karke final kya banega", "clubhouse waive nahi", "phir bhi share"),
        ("2 adjacent plots bulk booking best?", "cashflow plan konsa", "plan B"),
        ("tower d pre launch rate pls asap", "channel partner wait pe", "haan"),
        ("3.5 lakh aur niche ja sakta if token now?", "maybe if no waiver", "thik last ask"),
    ],
    "site_visit": [
        ("site visit 4pm Arora family confirmed", "pickup metro gate2?", "haan"),
        ("sample flat keys kiske paas", "mine", "ok ready rakh"),
        ("kal ka visit 11:30 shift karo", "noted", "kid exam hai"),
        ("buggy bhi chahiye couple with parents", "booked", "hard hats bhi"),
        ("visit done, amenities pasand but balcony small", "next tower dikhaye?", "haan maybe"),
        ("12 noon walkin aa raha keep helmets", "who handles?", "Rahul"),
        ("NRI lead wants virtual + physical combo", "Sunday rakhte?", "works"),
        ("walkin sample villa abhi dekhna chahta", "who free?", "Meera maybe"),
        ("bus visit me 8 guests confirmed tomo", "welcome kits ready?", "3 pending"),
        ("doctor client only after 7pm aa sakega", "clubhouse lights on rakhna", "done"),
    ],
    "legal_query": [
        ("OC mil gaya ya still process me?", "legal se check karu", "haan"),
        ("registry cost old sheet me galat lag rha", "stamp duty update hua kya", "same doubt"),
        ("RERA cert pdf share karna hai kya", "share but watermark kar", "ok"),
        ("khata transfer builder karega ya buyer?", "resale wala case?", "nahi fresh"),
        ("GPA accepted for booking docs?", "booking maybe, deed nahi", "pls confirm"),
        ("mutation process kitna time bola tha", "30-45 days shyd", "shyd nahi yaar"),
        ("joint registration me dono present zaruri?", "poa ho to maybe", "ok"),
        ("maintenance deposit refundable hota?", "project specific", "which clause?"),
        ("GST under construction commercial pe lagta na", "haan mostly", "exact % bhejo"),
        ("lock in period before resale kuch hai?", "builder clause dekhna", "send para"),
    ],
    "general": [
        ("pls upload yesterday followup sheet", "done?", "abhi kar raha"),
        ("broker callback since morning waiting", "kaun lega", "Amit"),
        ("Can we block till evening w/o token?", "thoda risky", "hmm"),
        ("broker meet lounge 5:30 fyi", "ok", "be on time"),
        ("coffee machine down again", "admin ko ping kia", "haan"),
        ("badges pehen lena weekend pe", "noted", "front desk bhi"),
        ("blocked units list EOD bhejna", "ok done", "excel wala"),
        ("who has new brochure stack", "reception pe 4 bachi", "print aur karao"),
        ("cabin 3 projector not connecting", "hdmi check?", "still no"),
        ("walk in count share after lunch", "reception se leke bhejta", "👍"),
    ],
}


datasets = [
    {
        "filename": "electronics_retail.json",
        "business_name": "TechZone Electronics",
        "business_type": "electronics_retail",
        "group_name": "Seller Support Group",
        "description": "A messy WhatsApp-style seller support group for TechZone Electronics with interleaved stock checks, rate discussions, complaints, dispatch updates, and short context-heavy replies.",
        "available_labels": ["product_inquiry", "pricing", "complaint", "order_update", "general"],
        "triads": electronics_triads,
        "start_time": "2024-01-15T09:00:00",
    },
    {
        "filename": "restaurant_chain.json",
        "business_name": "Spice Garden Restaurants",
        "business_type": "restaurant_chain",
        "group_name": "Operations Team",
        "description": "A noisy WhatsApp-style operations group for Spice Garden Restaurants with interleaved menu questions, bookings, complaints, staffing chatter, and inventory follow-ups.",
        "available_labels": ["menu_query", "reservation", "complaint", "staff_update", "inventory"],
        "triads": restaurant_triads,
        "start_time": "2024-01-16T09:00:00",
    },
    {
        "filename": "real_estate.json",
        "business_name": "UrbanNest Realty",
        "business_type": "real_estate",
        "group_name": "Sales & Closures",
        "description": "A messy WhatsApp-style sales group for UrbanNest Realty with interleaved property questions, negotiations, visit coordination, legal clarifications, and short ambiguous follow-ups.",
        "available_labels": ["property_inquiry", "pricing", "site_visit", "legal_query", "general"],
        "triads": real_estate_triads,
        "start_time": "2024-01-17T09:00:00",
    },
]


def main():
    for dataset in datasets:
        output = {
            "business_name": dataset["business_name"],
            "business_type": dataset["business_type"],
            "group_name": dataset["group_name"],
            "description": dataset["description"],
            "available_labels": dataset["available_labels"],
            "messages": build_messages(
                dataset["available_labels"],
                dataset["triads"],
                dataset["start_time"],
            ),
        }

        with (BASE_DIR / dataset["filename"]).open("w", encoding="utf-8") as file_obj:
            json.dump(output, file_obj, indent=2, ensure_ascii=False)
            file_obj.write("\n")


if __name__ == "__main__":
    main()
    print("Synthetic chat datasets generated in data/.")
