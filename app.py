from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

import threading
import time
from collections import defaultdict
from flask import request, jsonify
import requests
import base64
import os
from dotenv import load_dotenv

# ─── Load credentials from .env ────────────────────────────────────────────────
load_dotenv()
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")


app = Flask(__name__)

# ─── Load models ───────────────────────────────────────────────────────────────
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ─── Shared camera state ───────────────────────────────────────────────────────
camera      = None
camera_lock = threading.Lock()

# latest_frame is written by generate_frames() and read by the scanner.
# Using a simple lock avoids double-opening the camera device.
latest_frame      = None
latest_frame_lock = threading.Lock()

# ─── Spotify token cache ───────────────────────────────────────────────────────
spotify_token      = None
spotify_token_time = 0
selected_language  = "english"   # default language


def get_spotify_access_token():
    global spotify_token, spotify_token_time

    if spotify_token and (time.time() - spotify_token_time < 3600):
        return spotify_token

    auth_str    = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    url     = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {b64_auth_str}",
        "Content-Type":  "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()

    spotify_token      = response.json()["access_token"]
    spotify_token_time = time.time()
    return spotify_token


# ─── Curated song database (no API needed, no duplicates) ─────────────────────
# Format: { language: { emotion: [ {title, artist}, ... ] } }
# Spotify search URLs are auto-built from title+artist in get_songs_by_emotion()

SONG_DB = {
    "english": {
        "Happy": [
            {"title": "Happy",                    "artist": "Pharrell Williams"},
            {"title": "Can't Stop the Feeling!",  "artist": "Justin Timberlake"},
            {"title": "Uptown Funk",               "artist": "Mark Ronson ft. Bruno Mars"},
            {"title": "Good as Hell",              "artist": "Lizzo"},
            {"title": "Shake It Off",              "artist": "Taylor Swift"},
            {"title": "Levitating",                "artist": "Dua Lipa"},
            {"title": "Blinding Lights",           "artist": "The Weeknd"},
            {"title": "As It Was",                 "artist": "Harry Styles"},
        ],
        "Sad": [
            {"title": "Someone Like You",          "artist": "Adele"},
            {"title": "Someone You Loved",         "artist": "Lewis Capaldi"},
            {"title": "The Night We Met",          "artist": "Lord Huron"},
            {"title": "Stay With Me",              "artist": "Sam Smith"},
            {"title": "All Too Well",              "artist": "Taylor Swift"},
            {"title": "Let Her Go",                "artist": "Passenger"},
            {"title": "Fix You",                   "artist": "Coldplay"},
            {"title": "When the Party's Over",     "artist": "Billie Eilish"},
        ],
        "Angry": [
            {"title": "In The End",                "artist": "Linkin Park"},
            {"title": "Killing in the Name",       "artist": "Rage Against the Machine"},
            {"title": "Chop Suey!",                "artist": "System of a Down"},
            {"title": "Smells Like Teen Spirit",   "artist": "Nirvana"},
            {"title": "Break Stuff",               "artist": "Limp Bizkit"},
            {"title": "Given Up",                  "artist": "Linkin Park"},
            {"title": "Bulls on Parade",           "artist": "Rage Against the Machine"},
            {"title": "Bodies",                    "artist": "Drowning Pool"},
        ],
        "Fear": [
            {"title": "Mad World",                 "artist": "Gary Jules"},
            {"title": "Sound of Silence",          "artist": "Simon & Garfunkel"},
            {"title": "Breathe Me",                "artist": "Sia"},
            {"title": "Hurt",                      "artist": "Johnny Cash"},
            {"title": "In the Air Tonight",        "artist": "Phil Collins"},
            {"title": "Say Something",             "artist": "A Great Big World"},
            {"title": "Skinny Love",               "artist": "Bon Iver"},
            {"title": "Exile",                     "artist": "Taylor Swift ft. Bon Iver"},
        ],
        "Disgust": [
            {"title": "Creep",                     "artist": "Radiohead"},
            {"title": "Hurt",                      "artist": "Nine Inch Nails"},
            {"title": "Boulevard of Broken Dreams","artist": "Green Day"},
            {"title": "Numb",                      "artist": "Linkin Park"},
            {"title": "My Immortal",               "artist": "Evanescence"},
            {"title": "The Sound of Silence",      "artist": "Disturbed"},
            {"title": "Black",                     "artist": "Pearl Jam"},
            {"title": "Behind Blue Eyes",          "artist": "Limp Bizkit"},
        ],
        "Neutral": [
            {"title": "Weightless",                "artist": "Marconi Union"},
            {"title": "Coffee",                    "artist": "beabadoobee"},
            {"title": "Electric Feel",             "artist": "MGMT"},
            {"title": "Motion Picture Soundtrack", "artist": "Radiohead"},
            {"title": "Float On",                  "artist": "Modest Mouse"},
            {"title": "Holocene",                  "artist": "Bon Iver"},
            {"title": "Lost in Translation",       "artist": "Phoenix"},
            {"title": "Be Still",                  "artist": "The Killers"},
        ],
        "Surprise": [
            {"title": "Don't Stop Me Now",         "artist": "Queen"},
            {"title": "Mr. Brightside",            "artist": "The Killers"},
            {"title": "Sweet Child O' Mine",       "artist": "Guns N' Roses"},
            {"title": "Bohemian Rhapsody",         "artist": "Queen"},
            {"title": "Take On Me",                "artist": "a-ha"},
            {"title": "Jump",                      "artist": "Van Halen"},
            {"title": "Living on a Prayer",        "artist": "Bon Jovi"},
            {"title": "Dynamite",                  "artist": "BTS"},
        ],
    },
    "hindi": {
        "Happy": [
            {"title": "Badtameez Dil",             "artist": "Pritam"},
            {"title": "Gallan Goodiyaan",          "artist": "Shankar Ehsaan Loy"},
            {"title": "London Thumakda",           "artist": "Amit Trivedi"},
            {"title": "Balam Pichkari",            "artist": "Vishal Shekhar"},
            {"title": "Tune Maari Entriyaan",      "artist": "Vishal Shekhar"},
            {"title": "Senorita",                  "artist": "Shankar Ehsaan Loy"},
            {"title": "Desi Girl",                 "artist": "Vishal Shekhar"},
            {"title": "Kar Gayi Chull",            "artist": "Badshah"},
        ],
        "Sad": [
            {"title": "Channa Mereya",             "artist": "Pritam ft. Arijit Singh"},
            {"title": "Tujhe Bhula Diya",          "artist": "Shafqat Amanat Ali"},
            {"title": "Kabira",                    "artist": "Pritam ft. Rekha Bhardwaj"},
            {"title": "Ae Dil Hai Mushkil",        "artist": "Pritam ft. Arijit Singh"},
            {"title": "Muskurane",                 "artist": "Arijit Singh"},
            {"title": "Ilahi",                     "artist": "Pritam ft. Arijit Singh"},
            {"title": "Teri Meri",                 "artist": "Himesh Reshammiya"},
            {"title": "Phir Bhi Tumko Chahunga",   "artist": "Mithoon"},
        ],
        "Angry": [
            {"title": "Emosanal Attyachaar",       "artist": "Ram Sampath"},
            {"title": "Dhoom Machale",             "artist": "Sunidhi Chauhan"},
            {"title": "Ziddi Dil",                 "artist": "Palak Muchhal"},
            {"title": "Rang De Basanti",           "artist": "AR Rahman"},
            {"title": "Rock the Party",            "artist": "Vishal Dadlani"},
            {"title": "Bhaag DK Bose",             "artist": "Ram Sampath"},
            {"title": "Sadda Haq",                 "artist": "AR Rahman"},
            {"title": "Sultan",                    "artist": "Vishal Shekhar"},
        ],
        "Fear": [
            {"title": "Tere Bina",                 "artist": "AR Rahman"},
            {"title": "Lag Ja Gale",               "artist": "Lata Mangeshkar"},
            {"title": "Pehla Nasha",               "artist": "Udit Narayan"},
            {"title": "Soch Na Sake",              "artist": "Arijit Singh"},
            {"title": "Aye Mere Humsafar",         "artist": "Udit Narayan"},
            {"title": "Tujhse Naraz Nahi Zindagi", "artist": "Lata Mangeshkar"},
            {"title": "Woh Ladki Jo",              "artist": "KK"},
            {"title": "Dil Se Re",                 "artist": "AR Rahman"},
        ],
        "Disgust": [
            {"title": "Tum Hi Ho",                 "artist": "Arijit Singh"},
            {"title": "Teri Galliyan",             "artist": "Ankit Tiwari"},
            {"title": "Saans",                     "artist": "Shreya Ghoshal"},
            {"title": "Raabta",                    "artist": "Pritam"},
            {"title": "Jeena Jeena",               "artist": "Atif Aslam"},
            {"title": "Kya Hua Tera Wada",         "artist": "Mohammed Rafi"},
            {"title": "Na Ja",                     "artist": "Pav Dharia"},
            {"title": "Judaai",                    "artist": "Rekha Bhardwaj"},
        ],
        "Neutral": [
            {"title": "Iktara",                    "artist": "Amit Trivedi"},
            {"title": "Tu Jaane Na",               "artist": "Atif Aslam"},
            {"title": "Khairiyat",                 "artist": "Arijit Singh"},
            {"title": "Safar",                     "artist": "Pritam ft. Arijit Singh"},
            {"title": "Dil Diyan Gallan",          "artist": "Atif Aslam"},
            {"title": "Yeh Fitoor Mera",           "artist": "Arijit Singh"},
            {"title": "Koi Fariyaad",              "artist": "Jagjit Singh"},
            {"title": "O Re Piya",                 "artist": "Rahat Fateh Ali Khan"},
        ],
        "Surprise": [
            {"title": "Aankh Marey",               "artist": "Kumar Sanu & Kavita Krishnamurthy"},
            {"title": "Morni Banke",               "artist": "Tanishk Bagchi"},
            {"title": "Jai Ho",                    "artist": "AR Rahman"},
            {"title": "Swag Se Swagat",            "artist": "Vishal Shekhar"},
            {"title": "Lungi Dance",               "artist": "Yo Yo Honey Singh"},
            {"title": "Party All Night",           "artist": "Yo Yo Honey Singh"},
            {"title": "Desi Beat",                 "artist": "Yo Yo Honey Singh"},
            {"title": "Nachde Ne Saare",           "artist": "Pritam"},
        ],
    },
    "kannada": {
        "Happy": [
            {"title": "Cotton Candy",              "artist": "Chandan Shetty"},
            {"title": "Ee Hrudaya Thumbidhe",      "artist": "Mungaru Male"},
            {"title": "Nee Beda Nee Beda",         "artist": "Gaalipata"},
            {"title": "Endendu Mareyalare",        "artist": "Rajkumar"},
            {"title": "Koogadali",                 "artist": "Rajkumar"},
            {"title": "Bombe Heluthaite",          "artist": "Mungaru Male"},
            {"title": "Cheluvina Chittara",        "artist": "Cheluvina Chittara"},
            {"title": "Thooduve Manashu",          "artist": "Gaalipata"},
        ],
        "Sad": [
            {"title": "Singara Siriye",            "artist": "Vijay Prakash"},
            {"title": "Ninna Nenapu",              "artist": "Mungaru Male"},
            {"title": "Manase Manase",             "artist": "Rajkumar"},
            {"title": "Bombe Heluthaite",          "artist": "Mungaru Male"},
            {"title": "Ee Preethiya Hoovugalu",    "artist": "Rajkumar"},
            {"title": "Nee Madhu Kudidalare",      "artist": "Rajkumar"},
            {"title": "Kaadidhe Kaadidhe",         "artist": "Rajkumar"},
            {"title": "Kavithe Kavithe",           "artist": "Rajkumar"},
        ],
        "Angry": [
            {"title": "Rocky Walks Up the Stairs", "artist": "Ravi Basrur"},
            {"title": "Salaam Rocky Bhai",         "artist": "Vijay Prakash"},
            {"title": "Toofan",                    "artist": "Ravi Basrur"},
            {"title": "KGF Theme",                 "artist": "Ravi Basrur"},
            {"title": "Garbha",                    "artist": "Ravi Basrur"},
            {"title": "Yeyyo",                     "artist": "Ravi Basrur"},
            {"title": "Kantara Theme",             "artist": "B Ajaneesh Loknath"},
            {"title": "Kolar Gold Fields",         "artist": "Ravi Basrur"},
        ],
        "Fear": [
            {"title": "Varaha Roopam",             "artist": "Vijay Prakash"},
            {"title": "Tony's Mayhem",             "artist": "Ravi Basrur"},
            {"title": "Huttidare Kannada Naadalli","artist": "Rajkumar"},
            {"title": "Chandana Mandara",          "artist": "Rajkumar"},
            {"title": "Gowri Ganesha",             "artist": "Kannada Devotional"},
            {"title": "Aa Kanasina Hakkige",       "artist": "Rajkumar"},
            {"title": "Nodi Swami Navirodu Hige",  "artist": "Rajkumar"},
            {"title": "Yennagide",                 "artist": "Rajkumar"},
        ],
        "Disgust": [
            {"title": "Yaare Koogadali",           "artist": "Rajkumar"},
            {"title": "Hoo Mallige",               "artist": "Rajkumar"},
            {"title": "Cheluve Cheluve",           "artist": "Rajkumar"},
            {"title": "Ee Madhura Venuve",         "artist": "Rajkumar"},
            {"title": "Entha Preetiya Kodli",      "artist": "Rajkumar"},
            {"title": "Hendathi Antha Karedare",   "artist": "Rajkumar"},
            {"title": "Ondu Hoo Thumbiro",         "artist": "Rajkumar"},
            {"title": "Naa Ninna Bidalaare",       "artist": "Rajkumar"},
        ],
        "Neutral": [
            {"title": "Jotheyali",                 "artist": "Rajkumar"},
            {"title": "Ondagi Nodu",               "artist": "Rajkumar"},
            {"title": "Moodalmare",                "artist": "Rajkumar"},
            {"title": "Aa Bhoomi",                 "artist": "Rajkumar"},
            {"title": "Kelu Jana Muddu Rama",      "artist": "Rajkumar"},
            {"title": "Ninna Nambide",             "artist": "Rajkumar"},
            {"title": "Naanu Nanna Hendthi",       "artist": "Rajkumar"},
            {"title": "Kaadu Kudure",              "artist": "Rajkumar"},
        ],
        "Surprise": [
            {"title": "Party Freak",               "artist": "Chandan Shetty"},
            {"title": "Namma Namma",               "artist": "Yuva Rajkumar"},
            {"title": "Gaalipata",                 "artist": "Gaalipata"},
            {"title": "Govinda Govinda",           "artist": "Rajkumar"},
            {"title": "Ee Sala Cup Namde",         "artist": "Kannada Fans"},
            {"title": "Aadade Aadade",             "artist": "Gaalipata"},
            {"title": "Thede Thede",               "artist": "Gaalipata"},
            {"title": "Lagna Patrika",             "artist": "Chandan Shetty"},
        ],
    }
}


def get_songs_by_emotion(emotion, limit=8):
    """
    Returns curated songs from the local database — no API for duplicates, but we
    fetch preview URLs dynamically from iTunes Search API so users get the 30-sec play feature.
    """
    import urllib.parse
    lang_db  = SONG_DB.get(selected_language, SONG_DB["english"])
    raw_list = lang_db.get(emotion, lang_db.get("Neutral", []))
    songs    = []
    for s in raw_list[:limit]:
        title  = s["title"]
        artist = s["artist"]
        
        # Fetch 30-sec preview from iTunes API
        preview_url = ""
        track_url = ""
        try:
            itunes_url = "https://itunes.apple.com/search"
            params = {
                "term": f"{title} {artist}",
                "entity": "song",
                "limit": 1
            }
            res = requests.get(itunes_url, params=params, timeout=3)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    preview_url = results[0].get("previewUrl", "")
                    track_url = results[0].get("trackViewUrl", "")
        except Exception:
            pass

        songs.append({
            "title":          title,
            "artist":         artist,
            "preview_url":    preview_url,
            "url":            track_url,
            "spotify_search": (
                "https://open.spotify.com/search/"
                + urllib.parse.quote(f"{title} {artist}")
            ),
            "youtube_search": (
                "https://www.youtube.com/results?search_query="
                + urllib.parse.quote(f"{title} {artist}")
            ),
        })
    print(f"DB: returning {len(songs)} songs for {selected_language}/{emotion} with previews")
    return songs


# ─── Global scan state ─────────────────────────────────────────────────────────
scan_active = False
scan_data   = {
    'emotion_counts':      defaultdict(int),
    'total_frames':        0,
    'emotion_percentages': {},
    'recommended_songs':   [],
    'dominant_emotion':    None,
    'scan_progress':       0,
    'live_emotions':       []
}


# ─── Video generator ───────────────────────────────────────────────────────────
def generate_frames():
    """
    Continuously reads from the camera, draws face boxes,
    stores each frame into latest_frame for the scanner to read,
    and yields MJPEG bytes for the /video_feed route.
    """
    global camera, camera_lock, latest_frame, latest_frame_lock

    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # reduce internal buffer to keep latency low
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        with camera_lock:
            if camera is None:
                break
            ret, frame = camera.read()

        if not ret:
            time.sleep(0.05)
            continue

        # Mirror
        frame = cv2.flip(frame, 1)

        # Store a raw (un-annotated) copy for the scanner
        with latest_frame_lock:
            latest_frame = frame.copy()



        ret2, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret2:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ─── Scanner thread ────────────────────────────────────────────────────────────
def detect_emotions_15sec():
    """
    Reads frames from the shared latest_frame (written by generate_frames).
    This avoids camera contention and keeps the video stream smooth.
    Real-time emotion_percentages are updated inside the loop.
    """
    global scan_active, scan_data, latest_frame, latest_frame_lock

    scan_duration = 5          # seconds
    start_time    = time.time()

    scan_data = {
        'emotion_counts':      defaultdict(int),
        'total_frames':        0,
        'emotion_percentages': {e: 0.0 for e in emotion_labels},
        'recommended_songs':   [],
        'dominant_emotion':    None,
        'scan_progress':       0,
        'live_emotions':       []
    }

    while scan_active:
        elapsed = time.time() - start_time

        if elapsed >= scan_duration:
            break

        # ── Grab latest video frame ───────────────────────────────────────────
        with latest_frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()

        # ── Progress ──────────────────────────────────────────────────────────
        scan_data['scan_progress'] = int((elapsed / scan_duration) * 100)

        # ── Detect faces & predict emotion ────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.2, 4, minSize=(60, 60))

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = cv2.equalizeHist(roi)
            roi = roi / 255.0
            roi = roi.reshape(1, 48, 48, 1)

            pred       = classifier.predict(roi, verbose=0)[0]
            idx        = pred.argmax()
            emotion    = emotion_labels[idx]
            confidence = pred[idx] * 100

            if confidence > 5:
                scan_data['emotion_counts'][emotion] += 1
                scan_data['total_frames']            += 1

                # ── Real-time percentage update ───────────────────────────────
                total = scan_data['total_frames']
                for emo, cnt in scan_data['emotion_counts'].items():
                    scan_data['emotion_percentages'][emo] = float(round(
                        (cnt / total) * 100, 1
                    ))

                # Track live emotions (last 10)
                scan_data['live_emotions'].append({
                    'emotion':    emotion,
                    'confidence': float(round(confidence, 1))
                })
                if len(scan_data['live_emotions']) > 10:
                    scan_data['live_emotions'].pop(0)

        # ── Throttle to avoid CPU spikes ──────────────────────────────────────
        time.sleep(0.05)

    # ── Finalise ──────────────────────────────────────────────────────────────
    if scan_data['total_frames'] > 0:
        dominant = max(scan_data['emotion_counts'],
                       key=scan_data['emotion_counts'].get)
        scan_data['dominant_emotion']  = dominant
        scan_data['recommended_songs'] = get_songs_by_emotion(dominant)
    else:
        scan_data['dominant_emotion'] = None

    scan_data['scan_progress'] = 100
    scan_active = False


# ─── Flask Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scan')
def scan_page():
    return render_template('scan.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/start_scan', methods=['POST'])
def start_scan():
    global scan_active, selected_language

    if scan_active:
        return jsonify({'status': 'already_running'})

    data              = request.get_json(silent=True) or {}
    selected_language = data.get("language", "english")

    scan_active = True
    threading.Thread(target=detect_emotions_15sec, daemon=True).start()

    return jsonify({'status': 'started'})


@app.route('/stop_scan', methods=['POST'])
def stop_scan():
    global scan_active
    scan_active = False
    return jsonify({'status': 'Scan stopped'})


@app.route('/get_scan_data', methods=['GET'])
def get_scan_data():
    return jsonify(scan_data)


@app.route('/is_scanning', methods=['GET'])
def is_scanning():
    return jsonify({'scanning': bool(scan_active)})


@app.route('/reset_data', methods=['POST'])
def reset_data():
    global scan_data, scan_active
    scan_active = False
    scan_data   = {
        'emotion_counts':      defaultdict(int),
        'total_frames':        0,
        'emotion_percentages': {e: 0.0 for e in emotion_labels},
        'recommended_songs':   [],
        'dominant_emotion':    None,
        'scan_progress':       0,
        'live_emotions':       []
    }
    return jsonify({'status': 'Data reset'})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
