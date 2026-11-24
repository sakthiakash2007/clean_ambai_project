import os, math, sqlite3, uuid
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, jsonify, g, session, redirect, url_for
BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "clean_ambai.db"); UP = os.path.join(BASE, "static", "uploads")
CENTER = (8.7107, 77.4538); RADIUS_KM = 12
# Approximate polylines (verify on ground before real deployment)
RIVER = [(8.7200,77.4250),(8.7120,77.4400),(8.7050,77.4530),(8.7000,77.4650),(8.6960,77.4780)]
HIGHWAY = [(8.7080,77.3730),(8.7107,77.4538),(8.7300,77.5200)]
LANDMARKS = [("Ambasamudram – Kallidaikurichi Thamirabarani Bridge",8.7010,77.4640),
 ("Aathu Salai / Riverfront Bathing Ghats",8.7060,77.4510),
 ("Ambasamudram Railway Station Road",8.7150,77.4580),
 ("Ambasamudram Arch / Main Bus Stand",8.7107,77.4538),
 ("Kasibanathaswami / Thirumulanathar Temple (Melapalayam St)",8.7085,77.4500),
 ("Theerthapathi HSS / Rice Research Station Area",8.7190,77.4610),
 ("Vikramasingapuram (V.K. Puram) / Papanasam Road",8.7070,77.4030),
 ("Other (Enter Manually)",None,None)]
ROUTES = {"waste":"Ambasamudram Municipality – Sanitary Inspector",
 "streetlight":"Ambasamudram Municipality – Electrical Foreman / JE",
 "transformer":"TANGEDCO Ambasamudram O&M Section"}
# ---- EDIT THESE: numbers marked "" are not verified. Fill in from the municipality office. ----
CONTACTS = {
 "municipality": {"name":"Ambasamudram Municipality – Commissioner","phone":"04634-250377","email":"commr.ambasamudram@tn.gov.in","addr":"149, Pudhugramam Street, Ambasamudram – 627401"},
 "sanitary": {"name":"Sanitary Inspector, Ambasamudram Municipality","phone":"04634-250377","email":"commr.ambasamudram@tn.gov.in","addr":"Municipal Office"},
 "panchayat": {"name":"Panchayat Officer – BDO, Ambasamudram Panchayat Union","phone":"04634-250397","mobile":"7402608463","addr":"For villages outside the municipality"},
 "vkpuram": {"name":"Vikramasingapuram Municipality","phone":"04634-220343","email":"commr.vikramasingapuram@tn.gov.in"},
 "tangedco": {"name":"TANGEDCO Minnagam","phone":"9498794987"},
 "helplines": [("Police","100"),("Fire & Rescue","101"),("Medical","104"),("District Disaster Helpline","1070")]}
def eta_for(cat, dept, prio, lvl=1):
    near = prio.startswith("HIGH")
    if cat == "waste": return ("within 24 hours" if lvl == 1 else "within 48 hours") if near else ("within 3 days" if lvl == 1 else "within 5–7 days")
    if cat == "streetlight": return "3–5 working days"
    if cat == "transformer": return "Emergency – same day (call Minnagam now)"
    return "15–30 days" if "Highways" in dept else "10–15 days"
app = Flask(__name__); app.config["MAX_CONTENT_LENGTH"] = 12*1024*1024
os.makedirs(UP, exist_ok=True)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret")
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True)
OFFICER_PASSWORD = os.environ.get("OFFICER_PASSWORD", "ambai123")  # CHANGE before going live
def officer_only(f):
    @wraps(f)
    def w(*a, **k):
        if not session.get("officer"):
            if request.path.startswith("/api/"): return jsonify(error="Login required"), 401
            return redirect(url_for("login", next=request.path))
        return f(*a, **k)
    return w

def db():
    if "db" not in g: g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
    return g.db
@app.teardown_appcontext
def close(e):
    d = g.pop("db", None)
    if d: d.close()
def init():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY AUTOINCREMENT, ticket TEXT, category TEXT,
      description TEXT, lat REAL, lng REAL, landmark TEXT, street TEXT, nearby TEXT, photo TEXT, department TEXT,
      priority TEXT, severity TEXT, ai_note TEXT, status TEXT DEFAULT 'Open', created TEXT)""")
    for col in ("waste_type","eta"):
        try: c.execute(f"ALTER TABLE reports ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError: pass
    c.commit(); c.close()

def hav(a, b):
    p = math.pi/180; dl = (b[1]-a[1])*p; dp = (b[0]-a[0])*p
    h = math.sin(dp/2)**2 + math.cos(a[0]*p)*math.cos(b[0]*p)*math.sin(dl/2)**2
    return 12742*math.asin(math.sqrt(h))
def dist_line(pt, line, steps=40):
    best = 1e9
    for a, b in zip(line, line[1:]):
        for i in range(steps+1):
            t = i/steps; best = min(best, hav(pt, (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t)))
    return best
def in_zone(lat, lng): return hav(CENTER, (lat, lng)) <= RADIUS_KM

_model = None
BIO = {"banana","apple","orange","broccoli","carrot","sandwich","pizza","donut","cake","hot dog","potted plant"}
NONBIO = {"bottle","cup","wine glass","bowl","handbag","backpack","suitcase","umbrella","cell phone","tv","chair","vase","fork","knife","spoon"}
def load_model():
    global _model
    if _model is not None: return _model or None
    try:
        from ultralytics import YOLO
        for w in ("yolo26n.pt", "yolov8n.pt"):
            try: _model = YOLO(w); return _model
            except Exception: continue
    except Exception: pass
    _model = False; return None

def color_split(path):
    """Heuristic: organic hues (green/brown/yellow, dark) vs plastic/paper/metal (bright, white, grey, blue/red)."""
    import numpy as np; from PIL import Image
    h, s, v = [np.array(x, dtype=float) for x in Image.open(path).convert("RGB").resize((120,120)).convert("HSV").split()]
    organic = ((h >= 8) & (h <= 120) & (s > 50) & (v < 215)) | ((v < 70) & (s > 30))
    synth = ((s > 110) & (v > 130) & ~((h >= 8) & (h <= 120))) | ((s < 45) & (v > 85)) | ((s > 50) & (v >= 215))
    o, n = organic.sum(), (synth & ~organic).sum()
    return 50.0 if o + n == 0 else 100.0*o/(o+n)

def triage_pile(path):
    """Returns (level, note, waste_type)."""
    cov, n, mode, names = 0.0, 0, "heuristic", []
    m = load_model()
    try:
        if m:
            r = m(path, verbose=False)[0]; W, H = r.orig_shape[1], r.orig_shape[0]
            boxes = [b for b in r.boxes if int(b.cls) != 0]
            names = [r.names[int(b.cls)] for b in boxes]
            n = len(boxes); cov = min(1, sum(float((b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1])) for b in boxes)/(W*H)); mode = "YOLO"
        import numpy as np; from PIL import Image, ImageFilter
        e = np.array(Image.open(path).convert("L").resize((160,160)).filter(ImageFilter.FIND_EDGES))
        cov = max(cov, float((e > 40).mean())*1.6); n = max(n, int(cov*30))
        bio = color_split(path) + 8*(sum(x in BIO for x in names) - sum(x in NONBIO for x in names))
        bio = max(0, min(100, bio))
    except Exception as ex: return 1, f"AI unavailable ({ex.__class__.__name__}); defaulted to Level 1", "Not classified"
    lvl = 2 if (cov > 0.35 or n >= 8) else 1
    wt = f"Mostly Bio / organic ({bio:.0f}%)" if bio >= 65 else f"Mostly Non-Bio – plastic/dry ({100-bio:.0f}%)" if bio <= 35 else f"Mixed waste (~{bio:.0f}% bio)"
    return lvl, f"{mode}: coverage {cov:.0%}, ~{n} objects", wt

@app.route("/")
def home(): return render_template("home.html", c=CONTACTS)
@app.route("/report")
def report(): return render_template("report.html", landmarks=LANDMARKS, center=CENTER, radius=RADIUS_KM, river=RIVER)
@app.route("/login", methods=["GET", "POST"])
def login():
    err = None
    if request.method == "POST":
        if request.form.get("password") == OFFICER_PASSWORD:
            session["officer"] = True; nxt = request.args.get("next", "/officer")
            return redirect(nxt if nxt.startswith("/") else "/officer")
        err = "Wrong password"
    return render_template("login.html", err=err)
@app.route("/logout")
def logout(): session.clear(); return redirect("/")
@app.route("/check")
@officer_only
def check(): return render_template("check.html", landmarks=[l for l in LANDMARKS if l[1]], river=RIVER, highway=HIGHWAY, center=CENTER)
@app.route("/officer")
@officer_only
def officer():
    rows = db().execute("SELECT * FROM reports ORDER BY (status='Resolved'), priority LIKE 'HIGH%' DESC, id DESC").fetchall()
    depts = {}
    for r in rows: depts.setdefault(r["department"], []).append(r)
    return render_template("officer.html", depts=depts)

@app.route("/api/geocheck")
def geocheck():
    lat, lng = float(request.args["lat"]), float(request.args["lng"])
    return jsonify(in_zone=in_zone(lat, lng), km=round(hav(CENTER, (lat, lng)), 1), river_m=int(dist_line((lat, lng), RIVER)*1000))
@app.route("/api/reports")
def reports():
    return jsonify([dict(r) for r in db().execute("SELECT * FROM reports ORDER BY id DESC LIMIT 300")])
@app.route("/api/report", methods=["POST"])
def submit():
    f = request.form; cat = f.get("category")
    try: lat, lng = float(f["lat"]), float(f["lng"])
    except Exception: return jsonify(error="Please set a location on the map."), 400
    if cat not in ("waste","streetlight","road","transformer"): return jsonify(error="Choose an issue type."), 400
    if not f.get("street", "").strip(): return jsonify(error="Street / address is required."), 400
    photo = None; fl = request.files.get("photo")
    if fl and fl.filename:
        photo = uuid.uuid4().hex + os.path.splitext(fl.filename)[1].lower()[:5]; fl.save(os.path.join(UP, photo))
    river_km = dist_line((lat, lng), RIVER); sev = ai = wt = None; lvl = 1; prio = "STANDARD (3-7 Days)"
    if cat == "waste":
        near = river_km <= 0.5
        prio = "HIGH PRIORITY (24-48h)" if near else "STANDARD (3-7 Days)"
        lvl, ai, wt = triage_pile(os.path.join(UP, photo)) if photo else (1, "No photo – defaulted to Level 1", "Not classified (no photo)")
        sev = "Level 1 - Pushcart (1 Worker)" if lvl == 1 else "Level 2 - Tipper Lorry / Tractor (Bulk Dump)"
        dept = ROUTES["waste"]
    elif cat == "road":
        hw = dist_line((lat, lng), HIGHWAY) <= 0.3
        dept = "State Highways AE" if hw else "Ambasamudram Municipality – Engineering Wing"
    else:
        dept = ROUTES[cat]
        if cat == "transformer": prio = "EMERGENCY – call TANGEDCO"
    eta = eta_for(cat, dept, prio, lvl)
    ticket = "AMB-" + datetime.now().strftime("%y%m%d") + "-" + uuid.uuid4().hex[:4].upper()
    db().execute("INSERT INTO reports(ticket,category,description,lat,lng,landmark,street,nearby,photo,department,priority,severity,ai_note,created,waste_type,eta) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ticket, cat, f.get("description",""), lat, lng, f.get("landmark",""), f["street"], f.get("nearby",""), photo, dept, prio, sev, ai, datetime.now().strftime("%Y-%m-%d %H:%M"), wt, eta))
    db().commit()
    return jsonify(waste_type=wt, eta=eta, ticket=ticket, department=dept, priority=prio, severity=sev, ai_note=ai, in_zone=in_zone(lat, lng))
@app.route("/api/status", methods=["POST"])
@officer_only
def status():
    db().execute("UPDATE reports SET status=? WHERE id=?", (request.form["status"], request.form["id"])); db().commit()
    return jsonify(ok=True)

init()
if __name__ == "__main__": app.run(host="0.0.0.0", port=5000, debug=True)
