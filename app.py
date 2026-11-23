import os
import io
import threading
import sqlite3
import datetime
import json
from flask import Flask, render_template, request, jsonify, send_file, abort
from model import train_model_background, extract_embedding_for_image, MODEL_PATH

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- In-memory attendance session ----------
_session_lock = threading.Lock()
_session_active = False
_session_seen: dict = {}

# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    roll TEXT,
                    class TEXT,
                    section TEXT,
                    reg_no TEXT,
                    created_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    name TEXT,
                    timestamp TEXT
                )""")
    conn.commit()
    conn.close()

init_db()

# ---------- Train status helpers ----------
def write_train_status(status_dict):
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        return json.load(f)

write_train_status({"running": False, "progress": 0, "message": "No training yet."})

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT timestamp FROM attendance", conn)
    conn.close()
    if df.empty:
        days = [(datetime.date.today() - datetime.timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
        return jsonify({"dates": days, "counts": [0]*30})
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    last_30 = [(datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1)]
    counts = [int(df[df['date'] == d].shape[0]) for d in last_30]
    dates = [d.strftime("%d-%b") for d in last_30]
    return jsonify({"dates": dates, "counts": counts})

@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")
    data = request.form
    name = data.get("name", "").strip()
    roll = data.get("roll", "").strip()
    cls = data.get("class", "").strip()
    sec = data.get("sec", "").strip()
    reg_no = data.get("reg_no", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO students (name, roll, class, section, reg_no, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              (name, roll, cls, sec, reg_no, now))
    sid = c.lastrowid
    conn.commit()
    conn.close()
    os.makedirs(os.path.join(DATASET_DIR, str(sid)), exist_ok=True)
    return jsonify({"student_id": sid})

@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error": "student_id required"}), 400
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    for f in files:
        try:
            fname = f"{datetime.datetime.utcnow().timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
            saved += 1
        except Exception as e:
            app.logger.error("save error: %s", e)
    return jsonify({"saved": saved})

@app.route("/train_model", methods=["GET"])
def train_model_route():
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status": "already_running"}), 202
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    t = threading.Thread(
        target=train_model_background,
        args=(DATASET_DIR, lambda p, m: write_train_status({"running": True, "progress": p, "message": m}))
    )
    t.daemon = True
    t.start()
    return jsonify({"status": "started"}), 202

@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    return render_template("mark_attendance.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    global _session_active, _session_seen
    with _session_lock:
        _session_active = True
        _session_seen = {}
    return jsonify({"status": "session_started"})

@app.route("/end_session", methods=["POST"])
def end_session():
    global _session_active, _session_seen
    with _session_lock:
        if not _session_active:
            return jsonify({"status": "no_active_session", "saved": 0})
        students_to_save = dict(_session_seen)
        _session_active = False
        _session_seen = {}

    if not students_to_save:
        return jsonify({"status": "session_ended", "saved": 0, "students": []})

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    saved_names = []
    for sid, info in students_to_save.items():
        today = datetime.date.today().isoformat()
        c.execute("SELECT id FROM attendance WHERE student_id=? AND date(timestamp)=?", (sid, today))
        if c.fetchone() is None:
            c.execute("INSERT INTO attendance (student_id, name, timestamp) VALUES (?, ?, ?)",
                      (sid, info["name"], info["first_seen"]))
            saved_names.append(info["name"])
    conn.commit()
    conn.close()

    return jsonify({"status": "session_ended", "saved": len(saved_names), "students": saved_names})

@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    global _session_active, _session_seen
    with _session_lock:
        if not _session_active:
            return jsonify({"recognized": False, "error": "no_active_session"}), 200

    if "image" not in request.files:
        return jsonify({"recognized": False, "error": "no image"}), 400

    img_file = request.files["image"]
    try:
        emb = extract_embedding_for_image(img_file.stream)
        if emb is None:
            return jsonify({"recognized": False, "error": "no face detected"}), 200

        from model import load_model_if_exists, predict_with_model
        clf = load_model_if_exists()
        if clf is None:
            return jsonify({"recognized": False, "error": "model not trained"}), 200

        pred_label, conf = predict_with_model(clf, emb)
        if conf < 0.5:
            return jsonify({"recognized": False, "confidence": float(conf)}), 200

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM students WHERE id=?", (int(pred_label),))
        row = c.fetchone()
        conn.close()
        name = row[0] if row else "Unknown"
        sid = int(pred_label)

        with _session_lock:
            if _session_active and sid not in _session_seen:
                egypt_tz = datetime.timezone(datetime.timedelta(hours=2))
                _session_seen[sid] = {"name": name, "first_seen": datetime.datetime.now(egypt_tz).isoformat()}
            already_seen = sid in _session_seen

        return jsonify({
            "recognized": True, "student_id": sid, "name": name,
            "confidence": float(conf), "already_logged": already_seen
        }), 200

    except Exception as e:
        app.logger.exception("recognize error")
        return jsonify({"recognized": False, "error": str(e)}), 500

@app.route("/session_status", methods=["GET"])
def session_status():
    with _session_lock:
        return jsonify({
            "active": _session_active,
            "seen_count": len(_session_seen),
            "students": [{"student_id": sid, "name": info["name"], "first_seen": info["first_seen"]}
                         for sid, info in _session_seen.items()]
        })

@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    period = request.args.get("period", "all")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    q = "SELECT id, student_id, name, timestamp FROM attendance"
    params = ()
    if period == "daily":
        today = datetime.date.today().isoformat()
        q += " WHERE date(timestamp) = ?"
        params = (today,)
    elif period == "weekly":
        start = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        q += " WHERE date(timestamp) >= ?"
        params = (start,)
    elif period == "monthly":
        start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        q += " WHERE date(timestamp) >= ?"
        params = (start,)
    q += " ORDER BY timestamp DESC LIMIT 5000"
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    return render_template("attendance_record.html", records=rows, period=period)

@app.route("/download_csv", methods=["GET"])
def download_csv():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, student_id, name, timestamp FROM attendance ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    output = io.StringIO()
    output.write("id,student_id,name,timestamp\n")
    for r in rows:
        output.write(f'{r[0]},{r[1]},{r[2]},{r[3]}\n')
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="attendance.csv", mimetype="text/csv")

@app.route("/students", methods=["GET"])
def students_list():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, roll, class, section, reg_no, created_at FROM students ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    data = [{"id": r[0], "name": r[1], "roll": r[2], "class": r[3],
             "section": r[4], "reg_no": r[5], "created_at": r[6]} for r in rows]
    return jsonify({"students": data})

@app.route("/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE id=?", (sid,))
    c.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    conn.commit()
    conn.close()
    folder = os.path.join(DATASET_DIR, str(sid))
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    return jsonify({"deleted": True})

# -------- NEW: Students listing page --------
@app.route("/students_page", methods=["GET"])
def students_page():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, roll, class, section, reg_no, created_at FROM students ORDER BY name ASC")
    rows = c.fetchall()
    students = []
    for r in rows:
        sid = r[0]
        c.execute("SELECT COUNT(*) FROM attendance WHERE student_id=?", (sid,))
        count = c.fetchone()[0]
        c.execute("SELECT timestamp FROM attendance WHERE student_id=? ORDER BY timestamp DESC LIMIT 1", (sid,))
        last = c.fetchone()
        last_seen = last[0][:10] if last else None
        students.append({
            "id": sid, "name": r[1], "roll": r[2], "class": r[3],
            "section": r[4], "reg_no": r[5], "created_at": r[6],
            "total_attendance": count, "last_seen": last_seen
        })
    conn.close()
    return render_template("students_page.html", students=students)

# -------- NEW: Individual student profile page --------
@app.route("/student_profile/<int:sid>", methods=["GET"])
def student_profile(sid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, roll, class, section, reg_no, created_at FROM students WHERE id=?", (sid,))
    row = c.fetchone()
    if not row:
        conn.close()
        abort(404)
    student = {
        "id": row[0], "name": row[1], "roll": row[2],
        "class": row[3], "section": row[4], "reg_no": row[5], "created_at": row[6]
    }
    c.execute("SELECT timestamp FROM attendance WHERE student_id=? ORDER BY timestamp DESC", (sid,))
    att_rows = c.fetchall()
    conn.close()
    attendance_dates = [r[0] for r in att_rows]
    return render_template("student_profile.html", student=student, attendance_dates=attendance_dates)

# -------- NEW: Per-student attendance stats API --------
@app.route("/student_stats/<int:sid>", methods=["GET"])
def student_stats(sid):
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT timestamp FROM attendance WHERE student_id=?", conn, params=(sid,))
    conn.close()

    last_30 = [(datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1)]
    dates = [d.strftime("%d-%b") for d in last_30]

    if df.empty:
        return jsonify({"dates": dates, "present": [0]*30, "total": 0, "streak": 0})

    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    present = [1 if d in df['date'].values else 0 for d in last_30]

    streak = 0
    for i in range(len(last_30)):
        if present[-(i+1)] == 1:
            streak += 1
        else:
            break

    return jsonify({"dates": dates, "present": present, "total": int(df.shape[0]), "streak": streak})

if __name__ == "__main__":
    app.run(debug=True)
