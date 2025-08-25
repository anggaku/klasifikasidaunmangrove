from flask import Flask, render_template, request, redirect, url_for, session, abort
import os
import math
import cv2
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

# --- (opsional) kalau modul ini dipakai di ekstraksi fitur
# from skimage.feature import graycomatrix, graycoprops

# Toleran: modul bisa bernama ektraksi.py atau ekstraksi.py
try:
    from ektraksi import extract_features
except Exception:
    try:
        from ekstraksi import extract_features
    except Exception as e:
        raise ImportError("Tidak dapat menemukan fungsi 'extract_features' di ektraksi/ekstraksi.py") from e

from tensorflow.keras.models import load_model

# =========================================================
# Path & App Config (pakai path absolut supaya aman di Render)
# =========================================================
APP_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_ROOT / "templates"
STATIC_DIR = APP_ROOT / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
PROCESSED_DIR = STATIC_DIR / "processed"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = "your_secret_key_here"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# =========================================================
# Mapping kode prediksi -> endpoint species
# =========================================================
PREDICTION_MAPPING = {
    "Am": ("avicennia-marina", "avicennia_marina"),
    "Ao": ("avicennia-officinalis", "avicennia_officinalis"),
    "Bg": ("bruguiera-gymnorhiza", "bruguiera_detail"),
    "Hl": ("heritiera-littoralis", "heritiera_littoralis"),
    "Lt+": ("lumnitzera-littorea", "lumnitzera_littorea"),
    "Ra": ("rhizophora-apiculata", "rhizophora_apiculata"),
    "Sa": ("sonneratia-alba", "sonneratia_alba"),
}

# =========================================================
# Load artefak ML (saat modul di-import oleh Gunicorn)
# =========================================================
model = None
scaler = None
le = None

def load_artifacts():
    global model, scaler, le
    model_path = APP_ROOT / "best_model_compressed.h5"
    scaler_path = APP_ROOT / "scaler.pkl"
    le_path = APP_ROOT / "label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not le_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {le_path}")

    model = load_model(str(model_path), compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    if scaler is None:
        raise RuntimeError("Loaded scaler is None")
    if le is None:
        raise RuntimeError("Loaded label encoder is None")

# panggil sekarang agar tersedia saat app dijalankan via gunicorn
load_artifacts()

# =========================================================
# Utils
# =========================================================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================================================
# Routes: Pages dasar
# =========================================================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/faq")
def contact():
    return render_template("faq.html")

# =========================================================
# Flow deteksi
# =========================================================
@app.route("/detect")
def detect_home():
    return render_template("page1_upload.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    # Pastikan field name di form adalah 'file'
    if "file" not in request.files:
        return redirect(url_for("detect_home"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("detect_home"))

    if file and allowed_file(file.filename):
        try:
            # Nama file berbasis timestamp supaya aman
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}.jpg"
            filepath = str(UPLOAD_DIR / safe_filename)

            # Simpan file
            file.save(filepath)
            if not os.path.exists(filepath):
                raise RuntimeError("File failed to save")

            # Ekstraksi fitur dari file
            feats = extract_features(filepath)
            if feats is None:
                # kalau gagal ekstraksi, kembali ke halaman upload
                return render_template("page1_upload.html", error="Ekstraksi fitur gagal. Coba gambar lain.")

            # Simpan ke sesi
            session["image_path"] = filepath
            session["image_url"] = f"/static/uploads/{safe_filename}"
            try:
                session["features"] = feats.tolist()
            except Exception:
                session["features"] = np.asarray(feats, dtype=float).tolist()

            return redirect(url_for("confirmation"))

        except Exception as e:
            app.logger.exception("Error at /upload")
            return render_template("page1_upload.html", error=f"Terjadi error: {e}")

    # Jika tidak lolos allowed_file
    return redirect(url_for("detect_home"))

@app.route("/confirmation")
def confirmation():
    if "image_path" not in session:
        return redirect(url_for("detect_home"))
    return render_template("page2_confirmation.html", image_url=session["image_url"])

@app.route("/process", methods=["POST"])
def process():
    if "image_path" not in session:
        return redirect(url_for("detect_home"))

    image_path = session["image_path"]

    # Pakai fitur dari session jika sudah ada, kalau tidak hitung ulang
    features = session.get("features")
    if features is None:
        feats = extract_features(image_path)
        if feats is None:
            return render_template("page1_upload.html", error="Ekstraksi fitur gagal. Coba gambar lain.")
        try:
            features = feats.tolist()
        except Exception:
            features = np.asarray(feats, dtype=float).tolist()

    features = np.asarray(features, dtype=float)

    # Siapkan input gambar (untuk cabang CNN bila model hybrid)
    img = cv2.imread(image_path)
    if img is None:
        return render_template("page1_upload.html", error="Gambar gagal dibaca.")
    img_input = cv2.resize(img, (128, 128)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Pastikan scaler ada
    if scaler is None:
        abort(500, description="Scaler not loaded")

    # Scale fitur tabular
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Prediksi (hybrid: [image, features])
    pred = model.predict([img_input, features_scaled], verbose=0)

    confidence = float(np.max(pred))
    pred_class = le.classes_[np.argmax(pred)]

    # Threshold
    confidence_threshold = 0.70

    species_endpoint = None
    species_name = None
    if confidence >= confidence_threshold and pred_class in PREDICTION_MAPPING:
        _, species_endpoint = PREDICTION_MAPPING[pred_class]
        species_name = pred_class
    else:
        pred_class = "Tidak Terdeteksi"

    # Nama fitur (sesuaikan dengan jumlah/urutan fitur yang sebenarnya)
    feature_names = [
        "Area", "Perimeter", "Aspect Ratio", "Circularity", "Rectangularity",
        "Diameter",
        "Hue Mean", "Saturation Mean", "Value Mean",
        "Hue Std", "Saturation Std", "Value Std",
        *[f"Hue Hist Bin {i+1}" for i in range(8)],
        "contrast", "correlation", "energy", "homogeneity", "entropy",
    ]

    # Zip akan mengikuti panjang terpendek—aman walau jumlah tidak persis sama
    features_list = [{"name": n, "value": float(v)} for n, v in zip(feature_names, features)]

    # Simpan hasil ke session untuk halaman results
    session["prediction"] = pred_class
    session["species_endpoint"] = species_endpoint
    session["species_name"] = species_name
    session["confidence"] = confidence
    session["features"] = features_list

    return redirect(url_for("results"))

@app.get("/_health")
def _health():
    return {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "le_loaded": le is not None,
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "model_path": str((APP_ROOT / "best_model_compressed.h5").exists()),
        "scaler_path": str((APP_ROOT / "scaler.pkl").exists()),
        "le_path": str((APP_ROOT / "label_encoder.pkl").exists()),
    }

@app.route("/results")
def results():
    if "prediction" not in session:
        return redirect(url_for("detect_home"))

    return render_template(
        "page3_results.html",
        image_url=session["image_url"],
        prediction=session["prediction"],
        species_endpoint=session.get("species_endpoint"),
        species_name=session.get("species_name"),
        confidence=session["confidence"],
        features=session["features"],
    )

# =========================================================
# Halaman species
# =========================================================
@app.route("/species/avicennia-marina")
def avicennia_marina():
    species = {
        "nama_latin": "Avicennia marina",
        "nama_lokal": ["Api-api putih", "Api-api abang", "Sia-sia putih", "Sie-sie", "Pejapi", "Nyapi", "Hajusia", "Pai"],
        "deskripsi_umum": """
            Belukar atau pohon yang tumbuh tegak atau menyebar, ketinggian pohon mencapai 30 meter. 
            Memiliki sistem perakaran horizontal yang rumit dan berbentuk pensil (atau berbentuk asparagus), 
            akar nafas tegak dengan sejumlah lentisel. Kulit kayu halus dengan burik-burik hijau-abu dan 
            terkelupas dalam bagian-bagian kecil. Ranting muda dan tangkai daun berwarna kuning, tidak berbulu.
        """,
        "daun": {
            "deskripsi": """
                Bagian atas permukaan daun ditutupi bintik-bintik kelenjar berbentuk cekung.
                Bagian bawah daun putih-abu-abu muda.
            """,
            "unit_letak": "Sederhana & berlawanan",
            "bentuk": "Elips, bulat memanjang, bulat telur terbalik",
            "ujung": "Meruncing hingga membundar",
            "ukuran": "9 x 4,5 cm",
        },
        "manfaat": [
            "Daun digunakan untuk mengatasi kulit yang terbakar",
            "Resin yang keluar dari kulit kayu digunakan sebagai alat kontrasepsi",
            "Buah dapat dimakan",
            "Kayu menghasilkan bahan kertas berkualitas tinggi",
            "Daun digunakan sebagai makanan ternak",
        ],
        "penyebaran": {
            "lokasi": ["Aceh", "Sumatera Utara", "Riau", "Jawa", "Bali", "Kalimantan", "Sulawesi", "Papua"],
            "koordinat": [
                {"nama": "Aceh", "lat": 4.695135, "lng": 96.749399},
                {"nama": "Sumatera Utara", "lat": 2.115354, "lng": 99.545097},
                {"nama": "Riau", "lat": 0.293347, "lng": 101.706829},
                {"nama": "Jawa", "lat": -7.245972, "lng": 112.737991},
                {"nama": "Bali", "lat": -8.409518, "lng": 115.188919},
                {"nama": "Kalimantan", "lat": -0.502106, "lng": 117.153709},
                {"nama": "Sulawesi", "lat": -3.549121, "lng": 121.727539},
                {"nama": "Papua", "lat": -2.533333, "lng": 140.716667},
            ],
        },
    }
    return render_template("avicennia_marina.html", species=species)

@app.route("/species/bruguiera-gymnorhiza")
def bruguiera_detail():
    species = {
        "nama_latin": "Bruguiera gymnorhiza",
        "nama_lokal": [
            "Pertut", "Taheup", "Tenggel", "Putut", "Tumu", "Tomo",
            "Kandeka", "Tanjang merah", "Tanjang", "Lindur", "Sala-sala",
            "Dau", "Tongke", "Totongkek", "Mutut besar", "Wako", "Bako",
            "Bangko", "Mangi-mangi", "Sarau",
        ],
        "deskripsi_umum": """
            Pohon yang selalu hijau dengan ketinggian kadang-kadang mencapai 30 m.
            Kulit kayu memiliki lentisel, permukaannya halus hingga kasar, berwarna
            abu-abu tua sampai coklat (warna berubah-ubah). Akarnya seperti papan
            melebar ke samping di bagian pangkal pohon, juga memiliki sejumlah akar lutut.
        """,
        "daun": {
            "deskripsi": """
                Daun berkulit, berwarna hijau pada lapisan atas dan hijau kekuningan
                pada bagian bawahnya dengan bercak-bercak hitam (ada juga yang tidak).
            """,
            "unit_letak": "Sederhana & berlawanan",
            "bentuk": "Elips sampai elips-lanset",
            "ujung": "Meruncing",
            "ukuran": "4,5-7 x 8,5-22 cm",
        },
        "manfaat": [
            "Bagian dalam hipokotil dimakan (manisan kandeka), dicampur dengan gula",
            "Kayu yang berwarna merah digunakan sebagai kayu bakar",
            "Untuk membuat arang berkualitas tinggi",
            "Ekstrak kulit digunakan dalam pengobatan tradisional",
            "Daun muda digunakan sebagai pakan ternak",
        ],
        "penyebaran": {
            "lokasi": [
                "Aceh", "Sumatera Utara", "Riau", "Kepulauan Riau",
                "Jawa Barat", "Jawa Timur", "Bali",
                "Kalimantan Barat", "Kalimantan Timur",
                "Sulawesi Selatan", "Sulawesi Tenggara",
                "Maluku", "Papua Barat", "Papua",
            ],
            "koordinat": [
                {"nama": "Aceh", "lat": 4.695135, "lng": 96.749399},
                {"nama": "Sumatera Utara", "lat": 2.115354, "lng": 99.545097},
                {"nama": "Riau", "lat": 0.293347, "lng": 101.706829},
                {"nama": "Kepulauan Riau", "lat": 3.945651, "lng": 108.142867},
                {"nama": "Jawa Barat", "lat": -6.914744, "lng": 107.609810},
                {"nama": "Jawa Timur", "lat": -7.245972, "lng": 112.737991},
                {"nama": "Bali", "lat": -8.409518, "lng": 115.188919},
                {"nama": "Kalimantan Barat", "lat": -0.278781, "lng": 111.475285},
                {"nama": "Kalimantan Timur", "lat": -0.502106, "lng": 117.153709},
                {"nama": "Sulawesi Selatan", "lat": -5.147665, "lng": 119.432731},
                {"nama": "Sulawesi Tenggara", "lat": -3.549121, "lng": 121.727539},
                {"nama": "Maluku", "lat": -3.238462, "lng": 130.145273},
                {"nama": "Papua Barat", "lat": -1.336115, "lng": 133.174716},
                {"nama": "Papua", "lat": -2.533333, "lng": 140.716667},
            ],
        },
    }
    return render_template("bruguiera_gymnorrhiza.html", species=species)  # perhatikan nama template

@app.route("/species/heritiera-littoralis")
def heritiera_littoralis():
    species = {
        "nama_latin": "Heritiera littoralis",
        "nama_lokal": [
            "Dungu", "Dungun", "Atung laut", "Lawanan kete", "Rumung",
            "Balang pasisir", "Lawang", "Cerlang laut", "Lulun", "Rurun",
            "Belohila", "Blakangabu", "Bayur laut",
        ],
        "deskripsi_umum": """
            Pohon yang selalu hijau dengan ketinggian mencapai 25 m. Akar papan
            berkembang sangat jelas. Kulit kayu gelap atau abu-abu, bersisik dan bercelah.
            Individu pohon memiliki salah satu bunga betina atau jantan.
        """,
        "daun": {
            "deskripsi": """
                Kukuh, berkulit, berkelompok pada ujung cabang. Warna daun hijau gelap
                bagian atas dan putih-keabu-abuan di bagian bawah karena adanya lapisan
                yang bertumpang-tindih.
            """,
            "gagang": "0,5-2 cm",
            "unit_letak": "Sederhana, bersilangan",
            "bentuk": "Bulat telur-elips",
            "ujung": "Meruncing",
            "ukuran": "10-20 x 5-10 cm (kadang sampai 30 x 15-18 cm)",
        },
        "manfaat": [
            "Kayu bakar yang baik",
            "Kayu tahan lama untuk bahan perahu, rumah, dan tiang telepon",
            "Buah untuk mengobati diare dan disentri",
            "Biji digunakan dalam pengolahan ikan",
            "Ekstrak daun digunakan dalam pengobatan tradisional",
        ],
        "penyebaran": {
            "lokasi": [
                "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau",
                "Kepulauan Riau", "Jawa", "Bali", "Nusa Tenggara",
                "Kalimantan Barat", "Kalimantan Timur", "Sulawesi",
                "Maluku", "Papua",
            ],
            "koordinat": [
                {"nama": "Aceh", "lat": 4.695135, "lng": 96.749399},
                {"nama": "Sumatera Utara", "lat": 2.115354, "lng": 99.545097},
                {"nama": "Sumatera Barat", "lat": -0.739939, "lng": 100.800005},
                {"nama": "Riau", "lat": 0.293347, "lng": 101.706829},
                {"nama": "Kepulauan Riau", "lat": 3.945651, "lng": 108.142867},
                {"nama": "Jawa", "lat": -7.245972, "lng": 112.737991},
                {"nama": "Bali", "lat": -8.409518, "lng": 115.188919},
                {"nama": "Nusa Tenggara", "lat": -8.652933, "lng": 117.361648},
                {"nama": "Kalimantan Barat", "lat": -0.278781, "lng": 111.475285},
                {"nama": "Kalimantan Timur", "lat": -0.502106, "lng": 117.153709},
                {"nama": "Sulawesi", "lat": -3.549121, "lng": 121.727539},
                {"nama": "Maluku", "lat": -3.238462, "lng": 130.145273},
                {"nama": "Papua", "lat": -2.533333, "lng": 140.716667},
            ],
        },
    }
    return render_template("heritiera_littoralis.html", species=species)

@app.route("/species/avicennia-officinalis")
def avicennia_officinalis():
    species = {
        "nama_latin": "Avicennia officinalis",
        "nama_lokal": [
            "Api-api", "Api-api daun lebar", "Api-api ludat",
            "Sia-sia putih", "Papi", "Api-api kacang",
            "Merahu", "Marahuf",
        ],
        "deskripsi_umum": """
            Pohon, biasanya memiliki ketinggian sampai 12 m, bahkan kadang-kadang
            sampai 20 m. Pada umumnya memiliki akar tunjang dan akar nafas yang tipis,
            berbentuk jari dan ditutupi oleh sejumlah lentisel. Kulit kayu bagian luar memiliki
            permukaan yang halus berwarna hijau-keabu-abuan sampai abu-abu-kecoklatan
            serta memiliki lentisel.
        """,
        "daun": {
            "deskripsi": """
                Berwarna hijau tua pada permukaan atas dan hijau-kekuningan atau abu-abu
                kehijauan di bagian bawah. Permukaan atas daun ditutupi oleh sejumlah bintik
                kelenjar berbentuk cekung.
            """,
            "unit_letak": "Sederhana & berlawanan",
            "bentuk": "Bulat telur terbalik, bulat memanjang-bulat telur terbalik atau elips bulat memanjang",
            "ujung": "Membundar, menyempit ke arah gagang",
            "ukuran": "12,5 x 6 cm",
        },
        "manfaat": [
            "Buah dapat dimakan",
            "Kayu digunakan sebagai kayu bakar",
            "Getah kayu dapat digunakan sebagai bahan alat kontrasepsi",
            "Daun digunakan dalam pengobatan tradisional",
            "Akar membantu stabilisasi sedimentasi pantai",
        ],
        "penyebaran": {
            "lokasi": [
                "Aceh", "Sumatera Utara", "Riau", "Jambi",
                "Sumatera Selatan", "Lampung", "Jawa",
                "Bali", "Kalimantan Barat", "Kalimantan Timur",
                "Sulawesi Selatan", "Sulawesi Tenggara",
                "Maluku", "Papua",
            ],
            "koordinat": [
                {"nama": "Aceh", "lat": 4.695135, "lng": 96.749399},
                {"nama": "Sumatera Utara", "lat": 2.115354, "lng": 99.545097},
                {"nama": "Riau", "lat": 0.293347, "lng": 101.706829},
                {"nama": "Jambi", "lat": -1.485183, "lng": 102.438058},
                {"nama": "Sumatera Selatan", "lat": -2.990934, "lng": 104.756556},
                {"nama": "Lampung", "lat": -5.109730, "lng": 105.547266},
                {"nama": "Jawa", "lat": -7.245972, "lng": 112.737991},
                {"nama": "Bali", "lat": -8.409518, "lng": 115.188919},
                {"nama": "Kalimantan Barat", "lat": -0.278781, "lng": 111.475285},
                {"nama": "Kalimantan Timur", "lat": -0.502106, "lng": 117.153709},
                {"nama": "Sulawesi Selatan", "lat": -5.147665, "lng": 119.432731},
                {"nama": "Sulawesi Tenggara", "lat": -3.549121, "lng": 121.727539},
                {"nama": "Maluku", "lat": -3.238462, "lng": 130.145273},
                {"nama": "Papua", "lat": -2.533333, "lng": 140.716667},
            ],
        },
    }
    return render_template("avicennia_officinalis.html", species=species)

@app.route("/species/lumnitzera-littorea")
def lumnitzera_littorea():
    species = {
        "nama_latin": "Lumnitzera littorea",
        "nama_lokal": [
            "Teruntum merah", "Api-api uding", "Sesop", "Sesak", "Geriting",
            "Randai", "Riang laut", "Taruntung", "Duduk agung", "Duduk gedeh",
            "Welompelong", "Posi-posi", "Ma gorago", "Kedukduk",
        ],
        "deskripsi_umum": """
            Pohon selalu hijau dan tumbuh tersebar, ketinggian pohon dapat mencapai
            25 m, meskipun pada umumnya lebih rendah. Akar nafas berbentuk lutut,
            berwarna coklat tua dan kulit kayu memiliki celah/retakan membujur
            (longitudinal).
        """,
        "daun": {
            "deskripsi": "Daun agak tebal berdaging, keras/kaku, dan berumpun pada ujung dahan",
            "tangkai": "Panjang mencapai 5 mm",
            "unit_letak": "Sederhana, bersilangan",
            "bentuk": "Bulat telur terbalik",
            "ujung": "Membundar",
            "ukuran": "2-8 x 1-2,5 cm",
        },
        "manfaat": [
            "Kayu kuat dan sangat tahan terhadap air",
            "Sangat cocok untuk pembuatan lemari dan furnitur",
            "Memiliki aroma wangi seperti mawar",
            "Digunakan dalam kerajinan kayu bernilai tinggi",
            "Ekstrak kulit digunakan dalam pengobatan tradisional",
        ],
        "penyebaran": {
            "lokasi": [
                "Aceh", "Sumatera Utara", "Riau", "Kepulauan Riau",
                "Jawa Barat", "Jawa Timur", "Bali",
                "Kalimantan Barat", "Kalimantan Timur",
                "Sulawesi Utara", "Sulawesi Selatan",
                "Maluku", "Papua Barat", "Papua",
            ],
            "koordinat": [
                {"nama": "Aceh", "lat": 4.695135, "lng": 96.749399},
                {"nama": "Sumatera Utara", "lat": 2.115354, "lng": 99.545097},
                {"nama": "Riau", "lat": 0.293347, "lng": 101.706829},
                {"nama": "Kepulauan Riau", "lat": 3.945651, "lng": 108.142867},
                {"nama": "Jawa Barat", "lat": -6.914744, "lng": 107.609810},
                {"nama": "Jawa Timur", "lat": -7.245972, "lng": 112.737991},
                {"nama": "Bali", "lat": -8.409518, "lng": 115.188919},
                {"nama": "Kalimantan Barat", "lat": -0.278781, "lng": 111.475285},
                {"nama": "Kalimantan Timur", "lat": -0.502106, "lng": 117.153709},
                {"nama": "Sulawesi Utara", "lat": 1.474830, "lng": 124.842079},
                {"nama": "Sulawesi Selatan", "lat": -5.147665, "lng": 119.432731},
                {"nama": "Maluku", "lat": -3.238462, "lng": 130.145273},
                {"nama": "Papua Barat", "lat": -1.336115, "lng": 133.174716},
                {"nama": "Papua", "lat": -2.533333, "lng": 140.716667},
            ],
        },
    }
    return render_template("lumnitzera_littorea.html", species=species)

@app.route("/species/rhizophora-apiculata")
def rhizophora_apiculata():
    species = {
        "nama_latin": "Rhizophora apiculata",
        "nama_lokal": [
            "Bakau minyak", "Bakau tandok", "Bakau akik", "Bakau puteh",
            "Bakau kacang", "Bakau leutik", "Akik", "Bangka minyak",
            "Donggo akit", "Jankar", "Abat", "Parai", "Mangi-mangi",
            "Slengkreng", "Tinjang", "Wako",
        ],
        "deskripsi_umum": """
            Pohon dengan ketinggian mencapai 30 m dengan diameter batang mencapai
            50 cm. Memiliki perakaran yang khas hingga mencapai ketinggian 5 meter,
            dan kadang-kadang memiliki akar udara yang keluar dari cabang. Kulit kayu
            berwarna abu-abu tua dan berubah-ubah.
        """,
        "daun": {
            "deskripsi": "Berkulit, warna hijau tua dengan hijau muda pada bagian tengah dan kemerahan di bagian bawah",
            "gagang": "17-35 mm, kemerahan",
            "unit_letak": "Sederhana & berlawanan",
            "bentuk": "Elips menyempit",
            "ujung": "Meruncing",
            "ukuran": "7-19 x 3,5-8 cm",
        },
        "manfaat": [
            "Kayu untuk bahan bangunan, kayu bakar dan arang",
            "Kulit kayu mengandung hingga 30% tanin",
            "Cabang akar digunakan sebagai jangkar",
            "Penahan pematang tambak",
            "Tanaman penghijauan pesisir",
            "Ekosistem penting bagi biota laut",
        ],
        "penyebaran": {
            "lokasi": [
                "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau",
                "Jambi", "Sumatera Selatan", "Lampung",
                "Banten", "Jakarta", "Jawa Barat", "Jawa Tengah",
                "Jawa Timur", "Bali", "Kalimantan Barat",
                "Kalimantan Timur", "Sulawesi Selatan",
                "Sulawesi Tenggara", "Maluku", "Papua",
            ],
            "koordinat": [
                {"nama": "Aceh", "lat": 4.695135, "lng": 96.749399},
                {"nama": "Sumatera Utara", "lat": 2.115354, "lng": 99.545097},
                {"nama": "Sumatera Barat", "lat": -0.739939, "lng": 100.800005},
                {"nama": "Riau", "lat": 0.293347, "lng": 101.706829},
                {"nama": "Jambi", "lat": -1.485183, "lng": 102.438058},
                {"nama": "Sumatera Selatan", "lat": -2.990934, "lng": 104.756556},
                {"nama": "Lampung", "lat": -5.109730, "lng": 105.547266},
                {"nama": "Banten", "lat": -6.405817, "lng": 106.064018},
                {"nama": "Jakarta", "lat": -6.208763, "lng": 106.845599},
                {"nama": "Jawa Barat", "lat": -6.914744, "lng": 107.609810},
                {"nama": "Jawa Tengah", "lat": -6.966667, "lng": 110.416664},
                {"nama": "Jawa Timur", "lat": -7.245972, "lng": 112.737991},
                {"nama": "Bali", "lat": -8.409518, "lng": 115.188919},
                {"nama": "Kalimantan Barat", "lat": -0.278781, "lng": 111.475285},
                {"nama": "Kalimantan Timur", "lat": -0.502106, "lng": 117.153709},
                {"nama": "Sulawesi Selatan", "lat": -5.147665, "lng": 119.432731},
                {"nama": "Sulawesi Tenggara", "lat": -3.549121, "lng": 121.727539},
                {"nama": "Maluku", "lat": -3.238462, "lng": 130.145273},
                {"nama": "Papua", "lat": -2.533333, "lng": 140.716667},
            ],
        },
    }
    return render_template("rhizophora_apiculata.html", species=species)

@app.route("/species/sonneratia-alba")
def sonneratia_alba():
    species = {
        "nama_latin": "Sonneratia alba",
        "nama_lokal": [
            "Pedada", "Perepat", "Pidada", "Bogem", "Bidada",
            "Posi-posi", "Wahat", "Putih", "Beropak", "Bangka",
            "Susup", "Kedada", "Muntu", "Sopo", "Barapak", "Pupat", "Mange-mange",
        ],
        "deskripsi_umum": """
            Pohon selalu hijau, tumbuh tersebar, ketinggian kadang-kadang hingga 15 m.
            Kulit kayu berwarna putih tua hingga coklat, dengan celah longitudinal yang halus.
            Akar berbentuk kabel di bawah tanah dan muncul kepermukaan sebagai akar nafas
            yang berbentuk kerucut tumpul dan tingginya mencapai 25 cm.
        """,
        "daun": {
            "deskripsi": "Berkulit, memiliki kelenjar yang tidak berkembang pada bagian pangkal gagang daun",
            "gagang": "6-15 mm",
            "unit_letak": "Sederhana & berlawanan",
            "bentuk": "Bulat telur terbalik",
            "ujung": "Membundar",
            "ukuran": "5-12,5 x 3-9 cm",
        },
        "manfaat": [
            "Buah asam dapat dimakan",
            "Kayu untuk perahu dan bahan bangunan",
            "Sebagai bahan bakar alternatif",
            "Akar nafas digunakan sebagai gabus dan pelampung",
            "Bunga sebagai sumber nektar lebah madu",
            "Daun muda digunakan sebagai pakan ternak",
        ],
        "penyebaran": {
            "lokasi": [
                "Aceh", "Sumatera Utara", "Riau", "Kepulauan Riau",
                "Jakarta", "Jawa Barat", "Jawa Timur", "Bali",
                "Kalimantan Barat", "Kalimantan Timur", "Sulawesi Utara",
                "Sulawesi Selatan", "Sulawesi Tenggara", "Maluku",
                "Papua Barat", "Papua",
            ],
            "koordinat": [
                {"nama": "Aceh", "lat": 4.695135, "lng": 96.749399},
                {"nama": "Sumatera Utara", "lat": 2.115354, "lng": 99.545097},
                {"nama": "Riau", "lat": 0.293347, "lng": 101.706829},
                {"nama": "Kepulauan Riau", "lat": 3.945651, "lng": 108.142867},
                {"nama": "Jakarta", "lat": -6.208763, "lng": 106.845599},
                {"nama": "Jawa Barat", "lat": -6.914744, "lng": 107.609810},
                {"nama": "Jawa Timur", "lat": -7.245972, "lng": 112.737991},
                {"nama": "Bali", "lat": -8.409518, "lng": 115.188919},
                {"nama": "Kalimantan Barat", "lat": -0.278781, "lng": 111.475285},
                {"nama": "Kalimantan Timur", "lat": -0.502106, "lng": 117.153709},
                {"nama": "Sulawesi Utara", "lat": 1.47483, "lng": 124.842079},
                {"nama": "Sulawesi Selatan", "lat": -5.147665, "lng": 119.432731},
                {"nama": "Sulawesi Tenggara", "lat": -3.549121, "lng": 121.727539},
                {"nama": "Maluku", "lat": -3.238462, "lng": 130.145273},
                {"nama": "Papua Barat", "lat": -1.336115, "lng": 133.174716},
                {"nama": "Papua", "lat": -2.533333, "lng": 140.716667},
            ],
        },
    }
    return render_template("sonneratia_alba_detail.html", species=species)

# =========================================================
# Local dev
# =========================================================
if __name__ == "__main__":
    # untuk run lokal: python app.py
    app.run(host="0.0.0.0", port=8000, debug=True)
