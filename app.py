from flask import Flask, render_template, request, redirect, url_for, session, jsonify, abort
import os
import math
import cv2
import numpy as np
import pickle
from skimage.feature import graycomatrix, graycoprops
from ektraksi import extract_features
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit


# Prediction mapping dictionary
PREDICTION_MAPPING = {
    'Am': ('avicennia-marina', 'avicennia_marina'),
    'Ao': ('avicennia-officinalis', 'avicennia_officinalis'),
    'Bg': ('bruguiera-gymnorhiza', 'bruguiera_detail'),
    'Hl': ('heritiera-littoralis', 'heritiera_littoralis'),
    'Lt+': ('lumnitzera-littorea', 'lumnitzera_littorea'),
    'Ra': ('rhizophora-apiculata', 'rhizophora_apiculata'),
    'Sa': ('sonneratia-alba', 'sonneratia_alba')
}

# Load models and utilities
model = None
scaler = None
le = None

def load_artifacts():
    global model, scaler, le
    model = load_model('best_model 80 dan 20.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

# (Keep the exact same extract_features function from your provided code)
# def extract_features(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         return None

#     # --- Shape Features ---
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)
#     kernel = np.ones((5, 5), np.uint8)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) == 0:
#         return None
    
#     leaf_contour = max(contours, key=cv2.contourArea)
#     points = leaf_contour[:, 0, :]

#     # Calculate maximum length (major axis)
#     max_length = 0
#     tip = base = None
#     for i in range(len(points)):
#         for j in range(i + 1, len(points)):
#             dist = np.linalg.norm(points[i] - points[j])
#             if dist > max_length:
#                 max_length = dist
#                 tip, base = points[i], points[j]

#     # Calculate width (minor axis) perpendicular to length
#     dy = base[1] - tip[1]
#     dx = base[0] - tip[0]
#     length_slope = dy / dx if dx != 0 else np.inf
#     width_slope = -1 / length_slope if length_slope not in [0, np.inf] else np.inf

#     max_width = 0
#     for i in range(len(points)):
#         for j in range(i + 1, len(points)):
#             dx = points[j][0] - points[i][0]
#             dy = points[j][1] - points[i][1]
#             slope = dy / dx if dx != 0 else np.inf
#             if abs(slope - width_slope) < 0.2:
#                 dist = np.linalg.norm(points[i] - points[j])
#                 if dist > max_width:
#                     max_width = dist

#     # Calculate all shape features
#     area = cv2.contourArea(leaf_contour)
#     perimeter = cv2.arcLength(leaf_contour, True)
#     aspect_ratio = max_length / max_width if max_width != 0 else 0
#     circularity = (4 * math.pi * area) / (perimeter**2) if perimeter != 0 else 0
#     rectangularity = area / (max_length * max_width) if (max_length * max_width) != 0 else 0
#     diameter = max([np.linalg.norm(p1 - p2) for p1 in points for p2 in points])
#     # narrow_factor = diameter / max_length if max_length != 0 else 0
#     # ratio_perim_diam = perimeter / diameter if diameter != 0 else 0
#     # ratio_perim_lenwidth = perimeter / (max_length + max_width) if (max_length + max_width) != 0 else 0

#     # --- Color Features ---
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
#     lower_green = np.array([25, 40, 40])
#     upper_green = np.array([85, 255, 255])
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     leaf_pixels = hsv[mask > 0]
#     if len(leaf_pixels) == 0:
#         return None

#     mean_hsv = np.mean(leaf_pixels, axis=0)  # 3 features (H,S,V)
#     std_hsv = np.std(leaf_pixels, axis=0)    # 3 features (H,S,V)
#     hist_hue, _ = np.histogram(leaf_pixels[:, 0], bins=8, range=(0, 180))  # 16 features
#     hist_hue = hist_hue / hist_hue.sum()
#     color_features = np.concatenate([mean_hsv, std_hsv, hist_hue])  # 22 color features total

#     # --- Texture Features ---
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray_resized = cv2.resize(gray, (128, 128))
#     angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#     glcm = graycomatrix(gray_resized, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
#     props = ['contrast', 'homogeneity']
#     texture_features = [np.mean(graycoprops(glcm, p)[0]) for p in props]  # 4 features
#     glcm_sum = np.sum(glcm, axis=3)
#     glcm_prob = glcm_sum / np.sum(glcm_sum)
#     entropy = -np.sum(glcm_prob * np.log2(glcm_prob + 1e-10))  # 1 feature
#     texture_features.append(entropy)  # 5 texture features total

#     # Combine all features (9 shape + 22 color + 5 texture = 36 features)
#     features = np.concatenate([
#         [area, perimeter, aspect_ratio, circularity, rectangularity, 
#          diameter],
#         color_features,
#         texture_features
#     ])
    
#     return features 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Base template with navbar
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faq')
def contact():
    return render_template('faq.html')

# Detection routes
@app.route('/detect')
def detect_home():
    return render_template('page1_upload.html')

# from werkzeug.utils import secure_filename  # Add this import at the top

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('detect_home'))
        
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('detect_home'))
        
    if file and allowed_file(file.filename):
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Gunakan timestamp sebagai nama file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

            # Simpan file ke folder
            file.save(filepath)

            if not os.path.exists(filepath):
                raise Exception("File failed to save")

            # Panggil ekstraksi fitur setelah file disimpan
            features = extract_features(filepath)

            # Simpan fitur jika diperlukan (opsional)
            # np.save(f'static/features/{timestamp}_features.npy', features)

            # Simpan ke sesi untuk digunakan di halaman lain
            session['image_path'] = filepath
            session['image_url'] = f'/static/uploads/{safe_filename}'
            session['features'] = features.tolist()  # jika ingin passing ke frontend

            return redirect(url_for('confirmation'))

        except Exception as e:
            print(f"Error: {str(e)}")
            return render_template('page1_upload.html', 
                                   error="An error occurred. Please try again.")
    
    return redirect(url_for('detect_home'))



@app.route('/confirmation')
def confirmation():
    if 'image_path' not in session:
        return redirect(url_for('detect_home'))
    return render_template('page2_confirmation.html', image_url=session['image_url'])

@app.route('/process', methods=['POST'])
def process():
    if 'image_path' not in session:
        return redirect(url_for('detect_home'))
    
    image_path = session['image_path']
    
    # Extract features (replace with your actual feature extraction)
    features = extract_features(image_path)  # This should be your actual function
    if features is None:
        return redirect(url_for('detect_home'))
    
    # Preprocess image for model (example - adapt to your model)
    img = cv2.imread(image_path)
    img_input = cv2.resize(img, (128, 128)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    # Scale features (example - adapt to your scaler)
    features_scaled = scaler.transform(features.reshape(1, -1))  # Use your actual scaler
    
    # Make prediction (example - adapt to your model)
    pred = model.predict([img_input, features_scaled])  # Use your actual model
    
    # Get confidence and class
    confidence = float(np.max(pred))
    pred_class = le.classes_[np.argmax(pred)]  # Use your actual label encoder
    
    # Set threshold for detection
    confidence_threshold = 0.70
    
    # Initialize variables
    species_endpoint = None
    species_name = None
    
    if confidence >= confidence_threshold and pred_class in PREDICTION_MAPPING:
        species_url, species_endpoint = PREDICTION_MAPPING[pred_class]
        species_name = pred_class
    else:
        pred_class = "Tidak Terdeteksi"
    
    # Prepare feature names and values (example - adapt to your features)
    feature_names = [
        'Area', 'Perimeter', 'Aspect Ratio', 'Circularity', 'Rectangularity',
        'Diameter',
        'Hue Mean', 'Saturation Mean', 'Value Mean',
        'Hue Std', 'Saturation Std', 'Value Std',
        *[f'Hue Hist Bin {i+1}' for i in range(8)],
        'contrast', 'correlation', 'energy', 'homogeneity', 'entropy'
    ]
    
    features_list = [{
        'name': name,
        'value': float(value)
    } for name, value in zip(feature_names, features)]
    
    # Store results in session
    session['prediction'] = pred_class
    session['species_endpoint'] = species_endpoint
    session['species_name'] = species_name
    session['confidence'] = confidence
    session['features'] = features_list
    
    return redirect(url_for('results'))


@app.route('/results')
def results():
    if 'prediction' not in session:
        return redirect(url_for('detect_home'))
    
    return render_template('page3_results.html',
                         image_url=session['image_url'],
                         prediction=session['prediction'],
                         species_endpoint=session.get('species_endpoint'),
                         species_name=session.get('species_name'),
                         confidence=session['confidence'],
                         features=session['features'])

@app.route('/species/avicennia-marina')
def avicennia_marina():
    species = {
        'nama_latin': 'Avicennia marina',
        'nama_lokal': ['Api-api putih', 'Api-api abang', 'Sia-sia putih', 'Sie-sie', 'Pejapi', 'Nyapi', 'Hajusia', 'Pai'],
        'deskripsi_umum': '''
            Belukar atau pohon yang tumbuh tegak atau menyebar, ketinggian pohon mencapai 30 meter. 
            Memiliki sistem perakaran horizontal yang rumit dan berbentuk pensil (atau berbentuk asparagus), 
            akar nafas tegak dengan sejumlah lentisel. Kulit kayu halus dengan burik-burik hijau-abu dan 
            terkelupas dalam bagian-bagian kecil. Ranting muda dan tangkai daun berwarna kuning, tidak berbulu.
        ''',
        'daun': {
            'deskripsi': '''
                Bagian atas permukaan daun ditutupi bintik-bintik kelenjar berbentuk cekung.
                Bagian bawah daun putih-abu-abu muda.
            ''',
            'unit_letak': 'Sederhana & berlawanan',
            'bentuk': 'Elips, bulat memanjang, bulat telur terbalik',
            'ujung': 'Meruncing hingga membundar',
            'ukuran': '9 x 4,5 cm'
        },
        'manfaat': [
            'Daun digunakan untuk mengatasi kulit yang terbakar',
            'Resin yang keluar dari kulit kayu digunakan sebagai alat kontrasepsi',
            'Buah dapat dimakan',
            'Kayu menghasilkan bahan kertas berkualitas tinggi',
            'Daun digunakan sebagai makanan ternak'
        ],
        'penyebaran': {
            'lokasi': ['Aceh', 'Sumatera Utara', 'Riau', 'Jawa', 'Bali', 'Kalimantan', 'Sulawesi', 'Papua'],
            'koordinat': [
                {'nama': 'Aceh', 'lat': 4.695135, 'lng': 96.749399},
                {'nama': 'Sumatera Utara', 'lat': 2.115354, 'lng': 99.545097},
                {'nama': 'Riau', 'lat': 0.293347, 'lng': 101.706829},
                {'nama': 'Jawa', 'lat': -7.245972, 'lng': 112.737991},
                {'nama': 'Bali', 'lat': -8.409518, 'lng': 115.188919},
                {'nama': 'Kalimantan', 'lat': -0.502106, 'lng': 117.153709},
                {'nama': 'Sulawesi', 'lat': -3.549121, 'lng': 121.727539},
                {'nama': 'Papua', 'lat': -2.533333, 'lng': 140.716667}
            ],
        }
    }
    return render_template('avicennia_marina.html', species=species)

@app.route('/species/bruguiera-gymnorhiza')
def bruguiera_detail():
    species = {
        'nama_latin': 'Bruguiera gymnorhiza',
        'nama_lokal': [
            'Pertut', 'Taheup', 'Tenggel', 'Putut', 'Tumu', 'Tomo', 
            'Kandeka', 'Tanjang merah', 'Tanjang', 'Lindur', 'Sala-sala',
            'Dau', 'Tongke', 'Totongkek', 'Mutut besar', 'Wako', 'Bako',
            'Bangko', 'Mangi-mangi', 'Sarau'
        ],
        'deskripsi_umum': '''
            Pohon yang selalu hijau dengan ketinggian kadang-kadang mencapai 30 m.
            Kulit kayu memiliki lentisel, permukaannya halus hingga kasar, berwarna
            abu-abu tua sampai coklat (warna berubah-ubah). Akarnya seperti papan
            melebar ke samping di bagian pangkal pohon, juga memiliki sejumlah akar lutut.
        ''',
        'daun': {
            'deskripsi': '''
                Daun berkulit, berwarna hijau pada lapisan atas dan hijau kekuningan
                pada bagian bawahnya dengan bercak-bercak hitam (ada juga yang tidak).
            ''',
            'unit_letak': 'Sederhana & berlawanan',
            'bentuk': 'Elips sampai elips-lanset',
            'ujung': 'Meruncing',
            'ukuran': '4,5-7 x 8,5-22 cm'
        },
        'manfaat': [
            'Bagian dalam hipokotil dimakan (manisan kandeka), dicampur dengan gula',
            'Kayu yang berwarna merah digunakan sebagai kayu bakar',
            'Untuk membuat arang berkualitas tinggi',
            'Ekstrak kulit digunakan dalam pengobatan tradisional',
            'Daun muda digunakan sebagai pakan ternak'
        ],
        'penyebaran': {
            'lokasi': [
                'Aceh', 'Sumatera Utara', 'Riau', 'Kepulauan Riau',
                'Jawa Barat', 'Jawa Timur', 'Bali', 
                'Kalimantan Barat', 'Kalimantan Timur',
                'Sulawesi Selatan', 'Sulawesi Tenggara',
                'Maluku', 'Papua Barat', 'Papua'
            ],
            'koordinat': [
                {'nama': 'Aceh', 'lat': 4.695135, 'lng': 96.749399},
                {'nama': 'Sumatera Utara', 'lat': 2.115354, 'lng': 99.545097},
                {'nama': 'Riau', 'lat': 0.293347, 'lng': 101.706829},
                {'nama': 'Kepulauan Riau', 'lat': 3.945651, 'lng': 108.142867},
                {'nama': 'Jawa Barat', 'lat': -6.914744, 'lng': 107.609810},
                {'nama': 'Jawa Timur', 'lat': -7.245972, 'lng': 112.737991},
                {'nama': 'Bali', 'lat': -8.409518, 'lng': 115.188919},
                {'nama': 'Kalimantan Barat', 'lat': -0.278781, 'lng': 111.475285},
                {'nama': 'Kalimantan Timur', 'lat': -0.502106, 'lng': 117.153709},
                {'nama': 'Sulawesi Selatan', 'lat': -5.147665, 'lng': 119.432731},
                {'nama': 'Sulawesi Tenggara', 'lat': -3.549121, 'lng': 121.727539},
                {'nama': 'Maluku', 'lat': -3.238462, 'lng': 130.145273},
                {'nama': 'Papua Barat', 'lat': -1.336115, 'lng': 133.174716},
                {'nama': 'Papua', 'lat': -2.533333, 'lng': 140.716667}
            ]
        }
    }
    return render_template('bruguiera_gymnorhiza.html', species=species)
@app.route('/species/heritiera-littoralis')
def heritiera_littoralis():
    species = {
        'nama_latin': 'Heritiera littoralis',
        'nama_lokal': [
            'Dungu', 'Dungun', 'Atung laut', 'Lawanan kete', 'Rumung',
            'Balang pasisir', 'Lawang', 'Cerlang laut', 'Lulun', 'Rurun',
            'Belohila', 'Blakangabu', 'Bayur laut'
        ],
        'deskripsi_umum': '''
            Pohon yang selalu hijau dengan ketinggian mencapai 25 m. Akar papan
            berkembang sangat jelas. Kulit kayu gelap atau abu-abu, bersisik dan bercelah.
            Individu pohon memiliki salah satu bunga betina atau jantan.
        ''',
        'daun': {
            'deskripsi': '''
                Kukuh, berkulit, berkelompok pada ujung cabang. Warna daun hijau gelap
                bagian atas dan putih-keabu-abuan di bagian bawah karena adanya lapisan
                yang bertumpang-tindih.
            ''',
            'gagang': '0,5-2 cm',
            'unit_letak': 'Sederhana, bersilangan',
            'bentuk': 'Bulat telur-elips',
            'ujung': 'Meruncing',
            'ukuran': '10-20 x 5-10 cm (kadang sampai 30 x 15-18 cm)'
        },
        'manfaat': [
            'Kayu bakar yang baik',
            'Kayu tahan lama untuk bahan perahu, rumah, dan tiang telepon',
            'Buah untuk mengobati diare dan disentri',
            'Biji digunakan dalam pengolahan ikan',
            'Ekstrak daun digunakan dalam pengobatan tradisional'
        ],
        'penyebaran': {
            'lokasi': [
                'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau',
                'Kepulauan Riau', 'Jawa', 'Bali', 'Nusa Tenggara',
                'Kalimantan Barat', 'Kalimantan Timur', 'Sulawesi',
                'Maluku', 'Papua'
            ],
            'koordinat': [
                {'nama': 'Aceh', 'lat': 4.695135, 'lng': 96.749399},
                {'nama': 'Sumatera Utara', 'lat': 2.115354, 'lng': 99.545097},
                {'nama': 'Sumatera Barat', 'lat': -0.739939, 'lng': 100.800005},
                {'nama': 'Riau', 'lat': 0.293347, 'lng': 101.706829},
                {'nama': 'Kepulauan Riau', 'lat': 3.945651, 'lng': 108.142867},
                {'nama': 'Jawa', 'lat': -7.245972, 'lng': 112.737991},
                {'nama': 'Bali', 'lat': -8.409518, 'lng': 115.188919},
                {'nama': 'Nusa Tenggara', 'lat': -8.652933, 'lng': 117.361648},
                {'nama': 'Kalimantan Barat', 'lat': -0.278781, 'lng': 111.475285},
                {'nama': 'Kalimantan Timur', 'lat': -0.502106, 'lng': 117.153709},
                {'nama': 'Sulawesi', 'lat': -3.549121, 'lng': 121.727539},
                {'nama': 'Maluku', 'lat': -3.238462, 'lng': 130.145273},
                {'nama': 'Papua', 'lat': -2.533333, 'lng': 140.716667}
            ]
        }
    }
    return render_template('heritiera_littoralis.html', species=species)

@app.route('/species/avicennia-officinalis')
def avicennia_officinalis():
    species = {
        'nama_latin': 'Avicennia officinalis',
        'nama_lokal': [
            'Api-api', 'Api-api daun lebar', 'Api-api ludat', 
            'Sia-sia putih', 'Papi', 'Api-api kacang',
            'Merahu', 'Marahuf'
        ],
        'deskripsi_umum': '''
            Pohon, biasanya memiliki ketinggian sampai 12 m, bahkan kadang-kadang
            sampai 20 m. Pada umumnya memiliki akar tunjang dan akar nafas yang tipis,
            berbentuk jari dan ditutupi oleh sejumlah lentisel. Kulit kayu bagian luar memiliki
            permukaan yang halus berwarna hijau-keabu-abuan sampai abu-abu-kecoklatan
            serta memiliki lentisel.
        ''',
        'daun': {
            'deskripsi': '''
                Berwarna hijau tua pada permukaan atas dan hijau-kekuningan atau abu-abu
                kehijauan di bagian bawah. Permukaan atas daun ditutupi oleh sejumlah bintik
                kelenjar berbentuk cekung.
            ''',
            'unit_letak': 'Sederhana & berlawanan',
            'bentuk': 'Bulat telur terbalik, bulat memanjang-bulat telur terbalik atau elips bulat memanjang',
            'ujung': 'Membundar, menyempit ke arah gagang',
            'ukuran': '12,5 x 6 cm'
        },
        'manfaat': [
            'Buah dapat dimakan',
            'Kayu digunakan sebagai kayu bakar',
            'Getah kayu dapat digunakan sebagai bahan alat kontrasepsi',
            'Daun digunakan dalam pengobatan tradisional',
            'Akar membantu stabilisasi sedimentasi pantai'
        ],
        'penyebaran': {
            'lokasi': [
                'Aceh', 'Sumatera Utara', 'Riau', 'Jambi',
                'Sumatera Selatan', 'Lampung', 'Jawa',
                'Bali', 'Kalimantan Barat', 'Kalimantan Timur',
                'Sulawesi Selatan', 'Sulawesi Tenggara',
                'Maluku', 'Papua'
            ],
            'koordinat': [
                {'nama': 'Aceh', 'lat': 4.695135, 'lng': 96.749399},
                {'nama': 'Sumatera Utara', 'lat': 2.115354, 'lng': 99.545097},
                {'nama': 'Riau', 'lat': 0.293347, 'lng': 101.706829},
                {'nama': 'Jambi', 'lat': -1.485183, 'lng': 102.438058},
                {'nama': 'Sumatera Selatan', 'lat': -2.990934, 'lng': 104.756556},
                {'nama': 'Lampung', 'lat': -5.109730, 'lng': 105.547266},
                {'nama': 'Jawa', 'lat': -7.245972, 'lng': 112.737991},
                {'nama': 'Bali', 'lat': -8.409518, 'lng': 115.188919},
                {'nama': 'Kalimantan Barat', 'lat': -0.278781, 'lng': 111.475285},
                {'nama': 'Kalimantan Timur', 'lat': -0.502106, 'lng': 117.153709},
                {'nama': 'Sulawesi Selatan', 'lat': -5.147665, 'lng': 119.432731},
                {'nama': 'Sulawesi Tenggara', 'lat': -3.549121, 'lng': 121.727539},
                {'nama': 'Maluku', 'lat': -3.238462, 'lng': 130.145273},
                {'nama': 'Papua', 'lat': -2.533333, 'lng': 140.716667}
            ]
        }
    }
    return render_template('avicennia_officinalis.html', species=species)

@app.route('/species/lumnitzera-littorea')
def lumnitzera_littorea():
    species = {
        'nama_latin': 'Lumnitzera littorea',
        'nama_lokal': [
            'Teruntum merah', 'Api-api uding', 'Sesop', 'Sesak', 'Geriting',
            'Randai', 'Riang laut', 'Taruntung', 'Duduk agung', 'Duduk gedeh',
            'Welompelong', 'Posi-posi', 'Ma gorago', 'Kedukduk'
        ],
        'deskripsi_umum': '''
            Pohon selalu hijau dan tumbuh tersebar, ketinggian pohon dapat mencapai
            25 m, meskipun pada umumnya lebih rendah. Akar nafas berbentuk lutut,
            berwarna coklat tua dan kulit kayu memiliki celah/retakan membujur
            (longitudinal).
        ''',
        'daun': {
            'deskripsi': 'Daun agak tebal berdaging, keras/kaku, dan berumpun pada ujung dahan',
            'tangkai': 'Panjang mencapai 5 mm',
            'unit_letak': 'Sederhana, bersilangan',
            'bentuk': 'Bulat telur terbalik',
            'ujung': 'Membundar',
            'ukuran': '2-8 x 1-2,5 cm'
        },
        'manfaat': [
            'Kayu kuat dan sangat tahan terhadap air',
            'Sangat cocok untuk pembuatan lemari dan furnitur',
            'Memiliki aroma wangi seperti mawar',
            'Digunakan dalam kerajinan kayu bernilai tinggi',
            'Ekstrak kulit digunakan dalam pengobatan tradisional'
        ],
        'penyebaran': {
            'lokasi': [
                'Aceh', 'Sumatera Utara', 'Riau', 'Kepulauan Riau',
                'Jawa Barat', 'Jawa Timur', 'Bali',
                'Kalimantan Barat', 'Kalimantan Timur',
                'Sulawesi Utara', 'Sulawesi Selatan',
                'Maluku', 'Papua Barat', 'Papua'
            ],
            'koordinat': [
                {'nama': 'Aceh', 'lat': 4.695135, 'lng': 96.749399},
                {'nama': 'Sumatera Utara', 'lat': 2.115354, 'lng': 99.545097},
                {'nama': 'Riau', 'lat': 0.293347, 'lng': 101.706829},
                {'nama': 'Kepulauan Riau', 'lat': 3.945651, 'lng': 108.142867},
                {'nama': 'Jawa Barat', 'lat': -6.914744, 'lng': 107.609810},
                {'nama': 'Jawa Timur', 'lat': -7.245972, 'lng': 112.737991},
                {'nama': 'Bali', 'lat': -8.409518, 'lng': 115.188919},
                {'nama': 'Kalimantan Barat', 'lat': -0.278781, 'lng': 111.475285},
                {'nama': 'Kalimantan Timur', 'lat': -0.502106, 'lng': 117.153709},
                {'nama': 'Sulawesi Utara', 'lat': 1.474830, 'lng': 124.842079},
                {'nama': 'Sulawesi Selatan', 'lat': -5.147665, 'lng': 119.432731},
                {'nama': 'Maluku', 'lat': -3.238462, 'lng': 130.145273},
                {'nama': 'Papua Barat', 'lat': -1.336115, 'lng': 133.174716},
                {'nama': 'Papua', 'lat': -2.533333, 'lng': 140.716667}
            ]
        }
    }
    return render_template('lumnitzera_littorea.html', species=species)

@app.route('/species/rhizophora-apiculata')
def rhizophora_apiculata():
    species = {
        'nama_latin': 'Rhizophora apiculata',
        'nama_lokal': [
            'Bakau minyak', 'Bakau tandok', 'Bakau akik', 'Bakau puteh', 
            'Bakau kacang', 'Bakau leutik', 'Akik', 'Bangka minyak',
            'Donggo akit', 'Jankar', 'Abat', 'Parai', 'Mangi-mangi',
            'Slengkreng', 'Tinjang', 'Wako'
        ],
        'deskripsi_umum': '''
            Pohon dengan ketinggian mencapai 30 m dengan diameter batang mencapai
            50 cm. Memiliki perakaran yang khas hingga mencapai ketinggian 5 meter,
            dan kadang-kadang memiliki akar udara yang keluar dari cabang. Kulit kayu
            berwarna abu-abu tua dan berubah-ubah.
        ''',
        'daun': {
            'deskripsi': 'Berkulit, warna hijau tua dengan hijau muda pada bagian tengah dan kemerahan di bagian bawah',
            'gagang': '17-35 mm, kemerahan',
            'unit_letak': 'Sederhana & berlawanan',
            'bentuk': 'Elips menyempit',
            'ujung': 'Meruncing',
            'ukuran': '7-19 x 3,5-8 cm'
        },
        'manfaat': [
            'Kayu untuk bahan bangunan, kayu bakar dan arang',
            'Kulit kayu mengandung hingga 30% tanin',
            'Cabang akar digunakan sebagai jangkar',
            'Penahan pematang tambak',
            'Tanaman penghijauan pesisir',
            'Ekosistem penting bagi biota laut'
        ],
        'penyebaran': {
            'lokasi': [
                'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau',
                'Jambi', 'Sumatera Selatan', 'Lampung',
                'Banten', 'Jakarta', 'Jawa Barat', 'Jawa Tengah',
                'Jawa Timur', 'Bali', 'Kalimantan Barat', 
                'Kalimantan Timur', 'Sulawesi Selatan',
                'Sulawesi Tenggara', 'Maluku', 'Papua'
            ],
            'koordinat': [
                {'nama': 'Aceh', 'lat': 4.695135, 'lng': 96.749399},
                {'nama': 'Sumatera Utara', 'lat': 2.115354, 'lng': 99.545097},
                {'nama': 'Sumatera Barat', 'lat': -0.739939, 'lng': 100.800005},
                {'nama': 'Riau', 'lat': 0.293347, 'lng': 101.706829},
                {'nama': 'Jambi', 'lat': -1.485183, 'lng': 102.438058},
                {'nama': 'Sumatera Selatan', 'lat': -2.990934, 'lng': 104.756556},
                {'nama': 'Lampung', 'lat': -5.109730, 'lng': 105.547266},
                {'nama': 'Banten', 'lat': -6.405817, 'lng': 106.064018},
                {'nama': 'Jakarta', 'lat': -6.208763, 'lng': 106.845599},
                {'nama': 'Jawa Barat', 'lat': -6.914744, 'lng': 107.609810},
                {'nama': 'Jawa Tengah', 'lat': -6.966667, 'lng': 110.416664},
                {'nama': 'Jawa Timur', 'lat': -7.245972, 'lng': 112.737991},
                {'nama': 'Bali', 'lat': -8.409518, 'lng': 115.188919},
                {'nama': 'Kalimantan Barat', 'lat': -0.278781, 'lng': 111.475285},
                {'nama': 'Kalimantan Timur', 'lat': -0.502106, 'lng': 117.153709},
                {'nama': 'Sulawesi Selatan', 'lat': -5.147665, 'lng': 119.432731},
                {'nama': 'Sulawesi Tenggara', 'lat': -3.549121, 'lng': 121.727539},
                {'nama': 'Maluku', 'lat': -3.238462, 'lng': 130.145273},
                {'nama': 'Papua', 'lat': -2.533333, 'lng': 140.716667}
            ]
        }
    }
    return render_template('rhizophora_apiculata.html', species=species)

@app.route('/species/sonneratia-alba')
def sonneratia_alba():
    species = {
        'nama_latin': 'Sonneratia alba',
        'nama_lokal': [
            'Pedada', 'Perepat', 'Pidada', 'Bogem', 'Bidada', 
            'Posi-posi', 'Wahat', 'Putih', 'Beropak', 'Bangka',
            'Susup', 'Kedada', 'Muntu', 'Sopo', 'Barapak', 'Pupat', 'Mange-mange'
        ],
        'deskripsi_umum': '''
            Pohon selalu hijau, tumbuh tersebar, ketinggian kadang-kadang hingga 15 m.
            Kulit kayu berwarna putih tua hingga coklat, dengan celah longitudinal yang halus.
            Akar berbentuk kabel di bawah tanah dan muncul kepermukaan sebagai akar nafas
            yang berbentuk kerucut tumpul dan tingginya mencapai 25 cm.
        ''',
        'daun': {
            'deskripsi': 'Berkulit, memiliki kelenjar yang tidak berkembang pada bagian pangkal gagang daun',
            'gagang': '6-15 mm',
            'unit_letak': 'Sederhana & berlawanan',
            'bentuk': 'Bulat telur terbalik',
            'ujung': 'Membundar',
            'ukuran': '5-12,5 x 3-9 cm'
        },
        'manfaat': [
            'Buah asam dapat dimakan',
            'Kayu untuk perahu dan bahan bangunan',
            'Sebagai bahan bakar alternatif',
            'Akar nafas digunakan sebagai gabus dan pelampung',
            'Bunga sebagai sumber nektar lebah madu',
            'Daun muda digunakan sebagai pakan ternak'
        ],
        'penyebaran': {
            'lokasi': [
                'Aceh', 'Sumatera Utara', 'Riau', 'Kepulauan Riau',
                'Jakarta', 'Jawa Barat', 'Jawa Timur', 'Bali',
                'Kalimantan Barat', 'Kalimantan Timur', 'Sulawesi Utara',
                'Sulawesi Selatan', 'Sulawesi Tenggara', 'Maluku',
                'Papua Barat', 'Papua'
            ],
            'koordinat': [
                {'nama': 'Aceh', 'lat': 4.695135, 'lng': 96.749399},
                {'nama': 'Sumatera Utara', 'lat': 2.115354, 'lng': 99.545097},
                {'nama': 'Riau', 'lat': 0.293347, 'lng': 101.706829},
                {'nama': 'Kepulauan Riau', 'lat': 3.945651, 'lng': 108.142867},
                {'nama': 'Jakarta', 'lat': -6.208763, 'lng': 106.845599},
                {'nama': 'Jawa Barat', 'lat': -6.914744, 'lng': 107.609810},
                {'nama': 'Jawa Timur', 'lat': -7.245972, 'lng': 112.737991},
                {'nama': 'Bali', 'lat': -8.409518, 'lng': 115.188919},
                {'nama': 'Kalimantan Barat', 'lat': -0.278781, 'lng': 111.475285},
                {'nama': 'Kalimantan Timur', 'lat': -0.502106, 'lng': 117.153709},
                {'nama': 'Sulawesi Utara', 'lat': 1.474830, 'lng': 124.842079},
                {'nama': 'Sulawesi Selatan', 'lat': -5.147665, 'lng': 119.432731},
                {'nama': 'Sulawesi Tenggara', 'lat': -3.549121, 'lng': 121.727539},
                {'nama': 'Maluku', 'lat': -3.238462, 'lng': 130.145273},
                {'nama': 'Papua Barat', 'lat': -1.336115, 'lng': 133.174716},
                {'nama': 'Papua', 'lat': -2.533333, 'lng': 140.716667}
            ]
        }
    }
    return render_template('sonneratia_alba_detail.html', species=species)

# @app.route('/species_detail/<prediction>')
# def species_detail(prediction):
#     # Mapping prediksi ke template yang sesuai
#     species_pages = {
#         "Avicennia marina": "avicennia_marina.html",
#         "Avicennia officinalis": "avicennia_officinalis.html",
#         "Rhizophora apiculata": "rhizophora_apiculata.html",
#         "Bruguiera gymnorrhiza": "bruguiera_gymnorhiza.html",
#         "Lumnitzera littorea": "lumnitzera_littorea.html",
#         "Heritiera littoralis": "heritiera_littoralis.html",
#         "Sonneratia alba": "sonneratia_alba_detail.html"

#     }
    
#     template_name = species_pages.get(prediction)
    
#     if not template_name:
#         abort(404, description="Spesies tidak ditemukan")
    
#     return render_template(template_name)
if __name__ == '__main__':
    load_artifacts()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)