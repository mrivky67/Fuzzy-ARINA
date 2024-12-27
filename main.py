import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt


class Selada:
    def __init__(self):
        self.umur = ctrl.Antecedent(np.arange(0, 45, 1), "umur")
        self.suhu = ctrl.Antecedent(np.arange(0, 45, 1), "suhu")
        self.kelembaban = ctrl.Antecedent(np.arange(0, 100, 1), "kelembaban")
        self.NutrisiA = ctrl.Consequent(np.arange(0, 100, 1), "nutrisi_a")
        self.NutrisiB = ctrl.Consequent(np.arange(0, 100, 1), "nutrisi_b")
        self.ketinggianAir = ctrl.Antecedent(np.arange(10, 20, 1), "ketinggian_air")
        self.pompa_nutrisi = ctrl.Consequent(np.arange(0, 2, 1), "pompa_nutrisi")
        self.pompa_sirkulasi = ctrl.Consequent(np.arange(0, 2, 1), "pompa_sirkulasi")
        self.pompa_airbaku = ctrl.Consequent(np.arange(0, 2, 1), "pompa_airbaku")

    def PompaNutrisi(self):
        self.NutrisiA["sedikit"] = fuzz.trapmf(self.NutrisiA.universe, [0, 0, 20, 50])

    def getKategoriUmur(self, input_umur: int):

        self.umur["muda"] = fuzz.trapmf(self.umur.universe, [0, 0, 7, 15])
        self.umur["sedang"] = fuzz.trapmf(self.umur.universe, [14, 22, 22, 32])
        self.umur["tua"] = fuzz.trapmf(self.umur.universe, [31, 40, 45, 45])

        self.kategoriUmur = ctrl.Consequent(np.arange(0, 45, 1), "kategori_umur")
        self.kategoriUmur["bibit"] = fuzz.trapmf(
            self.kategoriUmur.universe, [0, 0, 7, 14]
        )
        self.kategoriUmur["vegetatif"] = fuzz.trapmf(
            self.kategoriUmur.universe, [12, 22, 22, 32]
        )
        self.kategoriUmur["panen"] = fuzz.trapmf(
            self.kategoriUmur.universe, [30, 37, 45, 45]
        )

        rule1 = ctrl.Rule(self.umur["muda"], self.kategoriUmur["bibit"])
        rule2 = ctrl.Rule(self.umur["sedang"], self.kategoriUmur["vegetatif"])
        rule3 = ctrl.Rule(self.umur["tua"], self.kategoriUmur["panen"])

        self.outKategoriUmur = ctrl.ControlSystem([rule1, rule2, rule3])
        self.outputKategoriUmur = ctrl.ControlSystemSimulation(self.outKategoriUmur)

        self.outputKategoriUmur.input["umur"] = input_umur
        self.outputKategoriUmur.compute()

        keanggotaan_muda = fuzz.interp_membership(
            self.umur.universe, self.umur["muda"].mf, input_umur
        )
        keanggotaan_sedang = fuzz.interp_membership(
            self.umur.universe, self.umur["sedang"].mf, input_umur
        )
        keanggotaan_tua = fuzz.interp_membership(
            self.umur.universe, self.umur["tua"].mf, input_umur
        )

        if keanggotaan_muda > keanggotaan_sedang and keanggotaan_muda > keanggotaan_tua:
            return "Bibit"
        elif (
            keanggotaan_sedang > keanggotaan_muda
            and keanggotaan_sedang > keanggotaan_tua
        ):
            return "Vegetatif"
        elif (
            keanggotaan_tua > keanggotaan_muda and keanggotaan_tua > keanggotaan_sedang
        ):
            return "Panen"
        return None

    def GetPPM(self, KategoriUmur, input_ppm: int):
        # Define fuzzy ranges dynamically based on KategoriUmur
        ranges = {
            "Bibit": {
                "rendah": [0, 0, 100, 250],
                "sedang": [200, 300, 300, 400],
                "tinggi": [350, 500, 600, 600],
            },
            "Vegetatif": {
                "rendah": [0, 0, 300, 500],
                "sedang": [440, 650, 650, 860],
                "tinggi": [800, 950, 1200, 1200],
            },
            "Panen": {
                "rendah": [440, 440, 650, 860],
                "sedang": [800, 1000, 1000, 1200],
                "tinggi": [1140, 1300, 1500, 1500],
            },
        }

        if KategoriUmur not in ranges:
            raise ValueError(f"KategoriUmur '{KategoriUmur}' tidak valid.")

        # Set fuzzy PPM for the given category
        self.PPM = ctrl.Antecedent(np.arange(0, 1501, 1), "ppm")
        self.PPM["rendah"] = fuzz.trapmf(
            self.PPM.universe, ranges[KategoriUmur]["rendah"]
        )
        self.PPM["sedang"] = fuzz.trapmf(
            self.PPM.universe, ranges[KategoriUmur]["sedang"]
        )
        self.PPM["tinggi"] = fuzz.trapmf(
            self.PPM.universe, ranges[KategoriUmur]["tinggi"]
        )

        # Define statusPPM
        self.statusPPM = ctrl.Consequent(np.arange(0, 601, 1), "statusPPM")
        self.statusPPM["rendah"] = fuzz.trapmf(
            self.statusPPM.universe, [0, 0, 100, 250]
        )
        self.statusPPM["optimal"] = fuzz.trapmf(
            self.statusPPM.universe, [200, 300, 300, 400]
        )
        self.statusPPM["tinggi"] = fuzz.trapmf(
            self.statusPPM.universe, [350, 500, 600, 600]
        )

        # Define rules
        rule1 = ctrl.Rule(self.PPM["rendah"], self.statusPPM["rendah"])
        rule2 = ctrl.Rule(self.PPM["sedang"], self.statusPPM["optimal"])
        rule3 = ctrl.Rule(self.PPM["tinggi"], self.statusPPM["tinggi"])

        # Create fuzzy control system
        self.outPPM = ctrl.ControlSystem([rule1, rule2, rule3])
        self.outputPPM = ctrl.ControlSystemSimulation(self.outPPM)

        # Input PPM and compute
        self.outputPPM.input["ppm"] = input_ppm
        self.outputPPM.compute()

        # Membership degrees
        m_rendah = fuzz.interp_membership(
            self.PPM.universe, self.PPM["rendah"].mf, input_ppm
        )
        m_sedang = fuzz.interp_membership(
            self.PPM.universe, self.PPM["sedang"].mf, input_ppm
        )
        m_tinggi = fuzz.interp_membership(
            self.PPM.universe, self.PPM["tinggi"].mf, input_ppm
        )

        # Determine the status
        if m_rendah > m_sedang and m_rendah > m_tinggi:
            return "Rendah"
        elif m_sedang > m_rendah and m_sedang > m_tinggi:
            return "Optimal"
        elif m_tinggi > m_sedang and m_tinggi > m_rendah:
            return "Tinggi"
        return None

    def getSuhu(self, input):
        self.suhu["rendah"] = fuzz.trapmf(self.suhu.universe, [0, 0, 10, 24])
        self.suhu["optimal"] = fuzz.trapmf(self.suhu.universe, [22, 26, 26, 30])
        self.suhu["tinggi"] = fuzz.trapmf(self.suhu.universe, [28, 40, 45, 45])

        self.KategoriSuhu = ctrl.Consequent(np.arange(0, 45, 1), "kategori_suhu")
        self.KategoriSuhu["dingin"] = fuzz.trapmf(
            self.KategoriSuhu.universe, [0, 0, 10, 20]
        )
        self.KategoriSuhu["normal"] = fuzz.trapmf(
            self.KategoriSuhu.universe, [18, 24, 24, 30]
        )
        self.KategoriSuhu["panas"] = fuzz.trapmf(
            self.KategoriSuhu.universe, [28, 35, 45, 45]
        )

        rule1 = ctrl.Rule(self.suhu["rendah"], self.KategoriSuhu["dingin"])
        rule2 = ctrl.Rule(self.suhu["optimal"], self.KategoriSuhu["normal"])
        rule3 = ctrl.Rule(self.suhu["tinggi"], self.KategoriSuhu["panas"])

        self.outSuhu = ctrl.ControlSystem([rule1, rule2, rule3])
        self.outputSuhu = ctrl.ControlSystemSimulation(self.outSuhu)

        self.outputSuhu.input["suhu"] = input
        self.outputSuhu.compute()

        m_rendah = fuzz.interp_membership(
            self.suhu.universe, self.suhu["rendah"].mf, input
        )
        m_sedang = fuzz.interp_membership(
            self.suhu.universe, self.suhu["optimal"].mf, input
        )
        m_tinggi = fuzz.interp_membership(
            self.suhu.universe, self.suhu["tinggi"].mf, input
        )

        if m_rendah > m_sedang and m_rendah > m_tinggi:
            return "dingin"
        elif m_sedang > m_rendah and m_sedang > m_tinggi:
            return "normal"
        elif m_tinggi > m_sedang and m_tinggi > m_rendah:
            return "panas"
        return None

    def getKelembaban(self, input):
        self.kelembaban["rendah"] = fuzz.trapmf(
            self.kelembaban.universe, [0, 0, 20, 40]
        )
        self.kelembaban["optimal"] = fuzz.trapmf(
            self.kelembaban.universe, [30, 40, 60, 70]
        )
        self.kelembaban["tinggi"] = fuzz.trapmf(
            self.kelembaban.universe, [60, 70, 100, 100]
        )

        self.KategoriKelembaban = ctrl.Consequent(
            np.arange(0, 45, 1), "kategori_kelembaban"
        )
        self.KategoriKelembaban["kering"] = fuzz.trapmf(
            self.KategoriKelembaban.universe, [0, 0, 10, 20]
        )
        self.KategoriKelembaban["optimal"] = fuzz.trapmf(
            self.KategoriKelembaban.universe, [18, 24, 24, 30]
        )
        self.KategoriKelembaban["lembab"] = fuzz.trapmf(
            self.KategoriKelembaban.universe, [28, 35, 45, 45]
        )

        rule1 = ctrl.Rule(self.kelembaban["rendah"], self.KategoriKelembaban["kering"])
        rule2 = ctrl.Rule(
            self.kelembaban["optimal"], self.KategoriKelembaban["optimal"]
        )
        rule3 = ctrl.Rule(self.kelembaban["tinggi"], self.KategoriKelembaban["lembab"])

        self.outKelembaban = ctrl.ControlSystem([rule1, rule2, rule3])
        self.outputKelembaban = ctrl.ControlSystemSimulation(self.outKelembaban)

        self.outputKelembaban.input["kelembaban"] = input
        self.outputKelembaban.compute()

        m_rendah = fuzz.interp_membership(
            self.kelembaban.universe, self.kelembaban["rendah"].mf, input
        )
        m_sedang = fuzz.interp_membership(
            self.kelembaban.universe, self.kelembaban["optimal"].mf, input
        )
        m_tinggi = fuzz.interp_membership(
            self.kelembaban.universe, self.kelembaban["tinggi"].mf, input
        )

        if m_rendah > m_sedang and m_rendah > m_tinggi:
            return "Kering"
        elif m_sedang > m_rendah and m_sedang > m_tinggi:
            return "Optimal"
        elif m_tinggi > m_sedang and m_tinggi > m_rendah:
            return "Lembab"
        return None

    def getKetinggianAir(self, input):
        # Definisikan range ketinggian air
        self.ketinggianAir["rendah"] = fuzz.trapmf(
            self.ketinggianAir.universe, [16, 19, 20, 20]
        )
        self.ketinggianAir["tinggi"] = fuzz.trapmf(
            self.ketinggianAir.universe, [10, 10, 11, 14]
        )
        self.ketinggianAir["optimal"] = fuzz.trapmf(
            self.ketinggianAir.universe, [13, 15, 15, 17]
        )

        self.KategoriKetinggianAir = ctrl.Consequent(
            np.arange(10, 20, 1), "kategori_ketinggian_air"
        )
        self.KategoriKetinggianAir["tinggi"] = fuzz.trapmf(
            self.KategoriKetinggianAir.universe, [16, 19, 20, 20]
        )
        self.KategoriKetinggianAir["normal"] = fuzz.trapmf(
            self.KategoriKetinggianAir.universe, [13, 15, 15, 17]
        )
        self.KategoriKetinggianAir["rendah"] = fuzz.trapmf(
            self.KategoriKetinggianAir.universe, [10, 10, 11, 14]
        )

        # Aturan fuzzy untuk kategori ketinggian air
        rule1 = ctrl.Rule(
            self.ketinggianAir["rendah"], self.KategoriKetinggianAir["rendah"]
        )
        rule2 = ctrl.Rule(
            self.ketinggianAir["optimal"], self.KategoriKetinggianAir["normal"]
        )
        rule3 = ctrl.Rule(
            self.ketinggianAir["tinggi"], self.KategoriKetinggianAir["tinggi"]
        )

        self.outKetinggianAir = ctrl.ControlSystem([rule1, rule2, rule3])
        self.outputKetinggianAir = ctrl.ControlSystemSimulation(self.outKetinggianAir)

        # Input ketinggian air dan hitung
        self.outputKetinggianAir.input["ketinggian_air"] = input
        self.outputKetinggianAir.compute()

        # Mengambil derajat keanggotaan untuk menentukan kategori
        m_rendah = fuzz.interp_membership(
            self.ketinggianAir.universe, self.ketinggianAir["rendah"].mf, input
        )
        m_normal = fuzz.interp_membership(
            self.ketinggianAir.universe, self.ketinggianAir["optimal"].mf, input
        )
        m_tinggi = fuzz.interp_membership(
            self.ketinggianAir.universe, self.ketinggianAir["tinggi"].mf, input
        )

        # Menentukan kategori ketinggian air berdasarkan keanggotaan tertinggi
        if m_rendah > m_normal and m_rendah > m_tinggi:
            return "Rendah"
        elif m_normal > m_rendah and m_normal > m_tinggi:
            return "Normal"
        elif m_tinggi > m_rendah and m_tinggi > m_normal:
            return "Tinggi"
        return None


class NFTFuzzyControl:
    def __init__(self):
        self._setup_variables()
        self._setup_memberships()

    def _setup_variables(self):
        self.umur = ctrl.Antecedent(np.arange(0, 46, 1), "umur")
        self.ppm = ctrl.Antecedent(np.arange(0, 1501, 1), "ppm")
        self.ketinggian_air = ctrl.Antecedent(np.arange(0, 14, 1), "ketinggian_air")

        self.pompa_sirkulasi = ctrl.Consequent(np.arange(0, 2, 1), "pompa_sirkulasi")
        self.pompa_airbaku = ctrl.Consequent(np.arange(0, 2, 1), "pompa_airbaku")
        self.pompa_nutrisi = ctrl.Consequent(np.arange(0, 2, 1), "pompa_nutrisi")

    def _setup_memberships(self):
        # Age membership
        self.umur["muda"] = fuzz.trapmf(self.umur.universe, [0, 0, 7, 15])
        self.umur["sedang"] = fuzz.trapmf(self.umur.universe, [14, 22, 22, 32])
        self.umur["tua"] = fuzz.trapmf(self.umur.universe, [31, 40, 45, 45])

        # Water level membership
        self.ketinggian_air["rendah"] = fuzz.trapmf(
            self.ketinggian_air.universe, [7, 11, 13, 13]
        )
        self.ketinggian_air["sedang"] = fuzz.trapmf(
            self.ketinggian_air.universe, [4, 6, 6, 8]
        )
        self.ketinggian_air["tinggi"] = fuzz.trapmf(
            self.ketinggian_air.universe, [0, 0, 2, 5]
        )

        # Pump outputs
        for pump in [self.pompa_sirkulasi, self.pompa_airbaku, self.pompa_nutrisi]:
            pump["off"] = fuzz.trimf(pump.universe, [0, 0, 1])
            pump["on"] = fuzz.trimf(pump.universe, [0, 1, 1])

    def _setup_ppm_membership(self, umur_val):
        muda_degree = fuzz.interp_membership(
            self.umur.universe, self.umur["muda"].mf, umur_val
        )
        sedang_degree = fuzz.interp_membership(
            self.umur.universe, self.umur["sedang"].mf, umur_val
        )
        tua_degree = fuzz.interp_membership(
            self.umur.universe, self.umur["tua"].mf, umur_val
        )

        max_degree = max(muda_degree, sedang_degree, tua_degree)

        if max_degree == muda_degree:
            self.ppm["rendah"] = fuzz.trapmf(self.ppm.universe, [0, 0, 100, 250])
            self.ppm["sedang"] = fuzz.trapmf(self.ppm.universe, [200, 300, 300, 400])
            self.ppm["tinggi"] = fuzz.trapmf(self.ppm.universe, [350, 500, 600, 600])
        elif max_degree == sedang_degree:
            self.ppm["rendah"] = fuzz.trapmf(self.ppm.universe, [0, 0, 300, 500])
            self.ppm["sedang"] = fuzz.trapmf(self.ppm.universe, [450, 650, 650, 860])
            self.ppm["tinggi"] = fuzz.trapmf(self.ppm.universe, [800, 950, 1200, 1200])
        else:
            self.ppm["rendah"] = fuzz.trapmf(self.ppm.universe, [440, 440, 650, 850])
            self.ppm["sedang"] = fuzz.trapmf(self.ppm.universe, [800, 1000, 1000, 1200])
            self.ppm["tinggi"] = fuzz.trapmf(
                self.ppm.universe, [1150, 1300, 1500, 1500]
            )

    def _create_rules(self):
        return [
            # Rules for pompa_airbaku
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggian_air["rendah"],
                self.pompa_airbaku["on"],
            ),
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggian_air["sedang"],
                self.pompa_airbaku["off"],
            ),
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggian_air["tinggi"],
                self.pompa_airbaku["off"],
            ),
            ctrl.Rule(
                self.ppm["sedang"] & self.ketinggian_air["rendah"],
                self.pompa_airbaku["on"],
            ),
            ctrl.Rule(
                self.ppm["sedang"] & self.ketinggian_air["sedang"],
                self.pompa_airbaku["off"],
            ),
            ctrl.Rule(
                self.ppm["sedang"] & self.ketinggian_air["tinggi"],
                self.pompa_airbaku["off"],
            ),
            ctrl.Rule(
                self.ppm["tinggi"] & self.ketinggian_air["rendah"],
                self.pompa_airbaku["on"],
            ),
            ctrl.Rule(
                self.ppm["tinggi"] & self.ketinggian_air["sedang"],
                self.pompa_airbaku["off"],
            ),
            ctrl.Rule(
                self.ppm["tinggi"] & self.ketinggian_air["tinggi"],
                self.pompa_airbaku["off"],
            ),
            # Rules for pompa_nutrisi
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggian_air["rendah"],
                self.pompa_nutrisi["on"],
            ),
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggian_air["sedang"],
                self.pompa_nutrisi["on"],
            ),
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggian_air["tinggi"],
                self.pompa_nutrisi["on"],
            ),
            ctrl.Rule(
                self.ppm["sedang"] & self.ketinggian_air["rendah"],
                self.pompa_nutrisi["off"],
            ),
            ctrl.Rule(
                self.ppm["sedang"] & self.ketinggian_air["sedang"],
                self.pompa_nutrisi["off"],
            ),
            ctrl.Rule(
                self.ppm["sedang"] & self.ketinggian_air["tinggi"],
                self.pompa_nutrisi["off"],
            ),
            ctrl.Rule(
                self.ppm["tinggi"] & self.ketinggian_air["rendah"],
                self.pompa_nutrisi["off"],
            ),
            ctrl.Rule(
                self.ppm["tinggi"] & self.ketinggian_air["sedang"],
                self.pompa_nutrisi["off"],
            ),
            ctrl.Rule(
                self.ppm["tinggi"] & self.ketinggian_air["tinggi"],
                self.pompa_nutrisi["off"],
            ),
            # Rules for pompa_sirkulasi
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggian_air["rendah"],
                self.pompa_sirkulasi["off"],
            ),
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggian_air["sedang"],
                self.pompa_sirkulasi["off"],
            ),
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggian_air["tinggi"],
                self.pompa_sirkulasi["off"],
            ),
            ctrl.Rule(
                self.ppm["sedang"] & self.ketinggian_air["rendah"],
                self.pompa_sirkulasi["off"],
            ),
            ctrl.Rule(
                self.ppm["sedang"] & self.ketinggian_air["sedang"],
                self.pompa_sirkulasi["on"],
            ),
            ctrl.Rule(
                self.ppm["sedang"] & self.ketinggian_air["tinggi"],
                self.pompa_sirkulasi["on"],
            ),
            ctrl.Rule(
                self.ppm["tinggi"] & self.ketinggian_air["rendah"],
                self.pompa_sirkulasi["off"],
            ),
            ctrl.Rule(
                self.ppm["tinggi"] & self.ketinggian_air["sedang"],
                self.pompa_sirkulasi["on"],
            ),
            ctrl.Rule(
                self.ppm["tinggi"] & self.ketinggian_air["tinggi"],
                self.pompa_sirkulasi["off"],
            ),
        ]

    def evaluate(self, umur_val, ppm_val, ketinggian_val):
        self._setup_ppm_membership(umur_val)
        control_system = ctrl.ControlSystem(self._create_rules())
        system = ctrl.ControlSystemSimulation(control_system)

        system.input["ppm"] = ppm_val
        system.input["ketinggian_air"] = ketinggian_val

        try:
            system.compute()
            return {
                "pompa_sirkulasi": (
                    "ON" if system.output["pompa_sirkulasi"] >= 0.5 else "OFF"
                ),
                "pompa_airbaku": (
                    "ON" if system.output["pompa_airbaku"] >= 0.5 else "OFF"
                ),
                "pompa_nutrisi": (
                    "ON" if system.output["pompa_nutrisi"] >= 0.5 else "OFF"
                ),
            }
        except:
            return "Error in computation. Please check input values."


app = Flask(__name__)
selada = Selada()
controller = NFTFuzzyControl()

# # Variabel untuk menyimpan data
processed_data = {}


@app.route("/fuzzy", methods=["POST"])
def evaluate():
    data = request.get_json()
    try:
        umur = float(data["umur"])
        ppm = float(data["ppm"])
        ketinggian = float(data["ketinggian"])

        result = controller.evaluate(umur, ppm, ketinggian)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    tanaman = data.get("tanaman")
    umur = data.get("umur")

    if tanaman is None or umur is None:
        return (
            jsonify({"error": "Mohon sertakan tanaman dan umur dalam permintaan"}),
            400,
        )
    global processed_data
    processed_data.update({"tanaman": tanaman, "umur": umur})

    return jsonify(processed_data)


@app.route("/get_status", methods=["POST"])
def get_status():
    global processed_data
    global selada
    data = request.json
    umur = data.get("umur")
    ppm = data.get("ppm")
    suhu = data.get("suhu")
    kelembaban = data.get("kelembaban")
    ketinggian_air = data.get("ketinggian_air")

    if (
        umur is None
        or ppm is None
        or suhu is None
        or kelembaban is None
        or ketinggian_air is None
    ):
        return jsonify({"error": "Mohon lengkapi data dalam permintaan"}), 400

    kategori_umur = selada.getKategoriUmur(umur)
    status_ppm = selada.GetPPM(kategori_umur, ppm)
    status_suhu = selada.getSuhu(suhu)
    status_kelembaban = selada.getKelembaban(kelembaban)
    status_ketinggian_air = selada.getKetinggianAir(ketinggian_air)

    processed_data.update(
        {
            "kategori_umur": kategori_umur,
            "status_ppm": status_ppm,
            "status_suhu": status_suhu,
            "status_kelembaban": status_kelembaban,
            "status_ketinggian_air": status_ketinggian_air,
            "umur": umur,
        }
    )
    return jsonify(processed_data)


@app.route("/get_data", methods=["GET"])
def get_data():
    if not processed_data:
        return jsonify({"error": "Tidak ada data yang tersedia"}), 404

    return jsonify(processed_data)


if __name__ == "__main__":
    app.run(debug=True)
