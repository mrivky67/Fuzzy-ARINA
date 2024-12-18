import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from flask import Flask, request, jsonify


class Selada:
    def __init__(self):
        # Definisi variabel fuzzy
        self.umur = ctrl.Antecedent(np.arange(0, 45, 1), "umur")
        self.suhu = ctrl.Antecedent(np.arange(0, 45, 1), "suhu")
        self.ketinggianAir = ctrl.Antecedent(np.arange(0, 20, 1), "ketinggian_air")
        self.ppm = ctrl.Antecedent(np.arange(0, 1500, 1), "ppm")

        # Definisi fuzzy sets
        self._definisi_umur()
        self._definisi_suhu()
        self._definisi_ketinggian_air()
        self._definisi_ppm()

    def _definisi_umur(self):
        self.umur["muda"] = fuzz.trapmf(self.umur.universe, [0, 0, 7, 15])
        self.umur["sedang"] = fuzz.trapmf(self.umur.universe, [14, 22, 22, 32])
        self.umur["tua"] = fuzz.trapmf(self.umur.universe, [31, 40, 45, 45])

    def _definisi_suhu(self):
        self.suhu["rendah"] = fuzz.trapmf(self.suhu.universe, [0, 0, 10, 24])
        self.suhu["optimal"] = fuzz.trapmf(self.suhu.universe, [22, 26, 26, 30])
        self.suhu["tinggi"] = fuzz.trapmf(self.suhu.universe, [28, 40, 45, 45])

    def _definisi_ketinggian_air(self):
        self.ketinggianAir["rendah"] = fuzz.trapmf(
            self.ketinggianAir.universe, [0, 0, 5, 10]
        )
        self.ketinggianAir["optimal"] = fuzz.trapmf(
            self.ketinggianAir.universe, [5, 10, 10, 15]
        )
        self.ketinggianAir["tinggi"] = fuzz.trapmf(
            self.ketinggianAir.universe, [10, 15, 20, 20]
        )

    def _definisi_ppm(self):
        self.ppm["rendah"] = fuzz.trapmf(self.ppm.universe, [0, 0, 300, 500])
        self.ppm["sedang"] = fuzz.trapmf(self.ppm.universe, [440, 650, 650, 860])
        self.ppm["tinggi"] = fuzz.trapmf(self.ppm.universe, [800, 950, 1200, 1200])

    def getKategoriUmur(self, input_umur):
        m_umur = {
            "muda": fuzz.interp_membership(
                self.umur.universe, self.umur["muda"].mf, input_umur
            ),
            "sedang": fuzz.interp_membership(
                self.umur.universe, self.umur["sedang"].mf, input_umur
            ),
            "tua": fuzz.interp_membership(
                self.umur.universe, self.umur["tua"].mf, input_umur
            ),
        }

        kategori = max(m_umur, key=m_umur.get)

        if kategori == "muda":
            return "Bibit"
        elif kategori == "sedang":
            return "Vegetatif"
        else:
            return "Panen"

    def get_status_ppm(self, kategori_umur, input_ppm):
        if kategori_umur == "Bibit":
            ppm_range = [0, 600]
        elif kategori_umur == "Vegetatif":
            ppm_range = [0, 1200]
        else:  # Panen
            ppm_range = [440, 1500]

        self.ppm.universe = np.arange(ppm_range[0], ppm_range[1], 1)
        self._definisi_ppm()

        m_ppm = {
            "rendah": fuzz.interp_membership(
                self.ppm.universe, self.ppm["rendah"].mf, input_ppm
            ),
            "sedang": fuzz.interp_membership(
                self.ppm.universe, self.ppm["sedang"].mf, input_ppm
            ),
            "tinggi": fuzz.interp_membership(
                self.ppm.universe, self.ppm["tinggi"].mf, input_ppm
            ),
        }

        status = max(m_ppm, key=m_ppm.get)
        return status

    def MainSistem(self, umur, ppm, ketinggian_air):
        # Tentukan kategori umur dan status ppm
        kategori_umur = self.getKategoriUmur(umur)
        status_ppm = self.get_status_ppm(kategori_umur, ppm)

        # Definisi output fuzzy
        pompa_nutrisi = ctrl.Consequent(np.arange(0, 1, 0.1), "pompa_nutrisi")
        pompa_sirkulasi = ctrl.Consequent(np.arange(0, 1, 0.1), "pompa_sirkulasi")
        pompa_airbaku = ctrl.Consequent(np.arange(0, 1, 0.1), "pompa_airbaku")

        # Definisi keanggotaan untuk output
        pompa_nutrisi["off"] = fuzz.trimf(pompa_nutrisi.universe, [0, 0, 0.5])
        pompa_nutrisi["on"] = fuzz.trimf(pompa_nutrisi.universe, [0.5, 1, 1])

        pompa_sirkulasi["off"] = fuzz.trimf(pompa_sirkulasi.universe, [0, 0, 0.5])
        pompa_sirkulasi["on"] = fuzz.trimf(pompa_sirkulasi.universe, [0.5, 1, 1])

        pompa_airbaku["off"] = fuzz.trimf(pompa_airbaku.universe, [0, 0, 0.5])
        pompa_airbaku["on"] = fuzz.trimf(pompa_airbaku.universe, [0.5, 1, 1])

        # Fuzzy rules untuk kontrol pompa
        rules = [
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggianAir["rendah"],
                [pompa_nutrisi["on"], pompa_sirkulasi["off"], pompa_airbaku["on"]],
            ),
            ctrl.Rule(
                self.ppm["rendah"] & self.ketinggianAir["optimal"],
                [pompa_nutrisi["on"], pompa_sirkulasi["off"], pompa_airbaku["off"]],
            ),
            ctrl.Rule(
                self.ppm["sedang"] & self.ketinggianAir["optimal"],
                [pompa_nutrisi["off"], pompa_sirkulasi["on"], pompa_airbaku["off"]],
            ),
        ]

        # Sistem kontrol
        control_system = ctrl.ControlSystem(rules)
        simulasi = ctrl.ControlSystemSimulation(control_system)

        # Masukkan input
        simulasi.input["ppm"] = ppm
        simulasi.input["ketinggian_air"] = ketinggian_air

        # Hitung output
        simulasi.compute()

        return {
            "pompa_nutrisi": simulasi.output["pompa_nutrisi"],
            "pompa_sirkulasi": simulasi.output["pompa_sirkulasi"],
            "pompa_airbaku": simulasi.output["pompa_airbaku"],
        }


app = Flask(__name__)
selada = Selada()

processed_data = {}


@app.route("/fuzzy", methods=["POST"])
def fuzzy():
    data = request.get_json()
    umur = data["umur"]
    ppm = data["ppm"]
    ketinggian_air = data["ketinggian_air"]
    result = selada.MainSistem(umur, ppm, ketinggian_air)
    return jsonify(result)


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
    processed_data = {"tanaman": tanaman, "umur": umur}

    return jsonify(processed_data)


@app.route("/get_data", methods=["GET"])
def get_data():
    if not processed_data:
        return jsonify({"error": "Tidak ada data yang tersedia"}), 404

    return jsonify(processed_data)


if __name__ == "__main__":
    app.run(debug=True)
