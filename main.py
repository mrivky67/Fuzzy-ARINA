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
        if KategoriUmur == "Bibit":
            self.PPM = ctrl.Antecedent(np.arange(0, 600, 1), "ppm")
            self.PPM["rendah"] = fuzz.trapmf(self.PPM.universe, [0, 0, 100, 250])
            self.PPM["sedang"] = fuzz.trapmf(self.PPM.universe, [200, 300, 300, 400])
            self.PPM["tinggi"] = fuzz.trapmf(self.PPM.universe, [350, 500, 600, 600])

            self.statusPPM = ctrl.Consequent(np.arange(0, 600, 1), "statusPPM")
            self.statusPPM["rendah"] = fuzz.trapmf(
                self.statusPPM.universe, [0, 0, 100, 250]
            )
            self.statusPPM["optimal"] = fuzz.trapmf(
                self.statusPPM.universe, [200, 300, 300, 400]
            )
            self.statusPPM["tinggi"] = fuzz.trapmf(
                self.statusPPM.universe, [350, 500, 600, 600]
            )

            rule1 = ctrl.Rule(self.PPM["rendah"], self.statusPPM["rendah"])
            rule2 = ctrl.Rule(self.PPM["sedang"], self.statusPPM["optimal"])
            rule3 = ctrl.Rule(self.PPM["tinggi"], self.statusPPM["tinggi"])

            self.outPPM = ctrl.ControlSystem([rule1, rule2, rule3])
            self.outputPPM = ctrl.ControlSystemSimulation(self.outPPM)

            self.outputPPM.input["ppm"] = input_ppm
            self.outputPPM.compute()

            m_rendah = fuzz.interp_membership(
                self.PPM.universe, self.PPM["rendah"].mf, input_ppm
            )
            m_sedang = fuzz.interp_membership(
                self.PPM.universe, self.PPM["sedang"].mf, input_ppm
            )
            m_tinggi = fuzz.interp_membership(
                self.PPM.universe, self.PPM["tinggi"].mf, input_ppm
            )

            if m_rendah > m_sedang and m_rendah > m_tinggi:
                return "Rendah"
            elif m_sedang > m_rendah and m_sedang > m_tinggi:
                return "Optimal"
            elif m_tinggi > m_sedang and m_tinggi > m_rendah:
                return "Tinggi"
            return None

        elif KategoriUmur == "Vegetatif":
            self.PPM = ctrl.Antecedent(np.arange(0, 1200, 1), "ppm")
            self.PPM["rendah"] = fuzz.trapmf(self.PPM.universe, [0, 0, 300, 500])
            self.PPM["sedang"] = fuzz.trapmf(self.PPM.universe, [440, 650, 650, 860])
            self.PPM["tinggi"] = fuzz.trapmf(self.PPM.universe, [800, 950, 1200, 1200])

            self.statusPPM = ctrl.Consequent(np.arange(0, 600, 1), "statusPPM")
            self.statusPPM["rendah"] = fuzz.trapmf(
                self.statusPPM.universe, [0, 0, 100, 250]
            )
            self.statusPPM["optimal"] = fuzz.trapmf(
                self.statusPPM.universe, [200, 300, 300, 400]
            )
            self.statusPPM["tinggi"] = fuzz.trapmf(
                self.statusPPM.universe, [350, 500, 600, 600]
            )

            rule1 = ctrl.Rule(self.PPM["rendah"], self.statusPPM["rendah"])
            rule2 = ctrl.Rule(self.PPM["sedang"], self.statusPPM["optimal"])
            rule3 = ctrl.Rule(self.PPM["tinggi"], self.statusPPM["tinggi"])

            self.outPPM = ctrl.ControlSystem([rule1, rule2, rule3])
            self.outputPPM = ctrl.ControlSystemSimulation(self.outPPM)

            self.outputPPM.input["ppm"] = input_ppm
            self.outputPPM.compute()
            self.PPM.view(sim=self.outputPPM)

            m_rendah = fuzz.interp_membership(
                self.PPM.universe, self.PPM["rendah"].mf, input_ppm
            )
            m_sedang = fuzz.interp_membership(
                self.PPM.universe, self.PPM["sedang"].mf, input_ppm
            )
            m_tinggi = fuzz.interp_membership(
                self.PPM.universe, self.PPM["tinggi"].mf, input_ppm
            )

            if m_rendah > m_sedang and m_rendah > m_tinggi:
                return "Rendah"
            elif m_sedang > m_rendah and m_sedang > m_tinggi:
                return "Optimal"
            elif m_tinggi > m_sedang and m_tinggi > m_rendah:
                return "Tinggi"
            return None

        elif KategoriUmur == "Panen":
            self.PPM = ctrl.Antecedent(np.arange(440, 1500, 1), "ppm")
            self.PPM["rendah"] = fuzz.trapmf(self.PPM.universe, [440, 440, 650, 860])
            self.PPM["sedang"] = fuzz.trapmf(self.PPM.universe, [800, 1000, 1000, 1200])
            self.PPM["tinggi"] = fuzz.trapmf(
                self.PPM.universe, [1140, 1300, 1500, 1500]
            )

            self.statusPPM = ctrl.Consequent(np.arange(0, 600, 1), "statusPPM")
            self.statusPPM["rendah"] = fuzz.trapmf(
                self.statusPPM.universe, [0, 0, 100, 250]
            )
            self.statusPPM["optimal"] = fuzz.trapmf(
                self.statusPPM.universe, [200, 300, 300, 400]
            )
            self.statusPPM["tinggi"] = fuzz.trapmf(
                self.statusPPM.universe, [350, 500, 600, 600]
            )

            rule1 = ctrl.Rule(self.PPM["rendah"], self.statusPPM["rendah"])
            rule2 = ctrl.Rule(self.PPM["sedang"], self.statusPPM["optimal"])
            rule3 = ctrl.Rule(self.PPM["tinggi"], self.statusPPM["tinggi"])

            self.outPPM = ctrl.ControlSystem([rule1, rule2, rule3])
            self.outputPPM = ctrl.ControlSystemSimulation(self.outPPM)

            self.outputPPM.input["ppm"] = input_ppm
            self.outputPPM.compute()

            m_rendah = fuzz.interp_membership(
                self.PPM.universe, self.PPM["rendah"].mf, input_ppm
            )
            m_sedang = fuzz.interp_membership(
                self.PPM.universe, self.PPM["sedang"].mf, input_ppm
            )
            m_tinggi = fuzz.interp_membership(
                self.PPM.universe, self.PPM["tinggi"].mf, input_ppm
            )

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

        self.KategoriSuhu = ctrl.Consequent(np.arange(0, 45, 1))
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

    def MainSistem(
        self,
        input_kategoriUmur,
        input_kategoriSuhu,
        input_kategoriKelembaban,
        input_kategoriPpm,
    ):
        self.NutrisiA = ctrl.Consequent(np.arange(0, 100, 1), "nutrisi_a")
        self.NutrisiB = ctrl.Consequent(np.arange(0, 100, 1), "nutrisi_b")
        self.KecepatanMotor = ctrl.Consequent(np.arange(0, 100, 1), "kecepatan_motor")

        rule1 = ctrl.Rule(
            self.kategoriUmur[input_kategoriUmur]
            & self.kategoriSuhu[input_kategoriSuhu]
            & self.kelembaban[input_kategoriKelembaban]
            & self.statusPPM[input_kategoriPpm],
            self.NutrisiA["sedikit"]
            & self.NutrisiB["sedikit"]
            & self.KecepatanMotor["rendah"],
        )
        rule2 = ctrl.Rule(
            self.kategoriUmur[input_kategoriUmur]
            & self.kategoriSuhu[input_kategoriSuhu]
            & self.kelembaban[input_kategoriKelembaban]
            & self.statusPPM[input_kategoriPpm],
            self.NutrisiA["sedikit"]
            & self.NutrisiB["sedikit"]
            & self.KecepatanMotor["sedang"],
        )
        rule3 = ctrl.Rule(
            self.kategoriUmur[input_kategoriUmur]
            & self.kategoriSuhu[input_kategoriSuhu]
            & self.kelembaban[input_kategoriKelembaban]
            & self.statusPPM[input_kategoriPpm],
            self.NutrisiA["sedikit"]
            & self.NutrisiB["sedikit"]
            & self.KecepatanMotor["tinggi"],
        )
        rule4 = ctrl.Rule(
            self.kategoriUmur[input_kategoriUmur]
            & self.kategoriSuhu[input_kategoriSuhu]
            & self.kelembaban[input_kategoriKelembaban]
            & self.statusPPM[input_kategoriPpm],
            self.NutrisiA["sedikit"]
            & self.NutrisiB["sedikit"]
            & self.KecepatanMotor["rendah"],
        )
        rule5 = ctrl.Rule(
            self.kategoriUmur[input_kategoriUmur]
            & self.kategoriSuhu[input_kategoriSuhu]
            & self.kelembaban[input_kategoriKelembaban]
            & self.statusPPM[input_kategoriPpm],
            self.NutrisiA["sedikit"]
            & self.NutrisiB["sedikit"]
            & self.KecepatanMotor["sedang"],
        )
        rule6 = ctrl.Rule(
            self.kategoriUmur[input_kategoriUmur]
            & self.kategoriSuhu[input_kategoriSuhu]
            & self.kelembaban[input_kategoriKelembaban]
            & self.statusPPM[input_kategoriPpm],
            self.NutrisiA["sedikit"]
            & self.NutrisiB["sedikit"]
            & self.KecepatanMotor["tinggi"],
        )
        rule7 = ctrl.Rule(
            self.kategoriUmur[input_kategoriUmur]
            & self.kategoriSuhu[input_kategoriSuhu]
            & self.kelembaban[input_kategoriKelembaban]
            & self.statusPPM[input_kategoriPpm],
            self.NutrisiA["sedikit"]
            & self.NutrisiB["sedikit"]
            & self.KecepatanMotor["rendah"],
        )
        rule8 = ctrl.Rule(
            self.kategoriUmur[input_kategoriUmur]
            & self.kategoriSuhu[input_kategoriSuhu]
            & self.kelembaban[input_kategoriKelembaban]
            & self.statusPPM[input_kategoriPpm],
            self.NutrisiA["sedikit"]
            & self.NutrisiB["sedikit"]
            & self.KecepatanMotor["sedang"],
        )


# selada = Selada()
# umur_selada = selada.getKategoriUmur(12)
# ppm_selada = selada.GetPPM(umur_selada, 452)
# selada.view()
# plt.show()
# print(f"{umur_selada} , {ppm_selada}")
app = Flask(__name__)
selada = Selada()


# Variabel untuk menyimpan data
processed_data = {}


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


@app.route("/get_status", methods=["POST"])
def get_status():
    global processed_data
    data = request.json
    umur = processed_data.get("umur")
    ppm = data.get("ppm")

    if ppm is None:
        return jsonify({"error": "Mohon sertakan umur dan ppm dalam permintaan"}), 400

    kategori_umur = selada.getKategoriUmur(umur)
    status_ppm = selada.GetPPM(kategori_umur, ppm)

    processed_data = {
        "kategori_umur": kategori_umur,
        "status_ppm": status_ppm,
    }

    return jsonify(processed_data)


@app.route("/get_data", methods=["GET"])
def get_data():
    if not processed_data:
        return jsonify({"error": "Tidak ada data yang tersedia"}), 404

    return jsonify(processed_data)


# Jalankan aplikasi Flask
if __name__ == "__main__":
    app.run(debug=True)
