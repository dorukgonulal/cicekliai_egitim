# Gerekli kütüphanelerimizi içe aktarıyoruz.
# pip install pyqt5 matplotlib numpy
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Uygulamanın başlangıç değerlerini belirliyoruz.
initial_weight = 70  # Başlangıç ağırlığı (kg)
initial_height = 170  # Başlangıç boyu (cm)
initial_shoe_size = 38  # Başlangıç ayakkabı numarası (EU)

# Boydan ağırlık tahmini yapan bir fonksiyon tanımlıyoruz.
# Mantık olarak boy ile doğru orantılı bir ağırlık tahmini yapıyoruz.
def height_to_weight(height):
    return 45 + (5 / 6) * (height - 140)  # Boy arttıkça ağırlık da artar

# Ağırlıktan boy tahmini yapacak bir fonksiyon.
# Bu fonksiyon ağırlığa göre tahmini bir boy değeri döndürecek.
def weight_to_height(weight):
    return 140 + (6 / 5) * (weight - 45)

# Boya göre ayakkabı numarası tahmini yapan fonksiyon
# Ayakkabı numarasını belli bir aralıkta sınırlıyoruz ve en yakın tam ya da .5 değerine yuvarlıyoruz.
def height_to_shoes(height):
    shoe_size = 33 + (42 - 33) * (height - 150) / (200 - 150)  # Boya göre hesapla
    shoe_size = max(33, round(shoe_size * 2) / 2)  # 33'ün altına düşmemesi ve yarımlı yuvarlanması için
    return shoe_size

# Ana uygulama penceresi sınıfımızı tanımlıyoruz.
class ChainRuleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Pencerenin başlığını ayarlıyoruz ve boyutlandırıyoruz.
        self.setWindowTitle("The Chain Rule")
        self.setGeometry(100, 100, 800, 400)
        
        # Matplotlib ile çizim yapacağımız figürü oluşturuyoruz.
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        
        # Ana widget ve yerleşim ayarları
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # İlk grafik: Ağırlık-Boy ilişkisini gösterecek
        self.ax1 = self.figure.add_subplot(121)
        self.ax1.set_title("Weight - Height")  # Başlık
        self.ax1.set_xlabel("Weight (kg)")  # X ekseni etiketi
        self.ax1.set_ylabel("Height (cm)")  # Y ekseni etiketi
        self.ax1.set_xlim(40, 100)  # X ekseni sınırları
        self.ax1.set_ylim(140, 200)  # Y ekseni sınırları
        self.ax1.grid(True, linestyle="--", alpha=0.6)  # Kareli arka plan
        self.weight_point, = self.ax1.plot(initial_weight, initial_height, "ro", markersize=8)  # Başlangıç noktası
        self.weight_vline = self.ax1.axvline(x=initial_weight, color="gray", linestyle="--")  # Dikey çizgi
        self.weight_hline = self.ax1.axhline(y=initial_height, color="gray", linestyle="--")  # Yatay çizgi

        # Mantıklı bir eğri çizmek için ağırlık-boya göre değerler
        height_values = np.linspace(140, 200, 500)
        weight_values = height_to_weight(height_values)
        self.ax1.plot(weight_values, height_values, "g--", alpha=0.7)  # Yeşil çizgi olarak çiziyoruz

        # İkinci grafik: Boy-Ayakkabı numarası ilişkisini gösterecek
        self.ax2 = self.figure.add_subplot(122)
        self.ax2.set_title("Height - Shoe Size")
        self.ax2.set_xlabel("Height (cm)")
        self.ax2.set_ylabel("Shoe Size (EU)")
        self.ax2.set_xlim(140, 200)
        self.ax2.set_ylim(33, 42)
        self.ax2.grid(True, linestyle="--", alpha=0.6)
        self.shoe_point, = self.ax2.plot(initial_height, initial_shoe_size, "bo", markersize=8)  # Başlangıç noktası
        self.shoe_vline = self.ax2.axvline(x=initial_height, color="gray", linestyle="--")  # Dikey çizgi
        self.shoe_hline = self.ax2.axhline(y=initial_shoe_size, color="gray", linestyle="--")  # Yatay çizgi

        # Mouse sürükleme olaylarını ayarlıyoruz
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.dragging = False  # Sürükleme durumunu kontrol eden değişken

    # Fare tıklama olayı - kullanıcının noktayı sürüklemeye başladığını tespit ediyoruz
    def on_press(self, event):
        if event.inaxes == self.ax1:  # Sadece ilk grafikte tıklama varsa
            contains, _ = self.weight_point.contains(event)  # Tıklama noktada mı?
            if contains:
                self.dragging = True  # Sürüklemeye başla

    # Fare bırakma olayı - sürüklemenin sonlandığını işaretliyoruz
    def on_release(self, event):
        self.dragging = False  # Sürüklemeyi bırak

    # Mouse hareket ettikçe noktanın yerini değiştiriyoruz
    def on_motion(self, event):
        if self.dragging and event.inaxes == self.ax1:  # Sadece sürükleniyorsa ve ilk grafikteyse
            # Hareketi sınırlamak için boy aralığını kontrol ediyoruz
            new_height = np.clip(event.ydata, 140, 200)
            new_weight = height_to_weight(new_height)  # Yeni ağırlığı hesapla
            new_shoe_size = height_to_shoes(new_height)  # Yeni ayakkabı numarasını hesapla

            # İlk grafik: Ağırlık ve boy ilişkisini güncelle
            self.weight_point.set_data(new_weight, new_height)
            self.weight_vline.set_xdata(new_weight)  # Dikey çizgiyi güncelle
            self.weight_hline.set_ydata(new_height)  # Yatay çizgiyi güncelle
            self.ax1.set_title(f"Weight: {new_weight:.1f} kg, Height: {new_height:.1f} cm")  # Başlık güncelle

            # İkinci grafik: Boy ve ayakkabı numarasını güncelle
            self.shoe_point.set_data(new_height, new_shoe_size)
            self.shoe_vline.set_xdata(new_height)  # Dikey çizgiyi güncelle
            self.shoe_hline.set_ydata(new_shoe_size)  # Yatay çizgiyi güncelle
            self.ax2.set_title(f"Height: {new_height:.1f} cm, Shoe Size: {new_shoe_size:.1f}")  # Başlık güncelle

            self.canvas.draw_idle()  # Anında değişiklikleri göstermek için yeniden çiz

# Uygulamayı başlatıyoruz
app = QApplication(sys.argv)
window = ChainRuleApp()
window.show()
sys.exit(app.exec_())  # Uygulama çıkış komutu
