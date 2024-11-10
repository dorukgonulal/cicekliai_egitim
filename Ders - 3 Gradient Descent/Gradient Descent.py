import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# pip install pyqt5 matplotlib numpy


class GradientDescentVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        # İlk intercept ve slope değerlerini belirle
        self.initial_intercept = 0
        self.slope = 0.64  # Sabit slope değeri
        self.learning_rate = 0.1
        self.iterations = 100
        self.current_iteration = 0
        self.intercepts = []
        self.animation_running = False  # Animasyonun çalışıp çalışmadığını izlemek için

        # Pencere ve grafik ayarlarını yap
        self.setWindowTitle("Gradient Descent Visualizer")
        self.setGeometry(100, 100, 1200, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Grafik alanı
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        self.ax1 = self.figure.add_subplot(121)  # Sol grafik: X, Y noktaları ve çizgi
        self.ax2 = self.figure.add_subplot(122)  # Sağ grafik: Sum of Squared Residuals

        # Düğme alanı
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Başla")
        self.reset_line_button = QPushButton("Sıfırla")
        self.randomize_button = QPushButton("Rastgele Değiştir")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.reset_line_button)
        button_layout.addWidget(self.randomize_button)
        self.start_button.clicked.connect(self.start_animation)
        self.reset_line_button.clicked.connect(self.reset_graphs)
        self.randomize_button.clicked.connect(self.randomize_points)
        main_layout.addLayout(button_layout)

        # Timer ayarları (animasyon için)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gradient_descent)

        # Başlangıç grafiği ayarları
        self.randomize_points()

    def reset_graphs(self):
        # Animasyonu durdur ve tüm grafikleri sıfırla
        self.timer.stop()
        self.animation_running = False
        self.intercept = self.initial_intercept
        self.current_iteration = 0
        self.intercepts.clear()

        # Sol grafiği (noktalar) sıfırla
        self.ax1.clear()
        self.ax1.scatter(self.X, self.Y, color="green", label="Data Points")
        self.ax1.set_title("Data Points")
        self.ax1.set_xlim(0, 10)  # X eksenini 0-10 aralığında sabitle
        self.ax1.set_ylim(0, 10)  # Y eksenini 0-10 aralığında sabitle
        self.ax1.legend()
        self.ax1.set_xticks([])  # X eksenindeki sayıları kaldır
        self.ax1.set_yticks([])  # Y eksenindeki sayıları kaldır

        # Sağ grafiği (hata eğrisi) sıfırla
        self.ax2.clear()
        self.ax2.set_title("Sum of Squared Residuals vs Intercept")
        self.ax2.set_xlabel("Intercept")
        self.ax2.set_ylabel("Error")

        self.canvas.draw()

    def randomize_points(self):
        # X noktalarını geniş bir aralıkta, Y noktalarını ise daha dar bir aralıkta rastgele seç
        self.X = np.random.uniform(2, 8, 3)  # X ekseninde 2-8 arası 3 rastgele nokta
        base_y = np.random.uniform(4, 6)  # Orta bir y ekseni konumu seç (örneğin, 5 civarı)
        self.Y = base_y + np.random.uniform(-0.5, 0.5, 3)  # Y ekseninde nokta yakınlığı: 4.5-5.5 arası 3 nokta
        self.reset_graphs()  # Grafik ve çizgiyi sıfırla

    def start_animation(self):
        # Animasyonu başlat (Başla butonuna tıklanmadığı sürece animasyon başlamaz)
        if not self.animation_running:
            self.animation_running = True
            self.timer.start(100)  # Her 100 ms'de bir iterasyon çalıştır

    def update_gradient_descent(self):
        # Maksimum iterasyon sayısına ulaşıldığında timer durdur
        if self.current_iteration >= self.iterations:
            self.timer.stop()
            self.animation_running = False
            return

        # Tahmin edilen Y değerlerini hesapla
        Y_pred = self.slope * self.X + self.intercept

        # Kare hatalar toplamını hesapla
        error = np.sum((self.Y - Y_pred) ** 2) / len(self.Y)
        self.intercepts.append((self.intercept, error))

        # Gradyanı hesapla ve intercept'i güncelle
        intercept_gradient = -2 * np.sum(self.Y - Y_pred) / len(self.Y)
        self.intercept -= self.learning_rate * intercept_gradient

        # Sol grafik: X, Y noktaları ve en uygun çizgi
        self.plot_current_line()

        # Sağ grafik: U şeklinde Sum of Squared Residuals grafiği üzerine iterasyonları işaretle
        intercept_values = [pt[0] for pt in self.intercepts]
        errors = [pt[1] for pt in self.intercepts]
        self.ax2.clear()
        self.ax2.plot(intercept_values, errors, 'ro-')  # Kırmızı noktalarla her iterasyondaki intercept değerini işaretle
        self.ax2.set_title("Sum of Squared Residuals vs Intercept")
        self.ax2.set_xlabel("Intercept")
        self.ax2.set_ylabel("Error")

        # Grafiklerin güncellenmesi
        self.canvas.draw()

        # İterasyon sayısını artır
        self.current_iteration += 1

    def plot_current_line(self):
        # Sol grafikte X ve Y noktaları ile sonsuz gibi görünen çizgiyi göster
        self.ax1.clear()
        self.ax1.scatter(self.X, self.Y, color="green", label="Data Points")

        # Sonsuz çizgi görünümü için grafiğin sınırları kadar uzatılmış bir çizgi
        x_vals = np.array([0, 10])  # X eksen sınırları
        y_vals = self.slope * x_vals + self.intercept  # Y değerlerini hesapla
        self.ax1.plot(x_vals, y_vals, color="blue", label="Current Line")

        self.ax1.set_title("Best Line")
        self.ax1.set_xlim(0, 10)  # X eksenini 0-10 aralığında sabitle
        self.ax1.set_ylim(0, 10)  # Y eksenini 0-10 aralığında sabitle
        self.ax1.legend()
        self.ax1.set_xticks([])  # X eksenindeki sayıları kaldır
        self.ax1.set_yticks([])  # Y eksenindeki sayıları kaldır

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GradientDescentVisualizer()
    window.show()
    sys.exit(app.exec_())
