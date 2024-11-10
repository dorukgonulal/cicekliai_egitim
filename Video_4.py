import numpy as np
import matplotlib.pyplot as plt

# Örnek için bazı sentetik veriler oluştur
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # X için 100 veri noktası
true_slope = 3.5
true_intercept = 2
noise = np.random.randn(100, 1)  # Rastgele gürültü
y = true_slope * X + true_intercept + noise  # Gürültülü doğrusal ilişki

# Parametreleri başlat
intercept = 0.0  # Başlangıç intercept değeri
slope = 0.0  # Başlangıç eğim değeri
learning_rate = 0.1
iterations = 100

# Artık değerleri ve karesel toplamı hesaplayan fonksiyon
def calculate_residuals(X, y, intercept, slope):
    predictions = intercept + slope * X
    residuals = y - predictions
    squared_residuals = residuals ** 2
    return np.sum(squared_residuals)

# Gradient Descent Optimizasyonu
loss_history = []
for _ in range(iterations):
    predictions = intercept + slope * X
    residuals = y - predictions
    
    # Intercept ve eğim için gradyanları hesapla
    intercept_gradient = -2 * np.sum(residuals) / len(X)
    slope_gradient = -2 * np.sum(residuals * X) / len(X)
    
    # Intercept ve eğim güncelle
    intercept -= learning_rate * intercept_gradient
    slope -= learning_rate * slope_gradient
    
    # Her iterasyondaki kaybı kaydet
    loss = calculate_residuals(X, y, intercept, slope)
    loss_history.append(loss)

# Optimizasyon sonuçlarını görselleştir
plt.plot(loss_history, label="Kayıp (Karesel Artıkların Toplamı)")
plt.xlabel("İterasyonlar")
plt.ylabel("Kayıp")
plt.legend()
plt.title("Gradient Descent Optimizasyonu")
plt.show()

print(f"Optimize edilmiş intercept: {intercept}")
print(f"Optimize edilmiş eğim: {slope}")

# Stochastic Gradient Descent (SGD)
batch_size = 10  # SGD için veri alt kümesi boyutu
learning_rate = 0.1
iterations = 100
intercept = 0.0  # SGD için intercept sıfırlama
slope = 0.0  # SGD için eğim sıfırlama
loss_history_sgd = []

for _ in range(iterations):
    # Rastgele bir veri alt kümesi seç
    indices = np.random.randint(0, len(X), batch_size)
    X_batch = X[indices]
    y_batch = y[indices]
    
    predictions = intercept + slope * X_batch
    residuals = y_batch - predictions
    
    # Alt kümede gradyanları hesapla
    intercept_gradient = -2 * np.sum(residuals) / batch_size
    slope_gradient = -2 * np.sum(residuals * X_batch) / batch_size
    
    # Parametreleri güncelle
    intercept -= learning_rate * intercept_gradient
    slope -= learning_rate * slope_gradient
    
    # Alt küme için kaybı hesapla
    loss = calculate_residuals(X_batch, y_batch, intercept, slope)
    loss_history_sgd.append(loss)

# SGD sonuçlarını görselleştir
plt.plot(loss_history_sgd, label="SGD Kayıp (Karesel Artıkların Toplamı)", color='orange')
plt.xlabel("İterasyonlar")
plt.ylabel("Kayıp")
plt.legend()
plt.title("Stochastic Gradient Descent Optimizasyonu")
plt.show()

print(f"SGD ile optimize edilmiş intercept: {intercept}")
print(f"SGD ile optimize edilmiş eğim: {slope}")