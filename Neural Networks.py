# Gerekli kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Veri setini yükleme
data = pd.read_excel('dozaj_etki_verileri.xlsx')
X = data['Dozaj'].values.reshape(-1, 1)  # Dozaj değerleri (giriş verisi)
y = data['Etki'].values.reshape(-1, 1)   # Etki değerleri (beklenen çıkış)

# Sinir ağının yapılandırma parametreleri
input_neurons = 1              # Giriş katmanında 1 nöron (sadece dozaj değeri)
hidden_neurons_layer1 = 20     # İlk gizli katmanda 20 nöron
hidden_neurons_layer2 = 15     # İkinci gizli katmanda 15 nöron
output_neurons = 1             # Çıkış katmanında 1 nöron (etki değeri)
learning_rate = 0.01           # Öğrenme hızı, ağırlıkları güncelleme oranı
epochs = 100000               # Eğitim döngüsü sayısı (kaç kez eğitim yapılacağı)

# Ağırlık ve bias (sabit değer) tanımları, rasgele değerlerle başlatma
np.random.seed(0)  # Sonuçların tekrarlanabilir olması için rastgelelik kontrolü
W1 = np.random.randn(input_neurons, hidden_neurons_layer1)  # Girişten ilk gizli katmana ağırlıklar
b1 = np.random.randn(1, hidden_neurons_layer1)              # İlk gizli katman için bias
W2 = np.random.randn(hidden_neurons_layer1, hidden_neurons_layer2)  # İlk gizli katmandan ikinci gizli katmana ağırlıklar
b2 = np.random.randn(1, hidden_neurons_layer2)                      # İkinci gizli katman için bias
W3 = np.random.randn(hidden_neurons_layer2, output_neurons)         # İkinci gizli katmandan çıkış katmanına ağırlıklar
b3 = np.random.randn(1, output_neurons)                             # Çıkış katmanı için bias

# Aktivasyon fonksiyonu (tanh), bir nöronun çıktısını doğrusal olmayan bir şekle dönüştürür
def tanh(x):
    return np.tanh(x)

# Tanh fonksiyonunun türevi, geri yayılımda kullanılır
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Eğitim süreci boyunca kaybı (mean squared error) takip etmek için liste
mse_history = []
for epoch in range(epochs):  # Belirlenen epoch sayısı kadar döngü (eğitim)
    # İleri yayılım işlemi (veri ağı ileri doğru geçer ve tahmin yapılır)
    
    # Giriş verisini ilk gizli katmana geçiriyoruz
    Z1 = X.dot(W1) + b1             # İlk gizli katmana gelen toplam ağırlıklı giriş
    A1 = tanh(Z1)                   # İlk gizli katmanda aktivasyon fonksiyonunu uygulama
    
    # İlk gizli katmandan ikinci gizli katmana geçiş
    Z2 = A1.dot(W2) + b2            # İkinci gizli katmana gelen toplam ağırlıklı giriş
    A2 = tanh(Z2)                   # İkinci gizli katmanda aktivasyon fonksiyonunu uygulama
    
    # İkinci gizli katmandan çıkış katmanına geçiş (nihai tahmin değeri)
    Z3 = A2.dot(W3) + b3
    y_pred = Z3  # Çıkış katmanı tahmini (aktivasyon fonksiyonu kullanılmıyor, doğrusal bir çıktı)

    # Hata hesaplama (beklenen ve tahmin edilen değerler arasındaki fark)
    loss = np.mean((y_pred - y) ** 2)  # Ortalama karesel hata (MSE)
    mse_history.append(loss)           # Her epoch'ta hata değerini kaydetme

    # Geri yayılım işlemi (hata geriye doğru yayılarak ağırlıklar güncellenir)
    
    # Çıkış katmanındaki hata türevi
    dL_dy_pred = 2 * (y_pred - y) / y.size

    # Çıkış katmanı ile ikinci gizli katman arasındaki ağırlıkların türevi
    dL_dZ3 = dL_dy_pred
    dL_dW3 = A2.T.dot(dL_dZ3)  # Ağırlıkların türevi (W3 için)
    dL_db3 = np.sum(dL_dZ3, axis=0, keepdims=True)  # Bias'ın türevi (b3 için)

    # İkinci gizli katmandaki hata türevi ve ağırlıklar
    dL_dA2 = dL_dZ3.dot(W3.T)
    dL_dZ2 = dL_dA2 * tanh_derivative(Z2)
    dL_dW2 = A1.T.dot(dL_dZ2)
    dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True)

    # İlk gizli katmandaki hata türevi ve ağırlıklar
    dL_dA1 = dL_dZ2.dot(W2.T)
    dL_dZ1 = dL_dA1 * tanh_derivative(Z1)
    dL_dW1 = X.T.dot(dL_dZ1)
    dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)

    # Ağırlıkları ve bias'ları güncelleme (öğrenme hızı ile türevleri çarpıp çıkarıyoruz)
    W3 -= learning_rate * dL_dW3
    b3 -= learning_rate * dL_db3
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1

 
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


plt.plot(mse_history)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Eğitim Süreci Kaybı (MSE)')
plt.show()

# Modelin öngördüğü tahmin eğrisini çizme
X_sorted = np.linspace(0, 1, 100).reshape(-1, 1)  # Dozaj için 0-1 aralığında yeni veriler

# Yeni dozaj verilerini ileri yayılım ile geçirerek tahmin yapma
Z1_sorted = X_sorted.dot(W1) + b1
A1_sorted = tanh(Z1_sorted)
Z2_sorted = A1_sorted.dot(W2) + b2
A2_sorted = tanh(Z2_sorted)
y_pred_sorted = A2_sorted.dot(W3) + b3  # Nihai tahmin


plt.figure(figsize=(8, 5))
plt.plot(X_sorted, y_pred_sorted, label="Squiggle", color='red')
plt.scatter(X, y, color='blue', label='Veriler')
plt.xlabel('Dozaj')
plt.ylabel('Etki')
plt.legend()
plt.show()
