import numpy as np

# SoftPlus aktivasyon fonksiyonu
def softplus(x):
    return np.log(1 + np.exp(x))

# Örnek giriş verisi: Dozaj değerleri (düşük, orta, yüksek)
dosage_values = np.array([1.0, 2.5, 4.0])

# Başlangıç ağırlıkları ve bias değerleri
weights_input_hidden = np.array([0.5, -0.5])  # İki gizli düğüm için ağırlıklar
biases_hidden = np.array([0.0, 0.0])  # Gizli düğümler için bias değerleri
weights_hidden_output = np.array([1.0, 1.0])  # Gizli düğümden çıktı düğümüne ağırlıklar
bias_output = 0.0  # Çıktı düğümü için bias değeri

# Ağ üzerinden ileri geçiş işlemi
def simple_neural_network(input_value):
    # Adım 1: Gizli düğümlerdeki değerleri hesapla
    hidden_input_1 = weights_input_hidden[0] * input_value + biases_hidden[0]
    hidden_input_2 = weights_input_hidden[1] * input_value + biases_hidden[1]

    # Gizli düğümlere SoftPlus aktivasyon fonksiyonunu uygula
    hidden_output_1 = softplus(hidden_input_1)
    hidden_output_2 = softplus(hidden_input_2)

    # Adım 2: Nihai çıktıyı hesapla
    output = (weights_hidden_output[0] * hidden_output_1 +
              weights_hidden_output[1] * hidden_output_2 +
              bias_output)
    
    return output

# Her dozaj değeri için sinir ağını çalıştır
for dosage in dosage_values:
    output = simple_neural_network(dosage)
    print(f"Giriş: {dosage}, Çıkış: {output}")