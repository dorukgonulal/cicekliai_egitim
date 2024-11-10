import numpy as np

# Örnek 1: Basit Zincir Kuralı (Ağırlık -> Boy -> Ayakkabı Numarası)
# Fonksiyonları ve türevlerini tanımla
def height_from_weight(weight):
    # Doğrusal bir ilişki varsayalım: boy = 2 * ağırlık
    return 2 * weight

def shoe_size_from_height(height):
    # Doğrusal bir ilişki varsayalım: ayakkabı_numarası = 0.25 * boy
    return 0.25 * height

# Türevler
def d_height_d_weight():
    return 2  # Boyun ağırlığa göre türevi

def d_shoe_size_d_height():
    return 0.25  # Ayakkabı numarasının boya göre türevi

# Zincir kuralı: ayakkabı numarasının ağırlığa göre türevi
def d_shoe_size_d_weight():
    return d_shoe_size_d_height() * d_height_d_weight()

# Ağırlıktaki bir birimlik değişiklik için ayakkabı numarasındaki değişimi hesapla
change_in_shoe_size = d_shoe_size_d_weight()
print(f"Ağırlık birim arttığında ayakkabı numarasındaki değişim: {change_in_shoe_size}")

# Örnek 2: Karmaşık Zincir Kuralı (Zaman -> Açlık -> Dondurma İsteği)
# Açlığın zamana bağlı üstel bir fonksiyon olarak tanımı
def hunger_from_time(time):
    return np.exp(time)  # Açlık zamana bağlı olarak üssel artar

# Dondurma isteği açlığın karekök fonksiyonu olarak tanımlı
def craving_from_hunger(hunger):
    return np.sqrt(hunger)  # Azalan kazançlar karekök ile modellenir

# Türevler
def d_hunger_d_time(time):
    return np.exp(time)  # Üssel fonksiyonun türevi

def d_craving_d_hunger(hunger):
    return 0.5 / np.sqrt(hunger)  # Karekök fonksiyonunun türevi

# Zincir kuralı: isteğin zamana göre türevi
def d_craving_d_time(time):
    hunger = hunger_from_time(time)
    return d_craving_d_hunger(hunger) * d_hunger_d_time(time)

# Son atıştırmadan itibaren belirli bir süre için istekteki değişimi hesapla
time_since_snack = 1.0  # Örnek zaman (saat olarak)
change_in_craving = d_craving_d_time(time_since_snack)
print(f"Son atıştırmadan itibaren bir saat geçince dondurma isteğindeki değişim: {change_in_craving:.3f}")
