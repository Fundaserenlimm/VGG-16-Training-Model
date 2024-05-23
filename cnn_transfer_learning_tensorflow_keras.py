import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Önceden eğitilmiş VGG16 modelini yükleme.
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# VGG16 modelinin katmanlarını gösterme.
conv_base.summary()

# Hangi katmanların eğitileceğine karar verme.
# 'block5_conv1' katmanına kadar olan katmanlar dondurulacak.
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Boş bir model oluşturma.
model = Sequential()

# VGG16 modelini evrişimli katman olarak ekleme.
model.add(conv_base)

# Evrişimli katmanların çıkışını düzleterek vektöre dönüştürme.
model.add(Flatten())

# Tam bağlantılı (fully connected) bir gizli katman eklenmesi.
model.add(Dense(256, activation='relu'))

# Çıkış katmanı eklenmesi (2 sınıf için softmax aktivasyonu).
model.add(Dense(2, activation='softmax'))

# Modeli derleme (optimizasyon yöntemi, kayıp fonksiyonu ve metriklerin belirlenmesi).
optimizer = RMSprop(learning_rate=1e-5)  # Optimizer nesnesini doğru şekilde oluştur
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,  # Optimizer nesnesini optimizer parametresine ver
              metrics=['accuracy'])


# Oluşturulan modelin özetini görüntüleme.
model.summary()

# Verilerin bulunduğu dizinlerin tanımlanması.
train_path = 'veriseti/EGITIM'
valid_path = 'veriseti/GECERLEME'
test_path = 'veriseti/TEST'

# Aşırı uydurmayı önlemek için veri artırma yöntemlerini uygulama.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Eğitim veri seti yükleyiciyi tanımlama.
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'  # Çoklu sınıflandırma için kategori modu
)

# Doğrulama için veri artırma yöntemlerini uygulamaya gerek yok.
validation_datagen = ImageDataGenerator(rescale=1./255)

# Doğrulama veri seti yükleyiciyi tanımlama.
validation_generator = validation_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'  # Çoklu sınıflandırma için kategori modu
)

# Modelin eğitimi.
history = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=1
)

# Eğitilen modeli çalışma dizinine kaydetme.
model.save('trained_tf_model.h5')

# Eğitilen modeli test etmek için veri artırma yöntemlerine gerek yok.
test_generator = validation_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',  # Çoklu sınıflandırma için kategori modu
    shuffle=False  # Test verisini karıştırmaya gerek yok
)

# Test sonuçlarını yazdırma.
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print('Test doğruluğu:', test_acc)

