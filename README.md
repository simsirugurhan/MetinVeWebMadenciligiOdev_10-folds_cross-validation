# MetinVeWebMadenciligiOdev_10-folds_cross-validation
 Metin ve Web MAdenciliği ödevi, 10-folds cross-validation, tf-idf, Multinomial Naive Bayes,

Veri kümesinin içeriği:
“Raw_texts” klasörü ve bu klasör altındaki 3 farklı
klasörde, 3 farklı sınıfına ait toplamda 3000 ayrı tweet kaydı bulunmaktadır. Bu kayıtlar, Internet’ten
alınmıştır ve çeşitli kişilere ait çeşitli farklı konularda gerçek kayıtlardır. Tweet’ler Türkçe yazılmıştır ama
yabancı kökenli sözcükler, ayrıca Türkçe dil kurallarında tanımlı olmayan kısaltmalar, vb. içerebilir.
Metin sınıflandırması, 3 ayrı sınıf şeklinde yapılacaktır ve ilgili tweet sınıfları aşağıdaki gibidir:
1- Olumlu Tweet ler (toplam 756 kayıt)
2- Olumsuz Tweet ler (toplam 1287 kayıt)
3- Nötr Tweet ler (toplam 957 kayıt)
Her sınıfa ait tweet metinleri, ayrı bir klasörde yer almaktadır.
Veri kümesindeki sınıf sayısı: 3
Veri kümesindeki toplam kayıt sayısı: 3000

Ödevde Yapılacaklar ve İstenenler:
* Bu veri kümesini kullanarak, multi-class classification ile bu 3 sınıf için belge sınıflandırması
yapılacaktır. Metinler, yani tweet kayıtları ham veridir.
* Bu veri üzerinde sözcüklerin ayrıştırılması (tokenization) gereklidir. Bu şekilde sözcüklerden
öznitelikler (features / attributes) oluşacaktır. Tokenization’da hangi karakterlerin ayırı olarak
kullanılacağı (boşluk, virgül, nokta, vb) öğrencilere bırakılmıştır.
* Metinlerdeki büyük harflerin küçük harfe çevrilmesi (lower-case), etkin olmayan sözcüklerin
(stop-words) kullanılması da önerilir. (Türkçe için stop-words dosyaları Moodle sisteminde önceki
haftalarda ilgili hafta kısmında yüklü bulunmaktadır, onu kullanabilirsiniz).
* Sözcüklerin köklerine göre gruplanması (stemming) ve ilgili stemmer araçları da kullanılabilir. Türkçe
için “Zemberek” uygulaması önerilmektedir.
* Özniteliklerin seçimi / azaltılması (feature selection / reduction) yöntemlerinin de kullanılması
özellikle önerilmektedir. Sizlere derste anlatılan ve örnekleri verilen yöntemlerden bir veya
birkaçını kullanabilirsiniz.
* Sözcüklerin metinlerde kaç kere geçtiğinin sayısal ve vektörel temsilinde tf-idf ‘leri hesaplayıp
kullanacaksınız.
* Sınıflandırma için hangi algoritmayı (k-NN yani k en yakın komşu veya Multinomial Naive Bayes) ve
k-NN kullanılacaksa da hangi uzaklık ya da benzerlik metrikleri (Cosine, Pearson, Euclidean, vb)
gene öğrencilerin tercihine bırakılmıştır.
* Programınızda ilgili aşamada oluşturacağınız metin-sözcük verisini (3000 metin instance yani
kayıtlardan, son hale getirdiğiniz sözcüklerin de attribute olduğu ve seçtiğiniz yönteme göre her
sözcüğün ilgili kayıttaki sayısal temsili değeri (tf-idf) bulunan veriyi) ödev tesliminde ayrıca bir .txt
dosya olarak (csv yani virgülle ayrılmış şekilde) teslim etmeniz zorunludur.

Programınızda eğitim ve test aşaması için stratified 10-folds cross-validation yöntemi
kullanılacaktır.
* Stratified 10-folds cross-validation sonucunda elde performans ölçüm değerlerini de ayrı bir dosyada teslim etmeniz zorunludur.
