#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <tuple>
#include <cstdlib> // std::stoi için

// 1. RESAMPLE (YENİDEN ÖRNEKLEME) FONKSİYONU
torch::Tensor resample_audio(torch::Tensor audio_data, int orig_sr, int target_sr) {
    if (orig_sr == target_sr) {
        std::cout << "-> Orijinal SR (" << orig_sr << ") ile Target SR ayni. Resample atlandi." << std::endl;
        return audio_data;
    }

    std::cout << "-> Resampling basladi: " << orig_sr << " Hz -> " << target_sr << " Hz" << std::endl;

    // Orijinal uzunluk (Zaman ekseni)
    int64_t orig_length = audio_data.size(1);
    
    // Hedef uzunluğu oran orantı ile buluyoruz
    int64_t target_length = static_cast<int64_t>(orig_length * (static_cast<double>(target_sr) / orig_sr));

    // PyTorch interpolate fonksiyonu [Batch, Channel, Time] (3D) tensör bekler.
    // Bizim tensörümüz [Channel, Time] (2D). Başına Batch boyutunu (0. index) ekliyoruz (unsqueeze).
    audio_data = audio_data.unsqueeze(0);

    // Lineer İnterpolasyon (Resample) ayarları
    namespace F = torch::nn::functional;
    auto options = F::InterpolateFuncOptions()
                       .size(std::vector<int64_t>{target_length})
                       .mode(torch::kLinear) // Ses verisi için 1 Boyutlu Lineer İnterpolasyon
                       .align_corners(false);

    audio_data = F::interpolate(audio_data, options);

    // İşlem bitince eklediğimiz Batch boyutunu siliyoruz (squeeze), tekrar [Channel, Time] oluyor.
    audio_data = audio_data.squeeze(0);

    return audio_data;
}

// 2. KIRPMA VE DOLDURMA (CROP & PAD) FONKSİYONU
std::tuple<torch::Tensor, int64_t> adjust_audio_length(torch::Tensor audio_data, int max_seconds, int target_sr) {
    int64_t max_frames = static_cast<int64_t>(max_seconds) * target_sr;
    int64_t current_frames = audio_data.size(1); 

    if (current_frames > max_frames) {
        audio_data = audio_data.slice(1, 0, max_frames);
        std::cout << "-> Ses uzun, " << max_seconds << " saniyeye kirpildi." << std::endl;
    } 
    else if (current_frames < max_frames) {
        int64_t padding = max_frames - current_frames;
        auto pad_options = torch::nn::functional::PadFuncOptions({0, padding})
                               .mode(torch::kConstant)
                               .value(0.0);
        
        audio_data = torch::nn::functional::pad(audio_data, pad_options);
        std::cout << "-> Ses kisa, sonuna " << padding << " frame (0.0) eklendi." << std::endl;
    } else {
        std::cout << "-> Ses tam " << max_seconds << " saniye, islem yapilmadi." << std::endl;
    }

    int64_t final_length = audio_data.size(1);
    return std::make_tuple(audio_data, final_length);
}

// 3. MAIN FONKSİYONU (Dışarıdan Argüman Alma)
int main(int argc, char* argv[]) {
    // Argüman kontrolü
    if (argc != 4) {
        std::cerr << "\n[HATA] Eksik Arguman Girdiniz!" << std::endl;
        std::cerr << "Kullanim: ./audio_reader <dosya_yolu> <target_sr> <max_seconds>" << std::endl;
        std::cerr << "Ornek   : ./audio_reader test_sesi.wav 16000 8\n" << std::endl;
        return -1;
    }

    // Konsoldan girilen argümanları değişkenlere atıyoruz
    std::string file_path = argv[1];
    int target_sr = std::stoi(argv[2]);   // String'i Integer'a çevir
    int max_seconds = std::stoi(argv[3]); // String'i Integer'a çevir

    std::cout << "\n=== ISLEM BASLIYOR ===" << std::endl;
    std::cout << "Dosya      : " << file_path << std::endl;
    std::cout << "Target SR  : " << target_sr << " Hz" << std::endl;
    std::cout << "Max Sure   : " << max_seconds << " Saniye" << std::endl;

    // --- SESİ OKUMA (dr_wav) ---
    unsigned int channels;
    unsigned int orig_sr; // Dosyanın orijinal sample rate'i
    drwav_uint64 totalPCMFrameCount;

    float* pSampleData = drwav_open_file_and_read_pcm_frames_f32(
        file_path.c_str(), &channels, &orig_sr, &totalPCMFrameCount, NULL);

    if (pSampleData == NULL) {
        std::cerr << "[HATA] WAV dosyasi acilamadi!" << std::endl;
        return -1;
    }

    int64_t total_samples = totalPCMFrameCount * channels;
    torch::Tensor data = torch::from_blob(pSampleData, {channels, total_samples}, torch::kFloat32).clone();
    drwav_free(pSampleData, NULL);

    // --- ADIM 1: YENİDEN ÖRNEKLEME (RESAMPLE) ---
    data = resample_audio(data, orig_sr, target_sr);

    // --- ADIM 2: UZUNLUK AYARLAMA (CROP/PAD) ---
    auto result = adjust_audio_length(data, max_seconds, target_sr);
    torch::Tensor final_audio = std::get<0>(result);
    int64_t final_length = std::get<1>(result);

    // --- SONUÇLAR ---
    std::cout << "\n=== ISLEM TAMAMLANDI ===" << std::endl;
    std::cout << "Final Tensor Boyutu : " << final_audio.sizes() << std::endl;
    std::cout << "Final Sample Rate   : " << target_sr << " Hz" << std::endl;
    std::cout << "Gecerli Uzunluk     : " << final_length << " frames" << std::endl;
    std::cout << "Min Deger           : " << final_audio.min().item<float>() << std::endl;
    std::cout << "Max Deger           : " << final_audio.max().item<float>() << std::endl;
    std::cout << "========================\n" << std::endl;

    return 0;
}