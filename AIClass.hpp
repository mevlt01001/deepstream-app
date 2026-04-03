#ifndef AICLASS_HPP
#define AICLASS_HPP

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <torch/torch.h>
#include <torch/script.h> 
#include <string>
#include <tuple>
#include <vector>
#include <mutex>
#include <iostream>
#include <cstdlib> 
#include <memory> 
#include <atomic> // Thread çakışmalarını önlemek için
#include <thread> // Arka planda işlem yapmak için

extern "C" {
    #include "deepstream_app.h"
}

class AudioModel {
    private:
        bool is_recording = false;          
        torch::jit::script::Module model;   
        ma_device device;                   
        ma_device_config deviceConfig;      
        std::vector<float> audio_buffer;    
        std::mutex buffer_mutex;                                                
        std::string ds_config_file_path;              
        std::string audio_model_file_path;             
        int target_sr;                                      
        int max_seconds;                                     

        
        std::vector<bool> current_targets = std::vector<bool>(3, false); 
        AudioModel* audio_model = nullptr;                            
        std::atomic<bool> is_audio_busy{false};
        std::vector<bool> targets;          

        static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
            AudioModel* audioModel = static_cast<AudioModel*>(pDevice->pUserData);
            
            if (audioModel->is_recording) {
                const float* pInputF32 = static_cast<const float*>(pInput);
                std::lock_guard<std::mutex> lock(audioModel->buffer_mutex);
                audioModel->audio_buffer.insert(
                    audioModel->audio_buffer.end(), 
                    pInputF32, 
                    pInputF32 + frameCount
                );
            }
        }

    public:
        AudioModel(const std::string& model_path, int sample_rate = 16000, int max_sec = 8) 
            : target_sr(sample_rate), max_seconds(max_sec) {
            
            std::cout << "[AudioModel::Constructor] Initializing AudioModel with Target SR: " << target_sr << ", Max Sec: " << max_seconds << std::endl;
            
            try {
                model = torch::jit::load(model_path);
                model.eval();
                std::cout << "[AudioModel::Constructor] PyTorch JIT Model loaded successfully." << std::endl;
            } catch (const c10::Error& e) {
                std::cerr << "[AudioModel::Constructor] ERROR: Failed to load model! " << e.msg() << std::endl;
                exit(1);
            }

            deviceConfig = ma_device_config_init(ma_device_type_capture);
            deviceConfig.capture.format   = ma_format_f32; 
            deviceConfig.capture.channels = 1;             
            deviceConfig.sampleRate       = target_sr;     
            deviceConfig.dataCallback     = data_callback; 
            deviceConfig.pUserData        = this;          

            if (ma_device_init(NULL, &deviceConfig, &device) != MA_SUCCESS) {
                std::cerr << "[AudioModel::Constructor] ERROR: Failed to initialize microphone!" << std::endl;
                exit(1);
            }
            std::cout << "[AudioModel::Constructor] Microphone initialized successfully." << std::endl;
        }

        ~AudioModel() {
            std::cout << "[AudioModel::Destructor] Uninitializing microphone device and destroying model." << std::endl;
            ma_device_uninit(&device);
        }

        void start_recording() {
            std::cout << "[AudioModel::start_recording] Starting audio capture..." << std::endl;
            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                audio_buffer.clear();
            }
            is_recording = true;
            ma_device_start(&device); 
        }

        void stop_recording() {
            std::cout << "[AudioModel::stop_recording] Stopping audio capture..." << std::endl;
            is_recording = false;
            ma_device_stop(&device); 
        }

        torch::Tensor toTorchTensor() {
            std::cout << "[AudioModel::toTorchTensor] Converting audio buffer to Torch Tensor..." << std::endl;
            torch::Tensor audio_data;
            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                if (audio_buffer.empty()) {
                    std::cerr << "[AudioModel::toTorchTensor] WARNING: Audio buffer is empty. Returning empty tensor!" << std::endl;
                    return torch::empty({0}); 
                }
                audio_data = torch::from_blob(
                    audio_buffer.data(), 
                    {1, static_cast<int64_t>(audio_buffer.size())}, torch::kFloat32
                ).clone();
            }
            return preprocess_audio(audio_data, max_seconds, target_sr);
        }

        torch::Tensor inference(torch::Tensor final_audio) {
            std::cout << "[AudioModel::inference] Preparing to run model inference..." << std::endl;

            if (final_audio.numel() == 0) {
                std::cerr << "[AudioModel::inference] ERROR: Received empty tensor for inference!" << std::endl;
                return torch::empty({0});
            }

            try {
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(final_audio);
                
                std::cout << "[AudioModel::inference] Forward pass started..." << std::endl;
                torch::Tensor output = model.forward(inputs).toTensor();
                std::cout << "[AudioModel::inference] Forward pass completed successfully." << std::endl;
                
                torch::Tensor flat_out = output.view({-1});
                
                targets.clear();
                for (int i = 0; i < flat_out.size(0); ++i) {
                    float val = flat_out[i].item<float>();
                    targets.push_back(val > 0.55f); 
                }
                
                return output;
            } catch (const c10::Error& e) {
                std::cerr << "[AudioModel::inference] ERROR: Inference failed! " << e.msg() << std::endl;
                return torch::empty({0});
            }
        }
        
        std::vector<bool> get_targets() const {
            return targets;
        }

        torch::Tensor resample_audio(torch::Tensor audio_data, int orig_sr, int target_sr) {
            if (orig_sr == target_sr) return audio_data;
            int64_t orig_length = audio_data.size(1);
            int64_t target_length = orig_length * static_cast<int64_t>(static_cast<double>(target_sr) / orig_sr);
            audio_data = audio_data.unsqueeze(0);
            
            namespace F = torch::nn::functional;
            auto options = F::InterpolateFuncOptions().size(std::vector<int64_t>{target_length}).mode(torch::kLinear).align_corners(false);
            audio_data = F::interpolate(audio_data, options);
            audio_data = audio_data.squeeze(0);
            
            return audio_data;
        }

        torch::Tensor preprocess_audio(torch::Tensor audio_data, int max_seconds, int target_sr) {
            int64_t max_frames = static_cast<int64_t>(max_seconds) * target_sr;
            int64_t current_frames = audio_data.size(1); 
            
            if (current_frames > max_frames) {
                audio_data = audio_data.slice(1, 0, max_frames);
            } 
            else if (current_frames < max_frames) {
                int64_t padding = max_frames - current_frames;
                auto pad_options = torch::nn::functional::PadFuncOptions({0, padding}).mode(torch::kConstant).value(0.0);
                audio_data = torch::nn::functional::pad(audio_data, pad_options);
            }
            return audio_data;
        }
};


class AI {
    private:
        std::string ds_config_file_path;              
        std::string audio_model_file_path;             
        int target_sr;                                      
        int max_seconds;                                     
        
        std::vector<bool> current_targets = std::vector<bool>(3, false); 
        AudioModel* audio_model = nullptr;                            

        std::atomic<bool> is_audio_busy{false}; 

    public:
        AI(std::string ds_config_path, std::string audio_model_path, int sample_rate, int max_sec) :
        ds_config_file_path(ds_config_path), audio_model_file_path(audio_model_path), target_sr(sample_rate), max_seconds(max_sec)
        {
            this->audio_model = new AudioModel(audio_model_file_path, target_sr, max_seconds);
        }

        void run_deepstream() {
            std::vector<std::string> args = { "my_ai_app", "-c", ds_config_file_path };
            std::vector<char*> fake_argv;
            for (const auto& arg : args) { fake_argv.push_back(strdup(arg.c_str())); }
            fake_argv.push_back(nullptr);

            std::cout << "\n[AI] DeepStream Pipeline baslatiliyor..." << std::endl;
            deepstream_app_main(fake_argv.size() - 1, fake_argv.data());

            for (size_t i = 0; i < fake_argv.size() - 1; ++i) { free(fake_argv[i]); }
        }

        void process_bboxes(DstObjectData* obj_list, int num_objects, int frame_num) {
            // her frame'de burası tetiklenir

            if (!is_audio_busy.load()) { // if audio is not busy
                
                is_audio_busy.store(true);
                std::thread([this]() {
                    this->start_recording_for_target(); 
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                    this->stop_recording_and_infer();
                    this->is_audio_busy.store(false); 
                }).detach(); // detach() komutu bu işlemi ana programdan bağımsız kılar.
            }

        }

        void start_recording_for_target() {
            if (!audio_model) return;
            audio_model->start_recording();
        }

        void stop_recording_and_infer() {
            if (!audio_model) return;

            audio_model->stop_recording();
            torch::Tensor audio_tensor = audio_model->toTorchTensor();
            torch::Tensor result = audio_model->inference(audio_tensor);
            
            std::vector<bool> targets = audio_model->get_targets();

            std::cout << "\n=============================================" << std::endl;
            std::cout << "[AI] TAHMIN SONUCU:" << std::endl;
            if (targets.size() >= 3) {
                std::cout << "RED   : " << (targets[0] ? "TRUE" : "FALSE") << std::endl;
                std::cout << "GREEN : " << (targets[1] ? "TRUE" : "FALSE") << std::endl;
                std::cout << "BLUE  : " << (targets[2] ? "TRUE" : "FALSE") << std::endl;
            }
            std::cout << "=============================================\n" << std::endl;
        }
};
#endif // AICLASS_HPP