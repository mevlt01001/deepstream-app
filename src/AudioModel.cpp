#include <torch/torch.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include "dr_wav.h"
#include "miniaudio.h"
#include "AudioModel.hpp"

// bool AudioModel::is_recording = false;
// torch::jit::script::Module AudioModel::model;
// ma_device AudioModel::device;
// ma_device_config AudioModel::deviceConfig;
// std::vector<float> AudioModel::audio_buffer;
// std::mutex AudioModel::buffer_mutex;
// std::string AudioModel::ds_config_file_path;              
// std::string AudioModel::audio_model_file_path;             
// int AudioModel::target_sr;                                      
// int AudioModel::max_seconds;                                     
// std::vector<bool> AudioModel::current_targets; 
// AudioModel* AudioModel::audio_model;                            
// std::atomic<bool> AudioModel::is_audio_busy{AudioModel::false};
// std::vector<bool> AudioModel::targets;          

// The definitions above are wrong. Should use these definition in constructor.

AudioModel::AudioModel(const std::string& model_path, int sample_rate = 16000, int max_sec = 8) : 
    AudioModel::target_sr(sample_rate), AudioModel::max_seconds(max_sec) {
    
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

void AudioModel::data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
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

AudioModel::~AudioModel() {
    ma_device_uninit(&device);
}

void AudioModel::start_recording() {
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        audio_buffer.clear();
    }
    is_recording = true;
    ma_device_start(&device); 
    std::cout << "[AudioModel::start_recording] Recording started..." << std::endl;
}

void AudioModel::stop_recording() {
    is_recording = false;
    ma_device_stop(&device); 
    std::cout << "[AudioModel::stop_recording] Recording stopped." << std::endl;
}

torch::Tensor AudioModel::toTorchTensor() {
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

torch::Tensor AudioModel::inference(torch::Tensor final_audio) {
    if (final_audio.numel() == 0) {
        std::cerr << "[AudioModel::inference] ERROR: Received empty tensor for inference!" << std::endl;
        return torch::empty({0});
    }

    try {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(final_audio);
        torch::Tensor output = model.forward(inputs).toTensor();
        std::cout << "[AudioModel::inference] Audio inference completed successfully." << std::endl;
        torch::Tensor flat_out = output.view({-1});
        std::cout << "[AudioModel::inference] Raw model output: " << flat_out[0].item<float>() << ", " << flat_out[1].item<float>() << ", " << flat_out[2].item<float>() << std::endl;
        
        targets.clear();
        for (int i = 0; i < flat_out.size(0); ++i) {
            float val = flat_out[i].item<float>();
            targets.push_back(val > 0.5f); 
        }
        
        return output;
    } catch (const c10::Error& e) {
        std::cerr << "[AudioModel::inference] ERROR: Inference failed! " << e.msg() << std::endl;
        return torch::empty({0});
    }
}

std::vector<bool> AudioModel::get_targets() const {
    return targets;
}

torch::Tensor AudioModel::resample_audio(torch::Tensor audio_data, int orig_sr, int target_sr) {
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

torch::Tensor AudioModel::preprocess_audio(torch::Tensor audio_data, int max_seconds, int target_sr) {
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

