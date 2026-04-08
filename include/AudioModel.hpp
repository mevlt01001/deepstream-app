#ifndef AUDIO_MODEL_HPP
#define AUDIO_MODEL_HPP

#include <vector>
#include <torch/torch.h>
#include "dr_wav.h"
#include "miniaudio.h"

class AudioModel {
    private:
        // holds whether AudioModel recording? 
        bool is_recording;
        // recorded audio data
        std::vector<float> audio_buffer;    
        torch::jit::script::Module model;   
        ma_device device;                   
        ma_device_config deviceConfig;      
        std::mutex buffer_mutex;                                                
        std::string ds_config_file_path;              
        std::string audio_model_file_path;             
        int target_sr;                                      
        int max_seconds;                                     
        std::vector<bool> targets;          
        AudioModel* audio_model;                            
        std::atomic<bool> is_audio_busy{false};

        static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

    public:
        AudioModel(const std::string& model_path, int sample_rate = 16000, int max_sec = 8);
        ~AudioModel();

        void start_recording();
        void stop_recording();
        torch::Tensor toTorchTensor();
        torch::Tensor resample_audio(torch::Tensor audio_data, int orig_sr, int target_sr);
        torch::Tensor preprocess_audio(torch::Tensor audio_data, int max_seconds, int target_sr);
        torch::Tensor inference(torch::Tensor final_audio);

        std::vector<bool> get_targets() const;
        bool get_is_recording() const;
        std::vector<float>* get_audio_buffer();


};
#endif  // AUDIO_MODEL_HPP