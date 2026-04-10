#ifndef AUDIO_MODEL_HPP
#define AUDIO_MODEL_HPP

#include <vector>
#include <torch/torch.h>
#include "dr_wav.h"
#include "miniaudio.h"

//TODO: DOCs each entity

/*
AudioModel is a thread-safe class works to access mic, to record audio data and provide inference funcitons it on `torch::jit`.
Example:
``` 
    AudioModel* audio_model = new AudioModel("path/to/model.pt", 16000, 8);
    audio_model->start_recording(); // Starts the voice recording.
    audio_model->stop_recording();  // Stops the voice recording.

    // Access to the recorded data
    std::vector<float>* audio_data = audio_model->get_audio_buffer();

    // Access info if voice recording
    bool audio_recording = audio_model->get_is_recording();

    // Convert to torch tensor from vector
    torch::Tensor audio_tensor = audio_model->toTorchTensor();

    // Inference on torch::jit
    torch::Tensor result = audio_model->inference(audio_tensor);

    // Access to targets 
    std::vector<bool>& targets = audio_model->get_targets();
```
*/
class AudioModel {
    private:
        // Holds a boolean if the model recording.
        bool is_recording;
        // Holds recorded data
        std::vector<float> audio_buffer;
        // Pretrained Audio recognition model
        torch::jit::script::Module model;
        ma_device device;       
        ma_device_config deviceConfig;      
        // Mutex for audio buffer while data saving
        std::mutex buffer_mutex;
        // Deepstream configuration file path
        std::string ds_config_file_path;     
        // Audio recognition model pack file path
        std::string audio_model_file_path;
        // Model trained sample rate      
        int target_sr;
        // Madel trained maximum audio lenght
        int max_seconds;
        // Recognized Targest boolean val for Red, Green, Blue respectively
        std::vector<bool> targets;

        // Microphone data record funciton in each frame (sound package)
        static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

    public:
        /*
        Class has just one constructor. 

        `model_path` is audio model absolute path. Used for `torch::jit::load(model_path)`
        `sample_rate` is microphone sound access count per time. It effect sound quality directly. And model must have trained dedicated sample rate. Default is 16000 (16Khz)
        `max_sec` is maximum second of model can process. if recorded data longer than `max_sec` then it will cropped till 'max_sec', but if recorded data shorter than `max_sec` then it will be padded with zeros till data riched as data as much amound of 'max_sec'. This default is 8 (8 seconds).
        */
        AudioModel(const std::string& model_path, int sample_rate = 16000, int max_sec = 8);
        // miniaudio need to be uninitialize to relase the mic `ma_device_uninit()`
        ~AudioModel();
        /*
        Firstly cleans to the `audio_buffer` using `buffer_mutex` and then starts to read microphone.
        */
        void start_recording();
        // Stops recording process.
        void stop_recording();
        // Convert to torch tensor from vector data. If `audio_buffer` is empty, it returns an empty torch tensor.
        torch::Tensor toTorchTensor();
        // Data preprocessor according to the 'max_sec'
        torch::Tensor preprocess_audio(torch::Tensor audio_data, int max_seconds, int target_sr);
        // Takes input data (`recorded_data`), runs model, attach `targets` returns out data. Returned data is empty when `audio_buffer` is empty or inference had met an error. In the other hand, returns raw color data which model recognized.
        torch::Tensor inference(torch::Tensor recorded_data);
        // Returns attached `targets`
        std::vector<bool> get_targets() const;
        // Returns `is_recordings` to figure out data whether recorfing.
        bool get_is_recording() const;
        // Returns 'audio_buffer', this is thread-safe.
        std::vector<float>* get_audio_buffer();
};
#endif  // AUDIO_MODEL_HPP