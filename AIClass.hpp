#ifndef AICLASS_HPP
#define AICLASS_HPP

#include <string>
#define AUDIO_FILE_PATH "./file.wav"

extern "C" {
    #include "deepstream_app.h"
}

class Audio {
    private:
        bool is_recording = false;
        std::string current_file_path = nullptr;

    public:
        void start_recording(uint8_t target_id);
        void stop_recording();
        void save_audio(const char* file_path);
        void read_audio(const char* file_path);
};

class AI {
    private:
        std::string ds_config_file_path  = ""; // DeepStream config file path
        std::string audio_model_file_path = ""; // ONNX model for Audio processing (resnet18, with melspectrogram input)

        
        uint8_t* selected_target_id = nullptr;
    public:
        AI(char* argv[]);
        std::string get_config_file(char* argv[]);
        void process_bboxes(DstObjectData* obj_list, int num_objects, int frame_num);
        void select_target(uint8_t target_id);
        void record_audio(uint8_t target_id);
        void read_audio_data(const char* audio_file_path);
        void run_inference_on_audio();
};
#endif // AICLASS_HPP