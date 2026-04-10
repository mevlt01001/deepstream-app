#ifndef AICLASS_HPP
#define AICLASS_HPP

#include <vector>
#include <atomic>

class AudioModel;
typedef struct DstObjectData DstObjectData;
struct DstObjectData;

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
        AI(std::string ds_config_path, std::string audio_model_path, int sample_rate, int max_sec);

        void run_deepstream();

        // Bounding box verileriyle işlemler
        void process_bboxes(DstObjectData* obj_list, int num_objects, int frame_num);

        // Ses kaydı başlatma fonksiyonu
        void start_recording();

        // Ses kaydı durdurma fonksiyonu
        void stop_recording();

        std::vector<float>* get_audio_data();
        bool get_is_recording();

        void get_class_info();

    private:
        void audio_inference(std::vector<bool>& targets);
        void class_info();
};
#endif // AICLASS_HPP