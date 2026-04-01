#include <iostream>
#include <vector>
#include <cstring>
// #include <torch/script.h> // LibTorch için
// #include <onnxruntime_cxx_api.h> // ONNX Runtime için
#include "AIClass.hpp"

extern "C" {
    AI* global_ai_instance = nullptr; // Global AI instance to be used in the callback
    void my_external_bbox_callback(DstObjectData* obj_list, int num_objects, int frame_num) {
        if (global_ai_instance != nullptr) {
            global_ai_instance->process_bboxes(obj_list, num_objects, frame_num);
        }
    }
}

AI::AI(char* argv[]) {
    this->ds_config_file_path = this->get_config_file(argv);
    std::cout << "AI class initialized with config file: " << ds_config_file_path << std::endl;
}

std::string AI::get_config_file(char* argv[]) {
    for (int i = 1; argv[i] != nullptr; i++) {
        if (std::string(argv[i]) == "-c" && argv[i + 1] != nullptr) {
            return std::string(argv[i + 1]);
        }
    }
    return ""; 
}

void AI::process_bboxes(DstObjectData* obj_list, int num_objects, int frame_num) {
    std::cout << "--- Frame " << frame_num << " | " << num_objects << " nesne algilandi ---" << std::endl;
    for (int i = 0; i < num_objects; i++) {
        std::cout << "Nesne " << i << ": Sinif=" << obj_list[i].class_id 
                  << ", Takip ID=" << obj_list[i].tracking_id
                  << ", Etiket=" << obj_list[i].label 
                  << " [x=" << obj_list[i].left << ", y=" << obj_list[i].top 
                  << ", w=" << obj_list[i].width << ", h=" << obj_list[i].height << "]" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    AI* ai = new AI(argv); exit(0); // Sadece AI sınıfını test etmek için main fonksiyonunu ekledim. Gerçek uygulamada deepstream_app_main çağrılacak.
    global_ai_instance = ai; // Set the global AI instance for the callback
    
    std::string config_file = global_ai_instance->get_config_file(argv);
    std::cout << "Found config file: " << config_file << std::endl;
    
    set_external_bbox_callback(my_external_bbox_callback);
    
    deepstream_app_main(argc, argv);
    
    return 0;
}