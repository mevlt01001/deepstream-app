#include <iostream>
#include <vector>
#include <cstring>
#include "AIClass.hpp"

extern "C" {
    AI* global_ai_instance = nullptr; 
    void my_external_bbox_callback(DstObjectData* obj_list, int num_objects, int frame_num) {
        if (global_ai_instance != nullptr) {
            global_ai_instance->process_bboxes(obj_list, num_objects, frame_num);
        }
    }
}

int main(int argc, char* argv[]) {
    AI* ai = new AI(
        "/home/orin2/Downloads/customDeepstreamSample/deepstream_app.txt", 
        "audio_model.pt",           
        16000,                      
        8                           
    ); 
    global_ai_instance = ai; 
    
    set_external_bbox_callback(my_external_bbox_callback);
    ai->run_deepstream();
    
    delete ai;
    return 0;
}