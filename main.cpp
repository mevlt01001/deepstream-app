#include <iostream>
#include <thread>
#include "AIClass.hpp"

void check_mic(AI* ai_class);

int main(int argc, char* argv[]) {
    AI* ai = new AI(
        "/home/orin2/Downloads/customDeepstreamSample/deepstream_app.txt", 
        "/home/orin2/workspace/deepstream-app/audio_model.pt",           
        16000,                      
        8                           
    ); 

    ai->run_deepstream();
    
    bool record_flag = false;
    bool flag_changed = false;
    std::cout << "SES KAYDI KONTROLU:" << std::endl;
    std::cout << "Kaydi BASLATMAK icin [ENTER] tusuna basin." << std::endl;
    std::cout << "Kaydi DURDURMAK icin tekrar [ENTER] tusuna basin." << std::endl;

    while(true) {
        std::cin.get(); 
        record_flag = !record_flag; 

        if (record_flag == true) {
            ai->start_recording();
            ai->get_class_info();

        } else {
            ai->stop_recording();
            ai->get_class_info();
        }
    }
    
    delete ai;
    return 0;
}

void voice_func(bool& record_flag, bool& flag_changed, AI* ai) {
    std::thread([record_flag, flag_changed, ai](){
        if (flag_changed) {
            if (record_flag == true) {
                ai->start_recording();
            } else {
                ai->stop_recording();
            }
        } else {
            return;
        }
    }).detach();
}