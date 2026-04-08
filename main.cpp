#include <iostream>
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
    std::cout << "SES KAYDI KONTROLU:" << std::endl;
    std::cout << "Kaydi BASLATMAK icin [ENTER] tusuna basin." << std::endl;
    std::cout << "Kaydi DURDURMAK icin tekrar [ENTER] tusuna basin." << std::endl;

    while(true) {
        std::cin.get(); 
        record_flag = !record_flag; 

        if (record_flag == true) {
            ai->start_recording();
        } else {
            ai->stop_recording();
        }
    }
    
    delete ai;
    return 0;
}