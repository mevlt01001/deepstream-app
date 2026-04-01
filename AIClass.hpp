#ifndef AICLASS_HPP
#define AICLASS_HPP

#include <string>

extern "C" {
    #include "deepstream_app.h"
}

class AI {
    private:
        std::string ds_config_file_path;
    public:
        std::string get_config_file(char* argv[]);
        void process_bboxes(DstObjectData* obj_list, int num_objects, int frame_num);
};
#endif // AICLASS_HPP