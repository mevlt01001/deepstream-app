#include <iostream>
#include "deepstream_app.h"

class AI {
    private:
        char* deepstream_configuration_file;
    public:
        std::vector<char*> parser(char* command) {
            std::vector<char*> tokens;
            char* token = std::strtok(command, " ");

            while (token != nullptr) {
                tokens.push_back(token);
                token = std::strtok(nullptr, " ");
            }

            return tokens;
        }

}