#include <iostream>
#include <cstring>
#include <vector>


std::vector<char*> parser(char* command) {
    // command = "AIClass -c config.txt"
    std::vector<char*> tokens;
    char* token = std::strtok(command, " ");

    while (token != nullptr) {
        tokens.push_back(token);
        token = std::strtok(nullptr, " ");
    }

    return tokens;
}


int main(int argc, char* argv[]) {
    std::string command;
    std::cout << "Enter command: ";
    std::cin >> command;
    
    


    std::vector<char*> args = parser(command[0]);
    for (char* arg : args) {
        std::cout << "Found argument: " << arg << std::endl;
    }
    return 0;
}