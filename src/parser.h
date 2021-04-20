#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
using json = nlohmann::json;

class Net;

static inline void InitParser(json& j, std::string&& json_file) {
    std::ifstream i(json_file);
    i >> j;
}

void Load(json& j, Net* net, std::unordered_map<std::string, double>& args);

bool parseArch(json& j, Net* net);

bool parseDigits(json& j, std::unordered_map<std::string, double>& args);