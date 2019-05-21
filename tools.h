#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <Windows.h>
#include <random>
#include "picosha2.h"
#include "reduction_cpu.h"
#include "markov3_t.h"

using std::string;
using std::vector;
using std::cout;
using std::cin;
using std::endl;

uint64_t getNum();

int dialog(vector<string> *msgs, string header);

vector<string> *FolderFiles(string folder);

string TimeStamp();

string random_random_string(uint8_t min_l, uint8_t max_l);

string next_random_string(uint8_t min_l, uint8_t max_l, string prev_str);

string random_human_string(uint8_t min_l, uint8_t max_l, markov3_t *mt, uint8_t humanity);

string next_human_string(uint8_t min_l, uint8_t max_l, markov3_t *mt, string prev_str);

uint8_t *hexSHAtoByte(string hex_string);

vector<string> *CreateUniqueArray(uint64_t amount, uint8_t min_l, uint8_t max_l, markov3_t *mt, bool rand, string prev_str, bool human, uint8_t humanity);

uint64_t WordCount(string filename);

uint64_t WordSizeCount(string filename, uint8_t size);

bool HasNoLetters(string str);

bool OnlyNumbers(string str);

bool IsRestricted(string str);

bool IsKeystroked(string str);