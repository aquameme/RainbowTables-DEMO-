#pragma once

#include <vector>
#include "markov3_t.h"
#include "tools.h"
#include "picosha2.h"
#include "reduction_cpu.h"

using std::vector;
using std::string;

const string def_charset = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

class rainbow_t
{
private:
	string filename;			//имя файла/папки для этой радужной таблицы
	string charset;				//строка алфавита
	uint8_t min_l;				//минимальная длина открытого текста; на малых данных по умолчанию совпадает с максимальной
	uint8_t max_l;				//максимальная длина открытого текста
	uint32_t chain_l;			//длина цепочки одной таблицы
	uint64_t chain_am;			//количество цепочек одной таблицы
	uint16_t part_am;			//если таблица большая - разделить на части по 256 МБ
	uint8_t part;				//номер части
	uint64_t chain_am_part;		//кол-во строк в данной части
	uint64_t temp_size;
	vector<float> *percs;		//вектор из двух элементов: 1. вероятность успешного поиска, 2. среднее время поиска в мс
	uint64_t gen_dur;			//время генерации в мс
	vector<string> *starts;		//вектор начальных строк
	vector<string> *ends;		//вектор конечных строк
	markov3_t *mt;				//указатель на объект марковской таблицы
	uint8_t humanity;			//показатель случайности строк

	void CheckSorted();
	void SortRTByEnd(uint64_t left, uint64_t right);				//метод сортировки таблицы по концам цепочек (для бинарного поиска)
	bool GenUniqueChain(uint8_t limit, bool human);								//метод генерации уникальной цепочки
	string ProbeForHash(string hash);									//метод поиска конкретного хеша в таблице
	void FindSearchPercPlusHashes(vector<string> *hashes);			//метод определения вероятности успешного поиска для таблицы

public:
	rainbow_t();
	rainbow_t(uint64_t chain_am, uint32_t chain_l, string charset, uint8_t min_length, uint8_t max_length, markov3_t *mt, uint8_t humanity);
	rainbow_t(string rt_file);
	~rainbow_t();

	string getFilename();
	string getMTFilename();
	string getCharset();
	uint8_t getMin_l();
	uint8_t getMax_l();
	uint32_t getChain_l();
	uint64_t getChain_am();
	uint16_t getPart_am();
	uint8_t getPart();
	uint64_t getChain_am_part();
	uint64_t getTemp_size();
	string getStart(uint64_t counter);
	string getEnd(uint64_t counter);
	float getProb();
	float getSearchTime();
	uint64_t getGen_dur();
	markov3_t *getMTptr();
	uint8_t getHumanity();

	void setFilename(string filename);
	void setGenDur(uint64_t gen_dur);
	void pushBackEnd(string str);

	/*CPU*/

	void GenerateRT(bool human);											//метод обычной генерации с повторениями
	void GenerateUniqueRT(bool human);									//метод генерации таблицы с уникальными концами - perfect RT
	void RTProbability(uint16_t amount);
	void WriteRT_Info();
	void WriteRT_Part();
	void Read_Part(uint8_t part_num);

	/*GPU*/

	void GenerateRT_CUDA(int blocksNum, int threadsNum, bool human);
	void GenerateUniqueRT_CUDA(int blocksNum, int threadsNum, bool human);
	void GenerateRTbyTime_CUDA(int blocksNum, int threadsNum, bool human, uint64_t time);
	void GenerateUniqueRTbyTime_CUDA(int blocksNum, int threadsNum, bool human, uint64_t time);
	void SearchProb_CUDA(uint16_t amount, int blocksNum, int threadsNum);
	void SearchProbExpress_CUDA(uint16_t amount, int blocksNum, int threadsNum);
};