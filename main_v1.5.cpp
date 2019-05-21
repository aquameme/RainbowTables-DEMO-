#include <time.h>
#include <random>

#include "rainbow_t.h"
#include <cuda_runtime.h>
#include "cuda_defines.h"
#include "kernel_generate.cuh"

string curDir = "C:\\Temp\\NIR\\RT3_CUDA_10-1_x64\\RT3_CUDA_10-1_x64\\";

const std::map<char, vector<char>> replaces = {
{'!', {'i', 'l'}},
{'$', {'s'}},
{'0', {'o'}},
{'1', {'i', 'l', 't'}},
{'3', {'e'}},
{'4', {'a', 'f'}},
{'5', {'s'}},
{'6', {'b', 's'}},
{'7', {'s', 't'}},
{'8', {'b'}},
{'9', {'g', 'q'}},
{'@', {'a'}}
};

const string rep_str = "!$013456789@";

markov3_t *ChooseMT()
{
	string pattern = "mtables\\mt3*.txt";
	vector<string> *mtables = FolderFiles(pattern);
	int opt = dialog(mtables, "Available multi mtables:");
	markov3_t *temp_mt = new markov3_t(mtables->at(opt));
	return temp_mt;
}

bool IsThatLangWord(markov3_t *mt, string str)
{
	/*
	order:
		make big letter small
		make replacement template
		check these letters
	*/

	string tmp_str = str;
	string rep_temp = str;
	int found;

	for (int i = 0; i < rep_temp.size(); i++)
	{
		found = (int)(rep_str.find(rep_temp.at(i), 0));
		if (found < 0)
			rep_temp.at(i) = '_';
	}


	for (int i = 0; i < tmp_str.size(); i++)
		if ((tmp_str.at(i) > 64) && (tmp_str.at(i) < 91))
			tmp_str.at(i) += 32;

	uint64_t mult = 0;
	uint8_t i, j, k, length;

	do
	{
		/* check this comb */

		for (i = 0; i < tmp_str.size(); i++)
			if ((tmp_str.at(i) > 96) && (tmp_str.at(i) < 123))
				break;

		length = 0;
		for (j = 0; (i < tmp_str.size()) && (j < 21); j++, i++)
		{
			if ((tmp_str.at(i) < 97) || (tmp_str.at(i) > 122))
				break;
			if (!j)
				mult += mt->stats->at(j)->at(0)->at(0)->at(tmp_str.at(i) - 97);
			if (j == 1)
				mult *= mt->stats->at(j)->at(0)->at(tmp_str.at(i - 1) - 97)->at(tmp_str.at(i) - 97);
			if (j > 1)
				mult *= mt->stats->at(j)->at(tmp_str.at(i - 2) - 97)->at(tmp_str.at(i - 1) - 97)->at(tmp_str.at(i) - 97);
			length++;
			if (!mult)
				break;
		}

		/* make new rep comb */

		if (!mult || (length < 5))
		{
			for (j = 0; j < rep_temp.size(); j++)
			{
				if ((rep_temp.at(j) == str.at(j)) && (str.at(j) != '_'))
				{
					rep_temp.at(j) = replaces.at(str.at(j)).at(0);
					for (k = 0; k < j; k++)
					{
						found = (int)(rep_str.find(str.at(k), 0));
						if (found >= 0)
							rep_temp.at(k) = str.at(k);
					}
					mult = 0;
					break;
				}
				if (rep_temp.at(j) != '_')
				{
					for (k = 0; k < replaces.at(str.at(j)).size(); k++)
						if (replaces.at(str.at(j)).at(k) == rep_temp.at(j))
							break;
					if (k < (replaces.at(str.at(j)).size() - 1))
					{
						rep_temp.at(j) = replaces.at(str.at(j)).at(k + 1);
						for (k = 0; k < j; k++)
						{
							found = (int)(rep_str.find(str.at(k), 0));
							if (found >= 0)
								rep_temp.at(k) = str.at(k);
						}
						mult = 0;
						break;
					}
				}
			}
			if (j == rep_temp.size())
				return false;
		}

	} while (!mult);

	return true;
}

int GPUInfo(rainbow_t **rt)
{
	/*DEVICE INFO*/

	checkCudaErrors(cudaSetDevice(0));

	cudaDeviceProp deviceProperties;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProperties, 0));

	cout << endl << "Device 0: " << deviceProperties.name << "\n"
		<< "Compute capability: " << deviceProperties.major << "." << deviceProperties.minor << "\n"
		<< "Total global memory: " << deviceProperties.totalGlobalMem / 1073741824 << "GB" << "\n"
		<< "Total constant memory: " << deviceProperties.totalConstMem / 1024 << "KB" << "\n"
		<< "Shared memory per multiprocessor: " << deviceProperties.sharedMemPerMultiprocessor / 1024 << "KB" << "\n"
		<< "Shared memory per block: " << deviceProperties.sharedMemPerBlock / 1024 << "KB" << "\n"
		<< "Registers per multiprocessor: " << deviceProperties.regsPerMultiprocessor << "\n"
		<< "Registers per block: " << deviceProperties.regsPerBlock << "\n"
		<< "Multiprocessor count: " << deviceProperties.multiProcessorCount << "\n"
		<< endl;

	return 1;
}

int MakeLangPassDict(rainbow_t **rt)
{
	/* READ LANG MTABLE */

	markov3_t *mt = ChooseMT();

	/* READ MT AND DICTS */

	ofstream file_o;
	file_o.open("dicts\\" + mt->filename.substr(10, 3) + "_pass_dict.txt", std::ios::app);

	string pattern = "dicts\\*.txt";
	vector<string> *dicts = FolderFiles(pattern);
	dicts->insert(dicts->begin(), "End choice");
	dicts->shrink_to_fit();

	vector<string> *add_dicts = new vector<string>();
	cout << "Choose dictionaries to add to new dict. To end your choice type '0':";
	bool choice_end = false;
	while (!choice_end)
	{
		int i = dialog(dicts, "\r");
		if (!i)
			choice_end = true;
		else
		{
			add_dicts->push_back(dicts->at(i));
			dicts->erase(dicts->begin() + i);
		}
	}

	if (add_dicts->size() == 0)
	{
		cout << "Your choice is empty." << endl;
		delete mt;
		delete dicts;
		delete add_dicts;
		return 1;
	}
	else
	{
		delete dicts;
	}

	ifstream file_i;
	string str;
	bool flag;
	uint64_t total_words = 0, lang_words = 0;

	/* CHECK WORDS */

	mt->Sort(1);

	for (int i = 0; i < add_dicts->size(); i++)
	{
		file_i.open("dicts\\" + add_dicts->at(i));

		while (getline(file_i, str))
		{
			if (IsRestricted(str))
				continue;

			/* check letter by letter */

			//cout << str << "\t\t\t\t\t\r";

			flag = IsThatLangWord(mt, str);

			/* write to new dict */

			if (flag)
			{
				file_o << str << endl;
				lang_words++;
			}
			total_words++;
			if (!(lang_words % 1000))
				cout << "Lang words: " << lang_words << "/" << total_words << "\t\t\t\r";
		}

		file_i.close();
	}

	file_o.close();

	cout << endl << endl;

	delete mt;
	delete add_dicts;

	return 1;
}

int DifferPasswordsByLang(rainbow_t **rt)
{
	// cut due to the lack of an objective assessment of the importance of development
}

int CreateOrUpdateMTable(rainbow_t **rt)
{
	/*
	FILES:
		dictname_dict.txt - main one
		dictname_mtable.txt:
			- dictname(s)
			- character set
			- max length of words
			- all stats in uint_fast64_t (SORTED!)
			- all chars (SORTED!)
	*/

	/*FIND ALL DICS AND CHOOSE*/

	bool success = false;
	bool choice_end;
	string pattern;
	string folder;
	vector<string> *dicts;
	vector<string> *add_dicts;
	vector<string> options = { "Quit", "New markov table", "Add stats to existing" };
	vector<string> yesno = { "Yes", "No" };
	int opt, sub_opt;
	int i, j;
	markov3_t *temp_mt = new markov3_t();

	while (!success)
	{
		folder = "";

		/*FIND AND CHOOSE MTABLE MODE: NEW OR ADD TO EXISTING*/

		opt = dialog(&options, "Choose option:");

		if (opt == 0)
		{
			delete temp_mt;
			return 1;
		}

		/*CHOOSE SEVERAL DICTS TO ADD*/

			/*read all dicts*/

		if (opt == 1)
		{
			sub_opt = dialog(&yesno, "Create language dependant:");
			if (sub_opt == 0)
			{
				pattern = "dicts\\lang_dicts\\*.txt";
				dicts = FolderFiles(pattern);
				int g = dialog(dicts, "Choose dictionary:");
				add_dicts = new vector<string>(1, "");
				add_dicts->at(0) = dicts->at(g);
				temp_mt->filename = "mt3l_pure_" + dicts->at(g).substr(0, 3) + ".txt";
				temp_mt->lang = true;
				temp_mt->AddStats(add_dicts, folder);
				success = true;
				continue;
			}
		}

		sub_opt = dialog(&yesno, "Choose dicts from differented:");
		if (sub_opt == 0)
		{
			pattern = "dicts\\diff_dicts\\*.txt";
			folder = "diff_dicts\\";
		}
		else
			pattern = "dicts\\*.txt";
		dicts = FolderFiles(pattern);
		dicts->insert(dicts->begin(), "End choice");
		dicts->shrink_to_fit();

		if (opt == 2) //if add to existing mtable
		{
			/*CHOOSE MULTI MTABLE TO ADD TO*/

			delete temp_mt;
			temp_mt = ChooseMT();

			/*READ DICTS USED ALREADY*/

			cout << "Dictionaries already used:" << endl;
			for (i = 0; i < temp_mt->dicts->size(); i++)
				cout << i << ". " << temp_mt->dicts->at(i) << endl;
			cout << endl;

			/*exclude already used dicts*/

			for (i = 0; i < temp_mt->dicts->size(); i++)
				for (j = 0; j < dicts->size(); j++)
					if (dicts->at(j) == temp_mt->dicts->at(i))
					{
						dicts->erase(dicts->begin() + j);
						j--;
					}
		}

		/*choose from what remains*/

		add_dicts = new vector<string>();
		cout << "Choose dictionaries to add to markov table. To end your choice type '0':";
		choice_end = false;
		while (!choice_end)
		{
			i = dialog(dicts, "\r");
			if (!i)
				choice_end = true;
			else
			{
				add_dicts->push_back(dicts->at(i));
				dicts->erase(dicts->begin() + i);
			}
		}

		if (add_dicts->size() == 0)
		{
			cout << "Your choice is empty." << endl;
			//delete temp_mt;
			delete add_dicts;
			continue;
		}
		else
		{
			cout << "Your choice is:" << endl;
			for (i = 0; i < add_dicts->size(); i++)
				cout << "\t" << add_dicts->at(i) << endl;

			i = dialog(&yesno, "Right?");
			if (!i)
			{
				temp_mt->lang = false;

				temp_mt->AddStats(add_dicts, folder);
				success = true;
			}
			else
			{
				delete temp_mt;
				delete add_dicts;
			}
		}
	}

	delete temp_mt;
	delete add_dicts;

	return 1;
}

int CheckMTRating(rainbow_t **rt)
{
	markov3_t *mt = ChooseMT();
	mt->CheckRating();

	return 1;
}

int CreateRT(rainbow_t **rt)
{
	uint8_t humanity = 0;
	int tmp_hum;
	uint8_t min_l;
	uint8_t max_l;
	uint32_t chain_l;
	uint64_t chain_am = 0;
	uint64_t time = 0, temp_time;
	string err = "";

	/*CHOOSE DEVICE: CPU OR GPU*/

	vector<string> devices = { "CPU", "GPU" };
	int dev = dialog(&devices, "Choose device for computing:");

	/*CHOOSE MODE*/

	vector<string> modes = { "Perfect", "Non Perfect" };
	int opt = dialog(&modes, "Choose gen mode for this RT:");

	vector<string> human = { "Random", "Human" };
	int hum = dialog(&human, "Choose plain mode for this RT:");

	markov3_t *mt = nullptr;
	if (hum)
	{
		mt = ChooseMT();
		do
		{
			cout << err << "Enter humanity (from 1 to 255): ";
			tmp_hum = (int)getNum();
			err = "Error! Repeat, please!\n";
		} while ((tmp_hum < 1) || (tmp_hum > 255));
		err = "";

		humanity = (uint8_t)tmp_hum;
	}

	vector<string> time_modes = { "By fixed time", "By fixed chain amount" };
	int time_mode = dialog(&time_modes, "Choose time mode for this RT:");

	/*ENTER PARAMETERS*/

	do
	{
		cout << err << "Enter minimum password length (from 6 to " << MAX_PASS_L << "): ";
		min_l = (uint8_t)getNum();
		err = "Error! Repeat, please!\n";
	} while ((min_l < 6) || (min_l > MAX_PASS_L));
	err = "";

	do
	{
		cout << err << "Enter maximum password length (from 6 to " << MAX_PASS_L << "): ";
		max_l = (uint8_t)getNum();
		err = "Error! Repeat, please!\n";
	} while ((max_l < 6) || (max_l > MAX_PASS_L));
	err = "";

	do
	{
		cout << err << "Enter chain length (from 1 to 2500): ";
		chain_l = (uint32_t)getNum();
		err = "Error! Repeat, please!\n";
	} while ((chain_l < 1) || (chain_l > 2500));
	err = "";

	int blocksNum_hum = SM_COUNT * BLOCKS_PER_SM;
	int blocksNum_ran = SM_COUNT * BLOCKS_PER_SM * 10;

	if (time_mode == 0)
	{
		do
		{
			cout << err << "Enter gen hours (from 0 to 5): ";
			temp_time = getNum();
			err = "Error! Repeat, please!\n";
		} while (temp_time > 5);
		err = "";

		time += temp_time * 3600;

		do
		{
			cout << err << "Enter gen minutes (from 0 to 59): ";
			temp_time = getNum();
			err = "Error! Repeat, please!\n";
		} while (temp_time > 59);
		err = "";

		time += temp_time * 60;

		do
		{
			cout << err << "Enter gen seconds (from 0 to 59): ";
			temp_time = getNum();
			err = "Error! Repeat, please!\n";
		} while (temp_time > 59);
		err = "";

		time += temp_time;

		cout << endl << "Expected gen time of this RT:\t" << time / 3600 << "h " << (time % 3600) / 60 << "m " << time % 60 << "s" << endl;
	}
	if (time_mode == 1)
	{
		do
		{
			cout << err << "Enter chain amount (from 500 to 1000000000): ";
			chain_am = getNum();
			err = "Error! Repeat, please!\n";
		} while ((chain_am < 500) || (chain_am > 1000000000));
		err = "";

		uint64_t rt_size = chain_am * (uint64_t)(1.5 * (float)min_l + 0.5 * (float)max_l + 2.0);
		cout << endl << "Average size of this RT:\t" << (float)rt_size / (1024 * 1024) << " MB" << endl;
	}

	/*CREATE RT*/

	delete *rt;
	string filename;
	if (hum)
	{
		//filename = "rth" + TimeStamp() + "_new";
		filename = "rth" + TimeStamp();
		*rt = new rainbow_t(chain_am, chain_l, mt->charset, min_l, max_l, mt, humanity);
	}
	else
	{
		//filename = "rtr" + TimeStamp() + "_new";
		filename = "rtr" + TimeStamp();
		*rt = new rainbow_t(chain_am, chain_l, def_charset, min_l, max_l, mt, humanity);
	}

	(*rt)->setFilename(filename);
	string full_path = curDir + "rtables\\" + filename;
	CreateDirectoryA(full_path.c_str(), NULL);

	if (dev == 0)
	{
		if (opt == 0)
		{
			if (hum == 0)
				(*rt)->GenerateUniqueRT(false);
			if (hum == 1)
				(*rt)->GenerateUniqueRT(true);
		}
		if (opt == 1)
		{
			if (hum == 0)
				(*rt)->GenerateRT(false);
			if (hum == 1)
				(*rt)->GenerateRT(true);
		}
	}
	if (dev == 1)
	{
		if (opt == 0)
		{
			if (hum == 0)
			{
				if (time_mode == 0)
					(*rt)->GenerateUniqueRTbyTime_CUDA(blocksNum_ran, 512, false, time);
				if (time_mode == 1)
					(*rt)->GenerateUniqueRT_CUDA(blocksNum_ran, 512, false);
			}
			if (hum == 1)
			{
				if (time_mode == 0)
					(*rt)->GenerateUniqueRTbyTime_CUDA(blocksNum_hum, 512, true, time);
				if (time_mode == 1)
					(*rt)->GenerateUniqueRT_CUDA(blocksNum_hum, 512, true);
			}
		}
		if (opt == 1)
		{
			if (hum == 0)
			{
				if (time_mode == 0)
					(*rt)->GenerateRTbyTime_CUDA(blocksNum_ran, 512, false, time);
				if (time_mode == 1)
					(*rt)->GenerateRT_CUDA(blocksNum_ran, 512, false);
			}
			if (hum == 1)
			{
				if (time_mode == 0)
					(*rt)->GenerateRTbyTime_CUDA(blocksNum_hum, 512, true, time);
				if (time_mode == 1)
					(*rt)->GenerateRT_CUDA(blocksNum_hum, 512, true);
			}

		}
	}

	(*rt)->WriteRT_Info();

	return 1;
}

int ReadRT(rainbow_t **rt)
{
	string pattern = "rtables\\rt*";
	vector<string> *rts = FolderFiles(pattern);
	int opt = dialog(rts, "Choose RT to open:");
	delete *rt;
	*rt = new rainbow_t(rts->at(opt));

	return 1;
}

int SearchHashes(rainbow_t **rt)
{
	/*ENTER AMOUNT OF HASHES IT NEEDS TO SEARCH FOR*/

	string err = "";
	uint16_t amount;

	if ((*rt)->getFilename() == "")
	{
		cout << "There's no RT to examine. Please read one first." << endl;
		return 1;
	}

	vector<string> modes = { "CPU", "GPU" };
	int mode = dialog(&modes, "Choose mode of prob checking:");

	if (mode == 0)
	{
		do
		{
			cout << err << "Enter hashes amount (from 10 to 17920): ";
			amount = (uint16_t)getNum();
			err = "Error! Repeat, please!\n";
		} while ((amount < 10) || (amount > 17920));

		(*rt)->RTProbability(amount);
	}
	if (mode == 1)
	{
		do
		{
			cout << err << "Enter hashes amount (from 10 to 17920; 0 - for hash file reading): ";
			amount = (uint16_t)getNum();
			err = "Error! Repeat, please!\n";
		} while (((amount > 0) && (amount < 10)) || (amount > 17920));

		vector<string> express = { "Half-accelerated", "Express" };
		int expr = dialog(&express, "Choose acceleration:");

		if ((*rt)->getMTptr())
		{
			if (expr == 0)
				(*rt)->SearchProb_CUDA(amount, SM_COUNT * BLOCKS_PER_SM * 2, 512);
			if (expr == 1)
				(*rt)->SearchProbExpress_CUDA(amount, SM_COUNT * BLOCKS_PER_SM * 2, 512);
		}
		else
		{
			if (expr == 0)
				(*rt)->SearchProb_CUDA(amount, SM_COUNT * BLOCKS_PER_SM * 20, 512);
			if (expr == 1)
				(*rt)->SearchProbExpress_CUDA(amount, SM_COUNT * BLOCKS_PER_SM * 20, 512);
		}
	}

	(*rt)->WriteRT_Info();

	return 1;
}

int WordCounter(rainbow_t **rt)
{
	vector<string> *options;
	int opt, finder;
	string pattern = "*";
	string filename;
	uint64_t counter;

	ofstream counts;
	counts.open("counts.txt", std::ios::app);

	do
	{
		options = FolderFiles(pattern);
		opt = dialog(options, "Choose file to read (you may open folders):");
		finder = (int)(options->at(opt).find("."));
		pattern.erase(pattern.end() - 1);
		if (finder >= 0)
		{
			finder = (int)(options->at(opt).find(".txt"));
			if (finder < 0)
			{
				cout << "You can read only .txt files!" << endl;
				pattern = "*";
				continue;
			}
			filename = pattern + options->at(opt);
			counter = WordCount(filename);
			cout << "Total lines: " << counter << endl;
			counts << filename << "\t\t" << counter << endl;
			//cout << "20%: " << counter / 5 << endl;
		}
		else
			pattern += options->at(opt) + "\\*";
		delete options;
	} while (finder < 0);

	counts.close();

	return 1;
}

int MakeHashList(rainbow_t **rt)
{
	vector<string> langs = { "eng_dict.txt", "neu_dict.txt", "fra_dict.txt", "ger_dict.txt" };
	vector<uint8_t> sizes = { 52, 31, 9, 8 };
	vector<uint8_t> sizes_sums = { 52, 83, 92, 100 };
	string mainfilename = "dicts\\diff_dicts\\hibp-v2_";
	uint64_t word_size_cntr, counter;
	uint8_t min_l = 6, max_l = 20;
	vector<string> *plains = new vector<string>();
	plains->reserve((max_l - min_l + 1) * 100);

	vector<string> *plains_splash = new vector<string>(50, "");
	ifstream temp_file;
	temp_file.open("plains_splashdata.txt");
	string str, filename;
	for (uint8_t i = 0; getline(temp_file, str); i++)
		plains_splash->at(i) = str;
	temp_file.close();

	ofstream splash_hash;
	splash_hash.open("hashes_splashdata.txt");
	for (uint8_t i = 0; i < (uint8_t)(plains_splash->size()); i++)
		splash_hash << picosha2::hash256_hex_string(plains_splash->at(i)) << endl;
	splash_hash.close();

	bool found;

	for (uint8_t length = min_l; length <= max_l; length++)
	{
		for (uint8_t lang = 0; lang < 4; lang++)
		{
			filename = mainfilename + langs.at(lang);

			cout << "Reading dict " << langs.at(lang) << ", length = " << (int)length << "...\t\t\t\r";


			word_size_cntr = WordSizeCount(filename, length);
			if (word_size_cntr < (uint64_t)(sizes.at(lang)))
			{
				cout << "Amount of words of " << langs.at(lang) << " with length of " << (int)length << " is less, than total needed!" << endl << endl;
				delete plains;
				delete plains_splash;
				return 1;
			}
			else
			{
				cout << "Amount of words of " << langs.at(lang) << " with length of " << (int)length << " is " << word_size_cntr << ". Adding...\t\t\r";
			}
			word_size_cntr /= sizes.at(lang);
			temp_file.open(filename);
			counter = 0;
			while (getline(temp_file, str) && (plains->size() != (100 * (length - min_l) + sizes_sums.at(lang))))
			{
				if ((uint8_t)(str.size()) == length)
				{
					if (!(counter % word_size_cntr))
					{
						/* check if it exists in splash */

						found = false;
						for (uint8_t i = 0; (i < 50) && (!found); i++)
							if (plains_splash->at(i) == str)
								found = true;

						if (found)
							continue;

						/* check if it exists in plains already */

						found = false;
						for (uint16_t i = (length - min_l) * 100; (i < plains->size()) && (!found); i++)
							if (plains->at(i) == str)
								found = true;

						if (found)
							continue;
						else
						{
							plains->push_back(str);
							counter++;
						}
					}
					else
						counter++;
				}
			}
			temp_file.close();

			if (plains->size() != (100 * (length - min_l) + sizes_sums.at(lang)))
			{
				cout << "Plains size is not right!\t\t\t\t" << endl << endl;
				delete plains;
				delete plains_splash;
				return 1;
			}
		}
	}

	ofstream plains_file, hashes_file;
	plains_file.open("plains.txt");
	hashes_file.open("hashes.txt");

	for (uint16_t i = 0; i < plains->size(); i++)
	{
		plains_file << plains->at(i) << endl;
		hashes_file << picosha2::hash256_hex_string(plains->at(i)) << endl;
	}

	plains_file.close();
	hashes_file.close();

	return 1;
}

int SearchHashesMultRT(rainbow_t **rt)
{
	/* choose RT */
	
	string pattern = "rtables\\rt*";
	vector<string> *rts_aval;
	vector<string> *rts_to_add = new vector<string>();

	int opt;
	bool choice_end = false;
	bool choice_done = false;
	vector<string> yesno = { "Yes", "No" };

	while (!choice_done)
	{
		rts_aval = FolderFiles(pattern);
		rts_aval->insert(rts_aval->begin(), "End choice");
		rts_aval->shrink_to_fit();

		while (!choice_end)
		{
			opt = dialog(rts_aval, "Choose RTs to search in:");
			if (!opt)
				choice_end = true;
			else
			{
				rts_to_add->push_back(rts_aval->at(opt));
				rts_aval->erase(rts_aval->begin() + opt);
			}
		}
		if (rts_to_add->size() == 0)
		{
			cout << "Your choice is empty!" << endl;
			delete rts_to_add;
			delete rts_aval;
			return 1;
		}
		else
		{
			cout << "Your choice is:" << endl;
			for (uint8_t i = 0; i < (uint8_t)(rts_to_add->size()); i++)
				cout << "\t" << rts_to_add->at(i) << endl;
			opt = dialog(&yesno, "Right?");
			if (!opt)
				choice_done = true;
			else
				rts_to_add->clear();
			delete rts_aval;
		}
	}

	/* sort rts in chain amount order (higher to lower) */

	vector<uint64_t> *ch_ams = new vector<uint64_t>();
	ch_ams->reserve(rts_to_add->size());
	ch_ams->resize(rts_to_add->size());

	rainbow_t *rt_tmp;
	for (uint8_t i = 0; i < rts_to_add->size(); i++)
	{
		rt_tmp = new rainbow_t(rts_to_add->at(i));
		ch_ams->at(i) = rt_tmp->getChain_am();
		delete rt_tmp;
	}
	for (uint8_t i = 0; i < rts_to_add->size(); i++)
		for (uint8_t j = 0; j < rts_to_add->size() - i - 1; j++)
			if (ch_ams->at(j) < ch_ams->at(j + 1))
			{
				std::swap(ch_ams->at(j), ch_ams->at(j + 1));
				std::swap(rts_to_add->at(j), rts_to_add->at(j + 1));
			}

	delete ch_ams;

	vector<string> *ids = new vector<string>();
	ids->reserve(rts_to_add->size());
	ids->resize(rts_to_add->size());

	for (uint8_t i = 0; i < (uint8_t)(rts_to_add->size()); i++)
		ids->at(i) = rts_to_add->at(i).substr(4, 3);

	vector<string> *plains = new vector<string>();
	vector<string> *hashes = new vector<string>();
	vector<string> *found_plains = new vector<string>();
	vector<vector<string> *> *found_pl_ids = new vector<vector<string> *>();

	ifstream file1, file2;
	file1.open("plains.txt");
	file2.open("hashes.txt");
	uint16_t h_amount = 1500;

	plains->reserve(h_amount);
	plains->resize(h_amount);
	hashes->reserve(h_amount);
	hashes->resize(h_amount);
	string str;
	for (uint16_t i = 0; i < h_amount; i++)
	{
		getline(file1, str);
		plains->at(i) = str;
		getline(file2, str);
		hashes->at(i) = str;
	}
	file1.close();
	file2.close();
	found_plains->assign(h_amount, "");
	found_pl_ids->reserve(h_amount);
	found_pl_ids->resize(h_amount);
	for (uint16_t i = 0; i < h_amount; i++)
		found_pl_ids->at(i) = new vector<string>();

	/* transform to uint32_t array */

	uint32_t *hash_array = (uint32_t *)calloc(h_amount * 8, sizeof(uint32_t));
	for (uint64_t counter = 0; counter < (h_amount * 8); counter++)
		hash_array[counter] = std::stoul(hashes->at(counter / 8).substr((counter % 8) * 8, 8), nullptr, 16);

	/* run kernel (get vector)*/

	vector<string> *result;

	/* run end search */

	uint64_t new_big_amount = h_amount;
	uint16_t loop_size = 10000;
	uint64_t skip = 0;
	uint64_t found_hashes = 0;
	uint64_t skip_found = 0;

	cout << "Trying to compute successful search percentage..." << endl;

	for (uint8_t k = 0; k < (uint8_t)(rts_to_add->size()); k++)
	{
		cout << "Readind RT " << rts_to_add->at(k) << "..." << endl;
		rt_tmp = new rainbow_t(rts_to_add->at(k));

		for (uint8_t part_counter = rt_tmp->getPart(), iter = 0; iter < rt_tmp->getPart_am(); part_counter = (part_counter + 1) % (rt_tmp->getPart_am()), iter++)
		{
			uint16_t loop_size_tmp = loop_size;
			uint16_t loops = (uint16_t)(new_big_amount / loop_size);
			if (new_big_amount % loop_size)
				loops++;

			if (part_counter != rt_tmp->getPart())
				rt_tmp->Read_Part(part_counter);

			cout << "Checking RT part " << (int)iter + 1 << "/" << rt_tmp->getPart_am() << "..." << endl;

			for (uint16_t loop_counter = 0; loop_counter < loops; loop_counter++)
			{
				uint32_t *hash_array_temp = &(hash_array[(uint64_t)loop_counter * (uint64_t)loop_size_tmp * 8]);

				if ((loop_counter == (loops - 1)) && (new_big_amount % loop_size))
					loop_size_tmp = new_big_amount % loop_size;

				result = Search_Prob_Express_CUDA(rt_tmp, hash_array_temp, loop_size_tmp, 80, 512);

				bool hash_found;
				uint64_t last_found = 0;
				for (uint16_t j = 0; j < loop_size_tmp; j++)
				{
					if (result->at(j) != "")
					{
						/* find hash */

						cout << "Checking hash " << j << "/" << loop_size_tmp << "\t\t\r";

						hash_found = false;
						for (uint64_t i = last_found + loop_counter * loop_size; (i < (last_found + loop_counter * loop_size + loop_size_tmp)) && (!hash_found); i++)
						{
							uint16_t hash_cntr;
							hash_found = true;
							for (hash_cntr = 0; (hash_cntr < 8) && (hash_found); hash_cntr++)
							{
								if (hash_array_temp[(uint64_t)j * 8 + hash_cntr] != std::stoul(hashes->at(i).substr((hash_cntr) * 8, 8), nullptr, 16))
									hash_found = false;
							}
							if ((hash_cntr == 8) && (hash_found))
							{
								last_found = (i + 1) % loop_size;
								found_plains->at(i) = result->at(j);
								if (found_pl_ids->at(i)->size())
									if (found_pl_ids->at(i)->back() != ids->at(k))
										found_pl_ids->at(i)->push_back(ids->at(k));
								if (!(found_pl_ids->at(i)->size()))
									found_pl_ids->at(i)->push_back(ids->at(k));
								found_hashes++;
							}
						}
					}
				}

				cout << "Found: " << found_hashes << "/" << h_amount << "\t\t" << endl;

				delete result;
			}

			/* renew hash array to calculate less */

			free(hash_array);

			new_big_amount = h_amount - found_hashes;
			uint64_t new_b_a_cntr = 0;
			hash_array = (uint32_t *)calloc(new_big_amount * 8, sizeof(uint32_t));

			for (uint64_t counter = 0; counter < h_amount; counter++)
			{
				if (found_plains->at(counter) == "")
				{
					for (uint16_t sub_cntr = 0; sub_cntr < 8; sub_cntr++)
						hash_array[new_b_a_cntr * 8 + sub_cntr] = std::stoul(hashes->at(counter).substr((sub_cntr) * 8, 8), nullptr, 16);
					new_b_a_cntr++;
				}
			}
		}

		delete rt_tmp;
	}

	free(hash_array);

	ofstream report_file;
	report_file.open("search_report_4.txt", std::ios::app);

	report_file << "\t\tSEARCH EFFICIENCY RESEARCH" << endl << endl;
	report_file << "Used RTs:" << endl;
	for (uint8_t i = 0; i < rts_to_add->size(); i++)
		report_file << "\t" << rts_to_add->at(i) << endl;
	report_file << endl;
	report_file << "Hash\t\t\t\t\t\t\t ---> Plain\t\tRT_ID" << endl << endl;

	vector<uint16_t> *lang_recovered = new vector<uint16_t>();
	lang_recovered->assign(rts_to_add->size(), 0);
	
	vector<uint16_t> *length_recovered = new vector<uint16_t>();
	length_recovered->assign(15, 0);
	
	for (uint16_t i = 0; i < h_amount; i++)
	{
		if (found_plains->at(i) != "")
		{
			report_file << hashes->at(i) << " ---> " << found_plains->at(i) << "\t";
			for (uint8_t j = 0; j < found_pl_ids->at(i)->size(); j++)
			{
				report_file << found_pl_ids->at(i)->at(j) << " ";
				for (uint8_t k = 0; k < ids->size(); k++)
					if (found_pl_ids->at(i)->at(j) == ids->at(k))
						lang_recovered->at(k)++;
			}
			report_file << endl;

			length_recovered->at(i / 100)++;
		}
	}

	report_file << endl;
	report_file << "Total search efficiency:\t" << found_hashes << " (" << ((double)found_hashes / h_amount) * 100 << "%)" << endl;
	cout << "Total search efficiency:\t" << found_hashes << " (" << ((double)found_hashes / h_amount) * 100 << "%)" << endl;
	report_file << "Plains recovered by RT:" << endl;
	for (uint8_t i = 0; i < lang_recovered->size(); i++)
		report_file << "\t" << ids->at(i) << ":\t" << lang_recovered->at(i) << " (" << ((double)(lang_recovered->at(i)) / h_amount) * 100 << "%)" << endl;
	report_file << endl;
	
	report_file << "Length recovery efficiency:" << endl;
	for (uint16_t k = 0; k < length_recovered->size(); k++)
		report_file << "\t" << k + 6 << ":\t" << length_recovered->at(k) << " (" << ((double)(length_recovered->at(k)) / h_amount) * 100 << "%)" << endl;
	
	report_file.close();

	delete rts_to_add;
	delete ids;
	delete plains;
	delete hashes;
	delete found_plains;
	for (uint16_t i = 0; i < found_pl_ids->size(); i++)
		delete found_pl_ids->at(i);
	delete found_pl_ids;
	delete lang_recovered;
	delete length_recovered;

	return 1;
}

int RTInfo(rainbow_t **rt)
{
	cout << "FILE:\t" << (*rt)->getFilename() << endl;
	if ((*rt)->getMTptr())
	{
		cout << "PLAIN MODE:\thuman" << endl;
		cout << "MT FILE:\t" << (*rt)->getMTFilename() << endl;
		cout << "HUMANITY:\t" << (int)((*rt)->getHumanity()) << endl;
	}
	else
		cout << "PLAIN MODE:\trandom" << endl;

	cout << "CHARSET:\t" << (*rt)->getCharset() << endl;
	cout << "MIN PASSWORD LENGTH:\t" << (int)((*rt)->getMin_l()) << endl;
	cout << "MAX PASSWORD LENGTH:\t" << (int)((*rt)->getMax_l()) << endl;
	cout << "CHAIN LENGTH:\t" << (*rt)->getChain_l() << endl;
	cout << "CHAIN AMOUNT:\t" << (*rt)->getChain_am() << endl;
	cout << "RT REAL PROBABILITY:\t" << (*rt)->getProb() << "%" << endl;
	cout << "HASH SEARCH TIME:\t" << (*rt)->getSearchTime() << "ms" << endl;
	cout << "RT GENERATING TIME:\t" << (*rt)->getGen_dur() / 60000 << "m " << ((*rt)->getGen_dur() % 60000) / 1000 << "s " << (*rt)->getGen_dur() % 1000 << "ms" << endl;

	return 1;
}

int(*f_ptr[])(rainbow_t **rt) = {
	NULL,
	GPUInfo,
	MakeLangPassDict,
	DifferPasswordsByLang,
	CreateOrUpdateMTable,
	CheckMTRating,
	CreateRT,
	ReadRT,
	SearchHashes,
	WordCounter,
	MakeHashList,
	SearchHashesMultRT,
	RTInfo
};

vector<string> msgs = {
	"Quit",
	"GPU Info",
	"Make New Language Dependant Dictionary",
	"Differ Passwords By Language",
	"Create or Update Markov Table",
	"Check MT Rating",
	"Create RT",
	"Read RT from file",
	"Search Hashes",
	"Word Counter",
	"Make Hashlists",
	"Search Efficiency Reasearch"
};

int main()
{
	int rc, n;
	rainbow_t *rt = new rainbow_t();
	bool inserted = false;
	while (rc = dialog(&msgs, "Main menu:"))
	{
		n = f_ptr[rc](&rt);
		if (!n)
			break;
		cout << endl << "====================" << endl;

		if ((!inserted) && (rt->getFilename() != ""))
		{
			//msgs.push_back("Test SHA-256");
			msgs.push_back("RT Info");
			//msgs.push_back("Test Hex to Byte");
			msgs.shrink_to_fit();
			inserted = true;
		}
	}
	delete rt;
	cudaDeviceReset();

	// No further CUDA API calls!!

	cout << endl << "That's all." << endl;
	system("pause");
	return 0;
}