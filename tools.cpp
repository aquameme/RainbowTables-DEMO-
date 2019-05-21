#include "tools.h"

uint64_t getNum()
{
	string err = "", str;
	uint64_t num;
	int i;
	bool found;
	do
	{
		cout << err;
		err = "Error! Repeat, please!\n";
		getline(cin, str);
		found = false;
		for (i = 0; (!(str.empty())) && (i < str.size()) && (!found); i++)
			if (str.at(i) < 48 || (str.at(i) > 57))
				found = true;
	} while (found);
	num = stoull(str, nullptr, 0);

	return num;
}

int dialog(vector<string> *msgs, string header)
{
	string err = "";
	int rc;
	cout << header;
	do
	{
		cout << err << endl;
		err = "Gde you see etot clause? Repeat pzhlsta!\n";
		for (int i = 0; i < msgs->size(); ++i)
			cout << i << ". " << msgs->at(i) << endl;
		cout << "Make your choice: ";
		rc = (int)getNum();
	} while (rc < 0 || rc >= msgs->size());
	cout << endl;

	return rc;
}

vector<string> *FolderFiles(string folder)
{
	vector<string> *results = new vector<string>();

	_WIN32_FIND_DATAA founddata;
	HANDLE h_found;
	string str;
	h_found = FindFirstFileA(folder.c_str(), &founddata);
	do
	{
		str = founddata.cFileName;
		results->push_back(str);
	} while (FindNextFileA(h_found, &founddata) > 0);
	results->shrink_to_fit();

	return results;
}

string TimeStamp()
{
	string stamp = "";
	time_t seconds = time(NULL);
	tm* timeinfo = new tm();
	localtime_s(timeinfo, &seconds);
	stamp += "_"
		+ std::to_string(timeinfo->tm_year - 100)
		+ std::to_string((timeinfo->tm_mon + 1) / 10)
		+ std::to_string((timeinfo->tm_mon + 1) % 10)
		+ std::to_string(timeinfo->tm_mday / 10)
		+ std::to_string(timeinfo->tm_mday % 10)
		+ "_"
		+ std::to_string(timeinfo->tm_hour / 10)
		+ std::to_string(timeinfo->tm_hour % 10)
		+ std::to_string(timeinfo->tm_min / 10)
		+ std::to_string(timeinfo->tm_min % 10);
	return stamp;
}

string random_random_string(uint8_t min_l, uint8_t max_l)
{
	string word;

	/*GET RANDOM KEY FOR EVERYTHING*/

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint64_t> dist(0, 0xffffffffffffffff);
	uint64_t key = dist(gen);

	/*GET RANDOM HASH AND REVERT IT*/

	word = picosha2::hash256_hex_string(std::to_string(key));
	word = reduction_rand_cpu(word, 0, min_l, max_l, 0, 0);

	return word;
}

string next_random_string(uint8_t min_l, uint8_t max_l, string prev_str)
{
	string word;
	if (prev_str == "")
	{
		word.assign(min_l, 32);
		return word;
	}

	word = prev_str;

	for (int i = (int)(prev_str.size() - 1); i >= 0; i--)
	{
		if ((int)(word.at(i) - 32) < 94)
		{
			word.at(i) += 1;
			for (int j = i + 1; j < prev_str.size(); j++)
				word.at(j) = 32;
			return word;
		}
	}
	word.resize(prev_str.size() + 1);
	for (int i = 0; i < word.size(); i++)
		word.at(i) = 32;
	return word;
}

string random_human_string(uint8_t min_l, uint8_t max_l, markov3_t *mt, uint8_t humanity)
{
	string word;

	/*GET RANDOM KEY FOR EVERYTHING*/

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint64_t> dist(0, 0xffffffffffffffff);
	uint64_t key = dist(gen);

	/*GET RANDOM HASH AND REVERT IT*/

	word = picosha2::hash256_hex_string(std::to_string(key));
	word = reduction_human_cpu(word, 0, min_l, max_l, mt, 0, 0, humanity);

	return word;
}

string next_human_string(uint8_t min_l, uint8_t max_l, markov3_t *mt, string prev_str)
{
	string word = prev_str;

	/* calculate size */

	int i, j, n;
	bool change = false;
	bool bad_letter;
	char prev_char = 32, prev_prev_char = 32;

	for (n = (int)(word.size() - 1); (n >= 0) && (!change); n--)
	{
		/* find prev char */

		i = 0;
		if (n > 0)
		{
			prev_char = word.at(n - 1);
			if (n > 1)
				prev_prev_char = word.at(n - 2);
		}
		while ((i < 95) && (mt->stats_char->at(n)->at(prev_prev_char - 32)->at(prev_char - 32)->at(i) != word.at(n)))
			i++;

		do
		{
			bad_letter = false;

			if ((i < 94) && (mt->stats->at(n)->at(prev_prev_char - 32)->at(prev_char - 32)->at(i + 1)))
			{
				word.at(n) = mt->stats_char->at(n)->at(prev_prev_char - 32)->at(prev_char - 32)->at(i + 1);
				change = true;

				for (j = n + 1; (j < word.size()) && (!bad_letter); j++)
				{
					prev_char = word.at(j - 1);
					if (j > 1)
						prev_prev_char = word.at(j - 2);
					else
						prev_prev_char = 32;
					if (mt->stats->at(j)->at(prev_prev_char - 32)->at(prev_char - 32)->at(0))
						word.at(j) = mt->stats_char->at(j)->at(prev_prev_char - 32)->at(prev_char - 32)->at(0);
					else
					{
						bad_letter = true;
						change = false;
						i++;
					}
				}
			}

		} while (bad_letter && (i < 94));
	}

	/* change length */

	int new_size = (int)(prev_str.size() + 1);
	vector<int> iters;

	if ((n < 0) && (!change))
	{
		while (!change)
		{
			if (new_size <= max_l)
			{
				if (new_size - 1)
					word.resize(new_size);
				else
				{
					word.resize(min_l);
					new_size = min_l;
				}

				j = 0;
				iters.push_back(0);
				for (i = 0; (i < word.size()) && (j < 95); i++)
				{
					j = iters.back();
					if (!i)
					{
						if (mt->stats->at(i)->at(0)->at(0)->at(j))
						{
							word.at(i) = mt->stats_char->at(i)->at(0)->at(0)->at(j);
							iters.push_back(0);
						}
						else
						{
							cout << "LOL, we ran out of ALL possible words" << endl;
							return "";
						}
					}
					if (i == 1)
					{
						if (mt->stats->at(i)->at(0)->at(word.at(i - 1) - 32)->at(j))
						{
							word.at(i) = mt->stats_char->at(i)->at(0)->at(word.at(i - 1) - 32)->at(j);
							iters.push_back(0);
						}
						else
						{
							i--;
							iters.pop_back();
							iters.at(iters.size() - 1)++;
						}
					}
					if (i > 1)
					{
						if (mt->stats->at(i)->at(word.at(i - 2) - 32)->at(word.at(i - 1) - 32)->at(j))
						{
							word.at(i) = mt->stats_char->at(i)->at(word.at(i - 2) - 32)->at(word.at(i - 1) - 32)->at(j);
							iters.push_back(0);
						}
						else
						{
							i--;
							iters.pop_back();
							iters.at(iters.size() - 1)++;
						}
					}
				}
				j = iters.back();
				if (j == 0)
					change = true;
			}
			else
			{
				cout << "LOL, we ran out of ALL possible words (cause of length)" << endl;
				return "";
			}
			new_size++;
		}
	}

	return word;
}

uint8_t *hexSHAtoByte(string hex_string)
{
	uint8_t *byte_string = (uint8_t *)calloc(32, sizeof(uint8_t));
	for (uint8_t i = 0; i < 32; i++)
	{
		if ((hex_string.at(2 * i) > 47) && (hex_string.at(2 * i) < 58))
			byte_string[i] = ((uint8_t)(hex_string.at(2 * i) - 48) * 16) & 0xf0;
		else
			byte_string[i] = ((uint8_t)(hex_string.at(2 * i) - 87) * 16) & 0xf0;

		if ((hex_string.at(2 * i + 1) > 47) && (hex_string.at(2 * i + 1) < 58))
			byte_string[i] |= (uint8_t)(hex_string.at(2 * i + 1) - 48) & 0x0f;
		else
			byte_string[i] |= (uint8_t)(hex_string.at(2 * i + 1) - 87) & 0x0f;
	}

	return byte_string;
}

vector<string>* CreateUniqueArray(uint64_t amount, uint8_t min_l, uint8_t max_l, markov3_t *mt, bool rand, string prev_str, bool human, uint8_t humanity) //true - random_s, false - next_s ; true - human, false - random
{
	string str;
	vector<string> *starts = new vector<string>();
	starts->assign(amount, "");
	uint64_t counter_chains, sub_counter;
	bool flag_exist;
	//double temp_perc;

	//cout << "Filling starts array progress: ";

	if (rand)
	{
		for (counter_chains = 0; counter_chains < amount; counter_chains++)
		{
			do
			{
				flag_exist = false;
				if (human)
					str = random_human_string(min_l, max_l, mt, humanity);
				else
					str = random_random_string(min_l, max_l);
				for (sub_counter = 0; (sub_counter < counter_chains) && (!flag_exist); sub_counter++)
					if (starts->at(sub_counter).compare(str) == 0)
						flag_exist = true;
			} while (flag_exist);
			starts->at(counter_chains) = str;
			/*
			temp_perc = int((counter_chains + 1) * 100 / amount);
			if (!((int)temp_perc % 10))
				cout << (int)temp_perc / 10 << (int)temp_perc % 10 << "%" << "\b\b\b";
			*/
		}
		//cout << (int)temp_perc / 10 << (int)temp_perc % 10 << "%\r";
	}
	else
	{
		str = prev_str;
		for (counter_chains = 0; counter_chains < amount; counter_chains++)
		{
			if (human)
				str = next_human_string(min_l, max_l, mt, str);
			else
				str = next_random_string(min_l, max_l, str);
			starts->at(counter_chains) = str;
			/*
			temp_perc = int((counter_chains + 1) * 100 / amount);
			if (!(counter_chains % 100))
				cout << (int)temp_perc / 10 << (int)temp_perc % 10 << "%" << "\b\b\b";
			*/
		}
		//cout << (int)temp_perc / 10 << (int)temp_perc % 10 << "%\r";
	}

	return starts;
}

uint64_t WordCount(string filename)
{
	ifstream file;
	file.open(filename);

	string str;
	uint64_t counter = 0;

	while (getline(file, str))
		counter++;

	file.close();

	return counter;
}

uint64_t WordSizeCount(string filename, uint8_t size)
{
	ifstream file;
	file.open(filename);

	string str;
	uint64_t counter = 0;

	while (getline(file, str))
		if ((uint8_t)(str.size()) == size)
			counter++;

	file.close();

	return counter;
}

bool HasNoLetters(string str)
{
	bool sym_only = true;
	for (int q = 0; (q < str.size()) && sym_only; q++)
		if (((str.at(q) > 64) && (str.at(q) < 91)) || ((str.at(q) > 96) && (str.at(q) <123)))
			sym_only = false;
	return sym_only;
}

bool OnlyNumbers(string str)
{
	bool num_only = true;
	for (int q = 0; (q < str.size()) && num_only; q++)
		if ((str.at(q) < 48) || (str.at(q) > 57))
			num_only = false;
	return num_only;
}

bool IsRestricted(string str)
{
	int restr;

	restr = (int)(str.find("$HEX", 0));
	if (restr >= 0)
		return true;

	/*
	restr = (int)(str.find(".com", 0));
	if (restr >= 0)
		return true;
	*/
	restr = (int)(str.find(".net", 0));
	if (restr >= 0)
		return true;

	restr = (int)(str.find(".co", 0));
	if (restr >= 0)
		return true;

	if ((str.size() < 6) || (str.size() > 20))
		return true;

	/*
	uint8_t letters = 0;
	//uint8_t nums = 0;
	for (int i = 0; i < str.size(); i++)
	{
		if (((str.at(i) > 64) && (str.at(i) < 91)) || ((str.at(i) > 96) && (str.at(i) < 123)))
			letters++;
		
		//if ((str.at(i) > 47) && (str.at(i) < 58))
			//nums++;
		
	}

	if (letters > 2)
	{
		// check if there's minimum 3 letters in a row

		int length = 0;
		for (int i = 0; i < str.size(); i++)
		{
			if (((str.at(i) > 64) && (str.at(i) < 91)) || ((str.at(i) > 96) && (str.at(i) < 123)))
			{
				length = 1;
				i++;
				while ((i < str.size()) && (((str.at(i) > 64) && (str.at(i) < 91)) || ((str.at(i) > 96) && (str.at(i) < 123))) && (length < 3))
				{
					i++;
					length++;
				}
				if (length > 2)
					return false;
			}
		}
	}
	*/
	return false;
}

bool IsKeystroked(string str)
{
	vector<string> keystrokes = {
		"qwertyuiop",
		"asdfghjkl",
		"zxcvbnm",
		"qaz",
		"wsx",
		"xyz",
		"abcdefghijklmnopqrstuvwxyz"
	};

	int k, n;
	for (uint8_t i = 0; i < keystrokes.size(); i++)
	{
		k = (int)(str.find(keystrokes.at(i).at(0), 0));
		if (k >= 0)
		{
			/* straight sequence check */

			for (n = k; (n < str.size()) && ((n - k) < keystrokes.at(i).size()); n++)
			{
				if (str.at(n) != keystrokes.at(i).at(n - k))
					break;
			}
			if (i == (keystrokes.size() - 1))
			{
				if ((n == str.size()) && ((n - k) >= 3))
					return true;
				if (n < (str.size() - 1))
					if (((n - k) >= 3) && ((str.at(n + 1) > 47) && (str.at(n + 1) < 58)))
						return true;
			}
			else
				if ((n - k) >= 3)
					return true;

			/* 1_2_3_ sequence check */

			for (n = k; (n < str.size()) && (((n - k) / 2) < keystrokes.at(i).size()); n += 2)
			{
				if (str.at(n) != keystrokes.at(i).at((n - k) / 2))
					break;
			}
			if (i == (keystrokes.size() - 1))
			{
				if ((n >= str.size()) && (((n - k) / 2) >= 3))
					return true;
				if (n < (str.size() - 1))
					if ((((n - k) / 2) >= 3) && ((str.at(n + 1) > 47) && (str.at(n + 1) < 58)))
						return true;
			}
			else
				if (((n - k) / 2) >= 3)
					return true;
		}
	}

	return false;
}
