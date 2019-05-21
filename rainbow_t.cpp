#include "rainbow_t.h"
#include "kernel_generate.cuh"

using picosha2::hash256_hex_string;

string EraserTillTab(string str)
{
	do
	{
		str.erase(str.begin());
	} while (str.at(0) != '\t');
	str.erase(str.begin());
	return str;
}

rainbow_t::rainbow_t()
{
	this->chain_am = 0;
	this->chain_l = 0;
	this->part_am = 0;
	this->part = 0;
	this->chain_am_part = 0;
	this->temp_size = 0;
	this->charset = "";
	this->filename = "";
	this->min_l = 0;
	this->max_l = 0;
	this->gen_dur = 0;
	this->mt = nullptr;
	this->humanity = 0;

	this->starts = nullptr;
	this->ends = nullptr;
	this->percs = nullptr;
}

rainbow_t::rainbow_t(uint64_t chain_am, uint32_t chain_l, string charset, uint8_t min_length, uint8_t max_length, markov3_t *mt, uint8_t humanity) //multi = false - common, multi = true - partial
{
	this->chain_am = chain_am;
	this->chain_l = chain_l;
	this->charset = charset;
	this->min_l = min_length;
	this->max_l = max_length;
	this->gen_dur = 0;
	this->mt = mt;
	this->humanity = humanity;

	this->starts = new vector<string>();
	this->ends = new vector<string>();
	this->percs = new vector<float>();
	this->percs->push_back(0.0);
	this->percs->push_back(0.0);
	this->percs->shrink_to_fit();

	if (chain_am)
	{
		this->part_am = (uint16_t)(chain_am * (1.5 * this->min_l + 0.5 * this->max_l + 3) / (256 * 1024 * 1024) + 1);
		if (chain_am > (256 * 1024 * 1024 / (1.5 * this->min_l + 0.5 * this->max_l + 3)))
			this->chain_am_part = (uint64_t)(256 * 1024 * 1024 / (1.5 * this->min_l + 0.5 * this->max_l + 3));
		else
			this->chain_am_part = chain_am;
		this->ends->reserve(this->chain_am_part);
	}
	else
	{
		this->part_am = 1;
		this->chain_am_part = 0;
	}

	this->part = 0;
	this->temp_size = 0;

	this->starts->assign(this->chain_am_part, "");
	this->ends->assign(this->chain_am_part, "");
}

rainbow_t::rainbow_t(string rt_folder)
{
	this->filename = rt_folder;

	ifstream file;
	file.open("rtables\\" + rt_folder + "\\" + "info.txt");

	/*NEW FILE FORMAT

	//	info.txt

	markov table filename
	humanity
	charset
	min_length
	max_length
	chain_length
	chain_amount
	part_amount
	gen_dur
	search_perc
	search_time

	//	rt_(timestamp)_(part_n).txt

	chain_amount_part

	texts[chain_amount * 2]

	*/

	uint64_t counter;
	string str;

	/* READ INFO */

	getline(file, str); //mode
	str = EraserTillTab(str);
	if (str == "human")
	{
		getline(file, str);
		str = EraserTillTab(str);
		this->mt = new markov3_t(str);

		getline(file, str);
		str = EraserTillTab(str);
		this->humanity = (uint8_t)stoi(str, nullptr, 0);
	}

	getline(file, str);
	str = EraserTillTab(str);
	this->charset = str;

	getline(file, str);
	str = EraserTillTab(str);
	this->min_l = (uint8_t)stoi(str, nullptr, 0);

	getline(file, str);
	str = EraserTillTab(str);
	this->max_l = (uint8_t)stoi(str, nullptr, 0);

	getline(file, str);
	str = EraserTillTab(str);
	this->chain_l = (uint32_t)stoul(str, nullptr, 0);

	getline(file, str);
	str = EraserTillTab(str);
	this->chain_am = (uint64_t)stoull(str, nullptr, 0);

	getline(file, str);
	str = EraserTillTab(str);
	this->part_am = (uint16_t)stoul(str, nullptr, 0);

	getline(file, str);
	str = EraserTillTab(str);
	this->gen_dur = (uint64_t)std::stoull(str, nullptr, 0);

	this->percs = new vector<float>(2);
	getline(file, str);
	if (str != "")
	{
		str = EraserTillTab(str);
		this->percs->at(0) = stof(str, nullptr);

		getline(file, str);
		str = EraserTillTab(str);
		this->percs->at(1) = stof(str, nullptr);
	}
	else
	{
		this->percs->at(0) = 0.0;
		this->percs->at(1) = 0.0;
	}

	file.close();

	/* READ FIRST PART */

	this->part = 0;
	file.open("rtables\\" + rt_folder + "\\" + "part_0.txt");

	getline(file, str);
	this->chain_am_part = (uint64_t)std::stoull(str, nullptr, 0);

	getline(file, str);

	this->starts = new vector<string>();
	this->ends = new vector<string>();

	this->starts->assign(this->chain_am_part, "");
	this->ends->assign(this->chain_am_part, "");

	for (counter = 0; counter < this->chain_am_part; counter++)
	{
		getline(file, str);
		this->starts->at(counter) = str;
		getline(file, str);
		this->ends->at(counter) = str;
	}

	file.close();
}

rainbow_t::~rainbow_t()
{
	delete this->starts;
	delete this->ends;
	delete this->percs;
	delete this->mt;
	this->mt = nullptr;

	this->chain_am = 0;
	this->chain_l = 0;
	this->part_am = 0;
	this->part = 0;
	this->chain_am_part = 0;
	this->temp_size = 0;
	this->charset = "";
	this->filename = "";
	this->min_l = 0;
	this->max_l = 0;
	this->gen_dur = 0;
	this->humanity = 0;
}

string rainbow_t::getFilename()
{
	return this->filename;
}

string rainbow_t::getMTFilename()
{
	if (mt)
		return this->mt->filename;
	else
		return "";
}

string rainbow_t::getCharset()
{
	return this->charset;
}

uint8_t rainbow_t::getMin_l()
{
	return this->min_l;
}

uint8_t rainbow_t::getMax_l()
{
	return this->max_l;
}

uint32_t rainbow_t::getChain_l()
{
	return this->chain_l;
}

uint64_t rainbow_t::getChain_am()
{
	return this->chain_am;
}

uint16_t rainbow_t::getPart_am()
{
	return this->part_am;
}

uint8_t rainbow_t::getPart()
{
	return this->part;
}

uint64_t rainbow_t::getChain_am_part()
{
	return this->chain_am_part;
}

uint64_t rainbow_t::getTemp_size()
{
	return this->temp_size;
}

string rainbow_t::getStart(uint64_t counter)
{
	return this->starts->at(counter);
}

string rainbow_t::getEnd(uint64_t counter)
{
	return this->ends->at(counter);
}

float rainbow_t::getProb()
{
	return this->percs->at(0);
}

float rainbow_t::getSearchTime()
{
	return this->percs->at(1);
}

uint_fast64_t rainbow_t::getGen_dur()
{
	return this->gen_dur;
}

markov3_t * rainbow_t::getMTptr()
{
	return this->mt;
}

uint8_t rainbow_t::getHumanity()
{
	return this->humanity;
}

void rainbow_t::setFilename(string filename)
{
	this->filename = filename;
}

void rainbow_t::setGenDur(uint64_t gen_dur)
{
	this->gen_dur += gen_dur;
}

void rainbow_t::pushBackEnd(string str)
{
	this->ends->at(this->temp_size) = str;
	this->temp_size++;
}

/*MAIN FUNCTIONS*/

/* private */

void rainbow_t::CheckSorted()
{
	uint64_t counter;
	for (counter = 1; counter < this->chain_am_part; counter++)
		if (this->ends->at(counter - 1) > this->ends->at(counter))
		{
			cout << "There is a mistake! " << counter << endl;
			break;
		}
}

void rainbow_t::SortRTByEnd(uint64_t left, uint64_t right)
{
	if ((right - left) < 2)
		return;

	uint64_t ll = left, rr = right - 1;
	string base = this->ends->at((left + right) / 2);

	while ((ll <= rr))
	{
		while ((this->ends->at(ll).compare(base) < 0))
			ll++;
		while ((this->ends->at(rr).compare(base) > 0))
			rr--;
		if (ll <= rr)
		{
			if (this->ends->at(ll).compare(this->ends->at(rr)) > 0)
			{
				std::swap(this->starts->at(ll), this->starts->at(rr));
				std::swap(this->ends->at(ll), this->ends->at(rr));
			}
			ll++;
			rr--;
		}
	}
	if (left < rr)
		this->SortRTByEnd(left, rr + 1);
	if (ll < right)
		this->SortRTByEnd(ll, right);
}

bool rainbow_t::GenUniqueChain(uint8_t limit, bool human)
{
	string str, start, hash_hex_str;
	if (this->temp_size)
		start = this->starts->at(this->temp_size - 1);
	else
		start = "";
	uint64_t sub_counter;
	uint32_t found_lim;
	bool found;

	/*CREATE UNIQUE START*/

	do
	{
		if (human)
			str = next_human_string(min_l, max_l, mt, start);
		else
			str = next_random_string(min_l, max_l, start);
		start = str;

		/*CREATE A CHAIN*/

		for (sub_counter = 0; sub_counter < this->chain_l; sub_counter++)
		{
			hash_hex_str = hash256_hex_string(str);
			if (human)
				str = reduction_human_cpu(hash_hex_str, (uint32_t)sub_counter, min_l, max_l, mt, this->part, this->chain_l, this->humanity);
			else
				str = reduction_rand_cpu(hash_hex_str, (uint32_t)sub_counter, min_l, max_l, this->part, this->chain_l);
		}

		/*CHECK IF THE END IS UNIQUE*/

		found_lim = 0;
		for (sub_counter = 0; (sub_counter < ends->size()) && (found_lim < limit); sub_counter++)
			if (!(str.compare(ends->at(sub_counter))))
			{
				found = true;
				found_lim++;
			}

		/*IF UNIQUE - ADD START AND END POINTS*/

		if (found_lim < limit)
		{
			starts->at(this->temp_size) = start;
			ends->at(this->temp_size) = str;
			this->temp_size++;
			found = false;
		}
	} while (found);

	return found;
}

string rainbow_t::ProbeForHash(string hash)
{
	uint32_t counter, increment;
	uint64_t counter_chains, st_a, en_a;
	bool flag = false;
	string str, hash_hex_str, end;
	float temp_perc;

	cout << "Hash Search Progress: ";
	for (counter = 0; (counter < this->chain_l) && (!flag); counter++)
	{
		hash_hex_str = hash;
		for (counter_chains = this->chain_l - counter - 1; counter_chains < this->chain_l; counter_chains++)
		{
			if (this->mt)
				str = reduction_human_cpu(hash_hex_str, (uint32_t)counter_chains, min_l, max_l, mt, this->part, this->chain_l, this->humanity);
			else
				str = reduction_rand_cpu(hash_hex_str, (uint32_t)counter_chains, min_l, max_l, this->part, this->chain_l);
			hash_hex_str = hash256_hex_string(str);
		}
		end = str; //сохраняем отдельно конечную строку для сформированного участка цепочки

		/*BINARY SEARCH*/

		st_a = 0;
		en_a = this->chain_am_part - 1;

		do
		{
			counter_chains = (st_a + en_a) / 2; //будет отвечать за середину между началом и концом
			if (end.compare(ends->at(counter_chains)) < 0)
			{
				if (counter_chains == 0)
					break; //конец цепочки в сравнении меньше наименьшего конца
				else
				{
					en_a = counter_chains - 1;
					continue;
				}
			}
			if (end.compare(ends->at(counter_chains)) > 0)
			{
				if (counter_chains == chain_am_part - 1)
					break; //конец цепочки в сравнении больше наибольшего конца
				else
				{
					st_a = counter_chains + 1;
					continue;
				}
			}
			if (end.compare(ends->at(counter_chains)) == 0)
			{
				/*HASH FOUND PROBABLY IN THIS TYPE OF LINE; NEED TO SEARCH STRING*/

				while ((counter_chains < chain_am_part) && (end.compare(ends->at(counter_chains)) == 0))
					counter_chains--;

				if (counter_chains > chain_am_part)
					counter_chains = 0;
				else
					counter_chains++;

				/*SEARCH IN LINES WITH SAME END*/

				while ((!flag) && (counter_chains < chain_am_part) && (end.compare(ends->at(counter_chains)) == 0))
				{
					str = starts->at(counter_chains);
					for (increment = 0; increment < (this->chain_l - counter - 1); increment++)
					{
						hash_hex_str = hash256_hex_string(str);
						if (this->mt)
							str = reduction_human_cpu(hash_hex_str, increment, min_l, max_l, mt, this->part, this->chain_l, this->humanity);
						else
							str = reduction_rand_cpu(hash_hex_str, increment, min_l, max_l, this->part, this->chain_l);
					}
					hash_hex_str = hash256_hex_string(str);
					if (hash == hash_hex_str)
					{
						flag = true;
						return str;
					}
					else
						counter_chains++;
				}
				break;
			}
		} while ((st_a <= en_a) && (!flag));

		temp_perc = (float)counter * 100 / this->chain_l;
		cout << (int)temp_perc / 10 << (int)temp_perc % 10 << "%" << "\b\b\b";
	}

	return "";
}

void rainbow_t::FindSearchPercPlusHashes(vector<string>* hashes)
{
	uint16_t found_hashes = 0;
	uint16_t i, size_h = (uint16_t)(hashes->size());;
	bool flag;
	float temp_perc, real_perc, time_sum = 0.0;
	uint8_t part_counter, iter;
	string result;
	vector<string> *plain_texts = new vector<string>();
	plain_texts->assign(hashes->size(), "");

	high_resolution_clock::time_point t1, t2;

	cout << "Trying to compute successful search percentage..." << endl;
	for (part_counter = this->part, iter = 0; iter < this->part_am; part_counter = (part_counter + 1) % (this->part_am), iter++)
	{
		if (part_counter != this->part)
			this->Read_Part(part_counter);

		cout << "Checking RT part " << (int)iter << "/" << this->part_am << "..." << endl;

		for (i = 0; i < size_h; i++) //find_hashes
		{
			temp_perc = (float)((i) * 100) / size_h;
			cout << "\t\t\t\t\t\t\t\t\r";
			cout << "Progress: " << temp_perc << "% (" << found_hashes << " found)\t";

			flag = false;

			result = "";

			if (plain_texts->at(i) == "")
			{
				t1 = high_resolution_clock::now();
				result = ProbeForHash(hashes->at(i));
				t2 = high_resolution_clock::now();
			}

			if (result != "")
			{
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
				time_sum += (float)duration;
				found_hashes++;
				plain_texts->at(i) = result;
			}
			/*
			else
			{
				if (!(this->multi))
					cout << "\"" << ends->at(i) << "\" (\"" << starts->at(i) << "\")" << endl;
				else
					cout << "\"" << parts_end->at(i / this->part_chain_am)->at(i % this->part_chain_am) << "\" (\"" << parts_start->at(i / this->part_chain_am)->at(i % this->part_chain_am) << "\")" << endl;
			}
			*/

			temp_perc = (float)((i + 1) * 100) / size_h;
			cout << "\rProgress: " << temp_perc << "% (" << found_hashes << " found)\t\t\r";
		}
		cout << endl;
	}

	/* CHECK HASHES, THEIR PLAINS AND OUTPUT THEM */

	cout << "Hashes ---> Plains" << endl << endl;
	for (i = 0; i < hashes->size(); i++)
	{
		if (plain_texts->at(i) != "")
		{
			cout << hashes->at(i) << endl;
			cout << "\t" << plain_texts->at(i) << endl << endl;
		}
	}

	cout << endl;
	real_perc = (float)(found_hashes * 100) / size_h;											//AVERAGE PERCENTAGE OF FOUND HASHES
	cout << "Real percentage of successful hash search: " << real_perc << "%" << endl;
	if (found_hashes)
		time_sum /= found_hashes;																	//AVERAGE SEARCH TIME
	cout << "Average successful search time: " << time_sum << " ms" << endl;

	this->percs->at(0) = real_perc;
	this->percs->at(1) = time_sum;
}

/* public */

/* CPU */

void rainbow_t::GenerateRT(bool human)
{
	if ((!(this->starts)) || (this->starts->size() == 0))
	{
		if ((this->starts) && (this->starts->size() == 0))
			delete this->starts;
		this->starts = CreateUniqueArray(this->chain_am, this->min_l, this->max_l, this->mt, false, "", human, this->humanity);
	}
	if (this->starts->at(0) == "")
	{
		delete this->starts;
		this->starts = CreateUniqueArray(this->chain_am, this->min_l, this->max_l, this->mt, false, "", human, this->humanity);
	}

	float temp_perc;
	high_resolution_clock::time_point t1, t2, t3, t4, t5, t6;
	uint64_t counter_chains, counter, dur, prev_dur = 0;
	string str, hash_hex_str;
	/*
	std::ofstream graph;
	graph.open("graph" + TimeStamp() + ".csv");
	uint_fast64_t step = chain_am / 30;

	graph << "METHOD:;NON-PERFECT" << endl << endl;
	graph << "OT MAX LEN:;" << (int)max_l << endl;
	graph << "OT MIN LEN:;" << (int)min_l << endl;
	graph << "CHAIN LENGTH:;" << chain_l << endl;
	graph << "CHAIN AM:;" << chain_am << endl << endl;
	graph << "T_MOMENT;CH-S GENERATED" << endl;
	*/
	cout << "Generating rainbow table..." << endl;

	t1 = high_resolution_clock::now();

	for (counter_chains = 0; counter_chains < this->chain_am; counter_chains++)
	{
		/*CHAIN COMPUTATION*/

		str = this->starts->at(counter_chains);
		for (counter = 0; counter < this->chain_l; counter++)
		{
			hash_hex_str = hash256_hex_string(str);
			if (human)
				str = reduction_human_cpu(hash_hex_str, (uint32_t)counter, this->min_l, this->max_l, this->mt, this->part, this->chain_l, this->humanity);
			else
				str = reduction_rand_cpu(hash_hex_str, (uint32_t)counter, this->min_l, this->max_l, this->part, this->chain_l);
		}
		this->ends->at(counter_chains) = str;

		t3 = high_resolution_clock::now();

		/*FOR GRAPH*/
		/*
		if (!((counter_chains + 1) % step))
		{
			t2 = high_resolution_clock::now();
			graph << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << ";" << (counter_chains + 1) << endl;
		}
		*/
		/*FOR PROGRESS VIEW*/

		dur = std::chrono::duration_cast<std::chrono::seconds>(t3 - t1).count();
		if (dur > prev_dur)
		{
			prev_dur = dur;
			temp_perc = (float)((counter_chains + 1) * 100 / chain_am);
			cout << "Progress: " << temp_perc << "%" << "\r";
		}
	}
	temp_perc = (float)((counter_chains) * 100 / chain_am);
	cout << "Progress: " << temp_perc << "%" << "\r";
	cout << endl;

	/*SORT CHAINS FOR BINARY SEARCH*/

	this->SortRTByEnd(0, this->chain_am);

	this->CheckSorted();

	t2 = high_resolution_clock::now();
	this->gen_dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

	cout << "Writing to file...";

	this->WriteRT_Part();

	cout << "done!" << endl;

	//graph.close();
}

void rainbow_t::GenerateUniqueRT(bool human)
{
	uint64_t counter, prev_cntr = 0;
	uint64_t dur, prev_dur = 0;
	uint64_t velocity;
	uint64_t time_left;
	bool found;
	high_resolution_clock::time_point t1, t2, t3;
	/*
	std::ofstream graph;
	graph.open("graph" + TimeStamp() + ".csv");
	uint_fast64_t step = chain_am / 30;

	graph << "METHOD:;PERFECT" << endl << endl;
	graph << "OT MAX LEN:;" << (int)max_l << endl;
	graph << "OT MIN LEN:;" << (int)min_l << endl;
	graph << "CHAIN LENGTH:;" << chain_l << endl;
	graph << "CHAIN AM:;" << chain_am << endl << endl;
	graph << "T_MOMENT;CH-S GENERATED" << endl;
	*/
	cout << "Generating rainbow table..." << endl;

	t1 = high_resolution_clock::now();

	for (counter = 0; counter < chain_am; counter++)
	{
		do
		{
			found = GenUniqueChain(1, human);

			/*FOR PROGRESS VIEW*/

			t3 = high_resolution_clock::now();
			dur = std::chrono::duration_cast<std::chrono::seconds>(t3 - t1).count();
			if (dur > prev_dur)
			{
				prev_dur = dur;
				velocity = ((counter + 1) - prev_cntr) + 1;
				prev_cntr = counter + 1;
				time_left = (this->chain_am - (counter + 1)) / velocity;
				cout << "Progress: " << (float)((counter + 1) * 100) / (this->chain_am) << "% (";
				if (velocity == 1)
					cout << "unknown)\t\t\r";
				else
					cout << (time_left / 60) << "m " << (time_left % 60) << "s)\t\t\r";
			}

		} while (found); //признак наличия данного конца

		/*FOR GRAPH*/
		/*
		if (!((counter + 1) % step))
		{
			t2 = high_resolution_clock::now();
			graph << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << ";" << (counter + 1) << endl;
		}
		*/
	}
	cout << "Progress: " << (float)((counter) * 100) / (this->chain_am) << "%\t\t\t\t\r";
	cout << endl;

	this->starts->shrink_to_fit();
	this->ends->shrink_to_fit();

	/*SORT CHAINS FOR BINARY SEARCH*/

	this->SortRTByEnd(0, this->chain_am_part);

	/*COUNT GENERATING TIME*/

	t2 = high_resolution_clock::now();
	this->gen_dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

	cout << "Writing to file...";

	this->WriteRT_Part();

	cout << "done!" << endl;

	//graph.close();
}

void rainbow_t::RTProbability(uint16_t amount)
{
	vector<string> *passwords;
	if (this->mt)
		passwords = CreateUniqueArray((uint64_t)amount, this->min_l, this->max_l, this->mt, true, "", true, this->humanity);
	else
		passwords = CreateUniqueArray((uint64_t)amount, this->min_l, this->max_l, this->mt, true, "", false, this->humanity);

	uint16_t i;
	vector<string> *hashes = new vector<string>();
	for (i = 0; i < amount; i++)
	{
		hashes->push_back(hash256_hex_string(passwords->at(i)));

		//hashes->push_back(hash256_hex_string(starts->at(i)));
	}

	hashes->shrink_to_fit();

	delete passwords;

	this->FindSearchPercPlusHashes(hashes);

	delete hashes;
}

void rainbow_t::WriteRT_Info()
{
	cout << "Writing info to file...\r";

	/*NEW FILE FORMAT

	//	info.txt

	markov table filename if exists
	humanity if exists
	charset
	min_length
	max_length
	chain_length
	chain_amount
	part_amount
	gen_dur
	search_perc
	search_time

	//	rt_(timestamp)_(part_n).txt

	chain_amount_part

	texts[chain_amount * 2]

	*/

	ofstream rainbow_file;
	rainbow_file.open("rtables\\" + this->filename + "\\info.txt");

	if (mt)
	{
		rainbow_file << "PLAIN MODE:\thuman" << endl;
		rainbow_file << "MT FILE:\t" << this->mt->filename << endl;
		rainbow_file << "HUMANITY:\t" << (int)(this->humanity) << endl;
	}
	else
		rainbow_file << "PLAIN MODE:\trandom" << endl;

	rainbow_file << "CHARSET:\t" << this->charset << endl;
	rainbow_file << "PLAIN TEXT MIN LENGTH:\t" << (int)(this->min_l) << endl;
	rainbow_file << "PLAIN TEXT MAX LENGTH:\t" << (int)(this->max_l) << endl;
	rainbow_file << "RT CHAIN LENGTH:\t" << this->chain_l << endl;
	rainbow_file << "RT CHAIN AMOUNT:\t" << this->chain_am << endl;
	rainbow_file << "RT PART AMOUNT:\t" << this->part_am << endl;
	rainbow_file << "RT GENERATING TIME (IN MS):\t" << this->gen_dur << endl;
	if (this->percs->at(0))
	{
		rainbow_file << "AVG SEARCH PROBABILITY (IN %):\t" << this->percs->at(0) << endl;
		rainbow_file << "AVG SEARCH TIME (IN MS):\t" << this->percs->at(1) << endl;
	}

	rainbow_file.close();

	cout << "RT info is successfullly written to folder \"" << this->filename << "\"." << endl;
}

void rainbow_t::WriteRT_Part()
{
	ofstream rainbow_file;
	rainbow_file.open("rtables\\" + this->filename + "\\part_" + std::to_string(this->part) + ".txt");

	rainbow_file << this->chain_am_part << endl << endl;

	uint64_t counter;
	for (counter = 0; counter < this->chain_am_part; counter++)
	{
		rainbow_file << this->starts->at(counter) << endl;
		rainbow_file << this->ends->at(counter) << endl;
	}

	rainbow_file.close();

	cout << "Part " << (int)(this->part) + 1 << /*" out of " << (int)(this->part_am) <<*/ " of this RT is successfullly written to folder \"" << this->filename << "\"." << endl;
}

void rainbow_t::Read_Part(uint8_t part_num)
{
	delete this->starts;
	delete this->ends;

	this->starts = new vector<string>();
	this->ends = new vector<string>();
	this->part = part_num;

	ifstream rainbow_file;
	rainbow_file.open("rtables\\" + this->filename + "\\part_" + std::to_string(part_num) + ".txt");

	string str;
	uint64_t counter;

	getline(rainbow_file, str);
	this->chain_am_part = std::stoull(str, nullptr, 0);

	this->starts->assign(this->chain_am_part, "");
	this->ends->assign(this->chain_am_part, "");

	getline(rainbow_file, str);

	for (counter = 0; counter < this->chain_am_part; counter++)
	{
		getline(rainbow_file, str);
		this->starts->at(counter) = str;
		getline(rainbow_file, str);
		this->ends->at(counter) = str;
	}

	rainbow_file.close();

	this->temp_size = this->chain_am_part;
}

/* GPU */

void rainbow_t::GenerateRT_CUDA(int blocksNum, int threadsNum, bool human)
{
	string last = "";

	for (uint16_t cntr = 0; cntr < this->part_am; cntr++)
	{
		delete this->starts;

		cout << "Generating part " << (int)(this->part) + 1 << "...\t\t\t\t\t\r";

		this->starts = CreateUniqueArray(this->chain_am_part, this->min_l, this->max_l, this->mt, false, last, human, this->humanity);
		this->ends->resize(this->chain_am_part);
		last = this->starts->back();

		Generate_RT_CUDA(this, blocksNum, threadsNum, human);

		//cout << "Sorting...";
		this->SortRTByEnd(0, this->chain_am_part);
		//cout << "done!" << endl;
		/*
		cout << "Checking sorted...";
		this->CheckSorted();

		cout << "done!" << endl;
		*/
		cout << "Writing to file...\t\t\t\t\r";
		this->WriteRT_Part();

		if (this->part_am > 1)
		{
			this->part++;
			if ((this->chain_am - (this->part * this->chain_am_part)) < this->chain_am_part)
				this->chain_am_part = this->chain_am - (this->part * this->chain_am_part);
			this->temp_size = 0;
			this->ends->clear();
			this->ends->reserve(this->chain_am_part);
		}
	}
}

void rainbow_t::GenerateUniqueRT_CUDA(int blocksNum, int threadsNum, bool human)
{
	string last = "";
	string temp_end;
	vector<string> *addition;
	//vector<string> *unique_ends = new vector<string>();
	//unique_ends->assign(this->chain_am, "");
	uint64_t counter, real_cntr;
	//uint64_t big_size = 256 * 1024 * 1024 / (1.5 * this->min_l + 0.5 * this->max_l + 3);
	uint64_t big_size = 500000;
	uint64_t tmp_size;
	uint64_t prev_dur = 0;

	this->chain_am_part = 0;
	this->part_am = (uint16_t)(this->chain_am / big_size) + 1;

	for (uint16_t cntr = 0; cntr < this->part_am; cntr++)
	{
		if (cntr == (this->part_am - 1))
			tmp_size = this->chain_am - cntr * big_size;
		else
			tmp_size = big_size;

		this->temp_size = 0;
		this->chain_am_part = 0;

		this->starts->clear();
		this->starts->reserve(tmp_size);
		this->ends->clear();
		this->ends->reserve(tmp_size);

		cout << "Generating part " << (int)(this->part) + 1 << "...\t\t\t\t\r";

		while (this->temp_size < tmp_size)
		{
			addition = CreateUniqueArray(blocksNum * threadsNum, this->min_l, this->max_l, this->mt, false, last, human, this->humanity);
			last = addition->back();
			this->chain_am_part += blocksNum * threadsNum;
			this->starts->resize(this->chain_am_part);
			this->ends->resize(this->chain_am_part);
			for (counter = this->temp_size; counter < this->chain_am_part; counter++)
				this->starts->at(counter) = addition->at(counter - this->temp_size);
			delete addition;

			Generate_RT_CUDA(this, blocksNum, threadsNum, human);

			//cout << "Sorting...";
			this->SortRTByEnd(0, this->temp_size);
			//cout << "done!" << endl;
			/*
			cout << "Checking sorted...";
			this->CheckSorted();
			cout << "done!" << endl;
			*/
			/* delete duplicates inside this part */

			counter = 0;
			real_cntr = 0;
			while (counter < this->chain_am_part)
			{
				temp_end = this->ends->at(counter);

				/* move distinct start and end up */

				this->starts->at(real_cntr) = this->starts->at(counter);
				//this->starts->at(counter) = "";
				this->ends->at(real_cntr) = this->ends->at(counter);
				//this->ends->at(counter) = "";

				counter++;
				real_cntr++;

				/* delete duplicates inside */

				while ((counter < this->chain_am_part) && (this->ends->at(counter) == temp_end))
				{
					this->starts->at(counter) = "";
					this->ends->at(counter) = "";
					this->temp_size--;
					counter++;
				}
			}

			this->chain_am_part = this->temp_size;

			if ((prev_dur / 1000) < (this->gen_dur / 1000))
			{
				prev_dur = this->gen_dur;
				cout << "Time spent: " << prev_dur / 3600000 << "h " << (prev_dur % 3600000) / 60000 << "m " << (prev_dur % 60000) / 1000 << "s\t" << cntr * big_size + this->chain_am_part << "\t\t\t\r";
			}
			/*
			if (real_cntr == this->temp_size)
				cout << "This is OK!" << endl;
			*/
			/*
			if (cntr)
			{
				// delete duplicates from unique

				for (counter = 0; counter < this->temp_size; counter++)
				{
					uint64_t st_a = 0;
					uint64_t en_a = (cntr * big_size) - 1;
					erased_flag = false;

					do
					{
						uint64_t counter_chains = (st_a + en_a) / 2; //будет отвечать за середину между началом и концом
						if (unique_ends->at(counter_chains).compare(this->ends->at(counter)) > 0)
						{
							if (counter_chains == 0)
								break; //конец цепочки в сравнении меньше наименьшего конца
							else
							{
								en_a = counter_chains - 1;
								continue;
							}
						}
						if (unique_ends->at(counter_chains).compare(this->ends->at(counter)) < 0)
						{
							if (counter_chains == chain_am - 1)
								break; //конец цепочки в сравнении больше наибольшего конца
							else
							{
								st_a = counter_chains + 1;
								continue;
							}
						}
						if (unique_ends->at(counter_chains).compare(this->ends->at(counter)) == 0)
						{
							this->starts->erase(this->starts->begin() + counter);
							this->starts->push_back("");
							this->ends->erase(this->ends->begin() + counter);
							this->ends->push_back("");
							this->temp_size--;
							erased_flag = true;
							counter--;
						}
					} while ((st_a <= en_a) && (!erased_flag));

					if (!(counter % 10000))
					cout << counter << "/" << this->temp_size << " checked\r";
				}
			}
			cout << endl;
			*/
		}

		this->chain_am_part = tmp_size;

		/* rewrite unique */

		/*

		cout << "Rewriting unique array..." << endl;

		uint64_t added = 0;
		uint64_t un_cntr = 0;
		for (counter = 0; counter < this->temp_size; counter++)
		{
			while ((un_cntr < (cntr * big_size + added)) && (unique_ends->at(un_cntr).compare(this->ends->at(counter)) < 0))
				un_cntr++;
			if (cntr)
			{
				unique_ends->insert(unique_ends->begin() + un_cntr, this->ends->at(counter));
				if (!(counter % 100))
					cout << counter << "/" << this->temp_size << "\r";
			}

			else
			{
				unique_ends->at(un_cntr) = this->ends->at(counter);
				if (!(counter % 10000))
					cout << counter << "/" << this->temp_size << "\r";
			}

			added++;
			un_cntr++;

		}

		cout << "Rewriting unique array done!\t\t" << endl;

		*/

		cout << "Writing to file...\t\t\t\r";
		this->WriteRT_Part();

		if (this->part_am > 1)
			this->part++;
	}

	//delete unique_ends;
}

void rainbow_t::GenerateRTbyTime_CUDA(int blocksNum, int threadsNum, bool human, uint64_t time)
{
	string last = "";
	high_resolution_clock::time_point t1, t2;
	uint64_t dur = 0, prev_dur = 0;

	uint64_t ch_am = (uint64_t)((256 * 1024 * 1024) / (1.5 * this->min_l + 0.5 * this->max_l + 3));
	if (this->starts)
		delete this->starts;

	t1 = high_resolution_clock::now();

	while (dur < time)
	{
		if (dur == 0)
		{
			cout << "Generating part " << (int)(this->part) + 1 << "...\t\t\t\t\t\r";
			this->starts = CreateUniqueArray(ch_am, this->min_l, this->max_l, this->mt, false, last, human, this->humanity);
			this->ends->reserve(ch_am);
		}

		/* check if there's need to make one more RT part */

		if (this->temp_size == ch_am)
		{
			last = this->starts->back();

			/* sorting */

			//cout << "Sorting...\t\t";
			this->SortRTByEnd(0, this->chain_am_part);
			//cout << "\b\bdone!\t\t\r";

			/* wriet part to file */

			cout << "Writing to file...\t\t\t\t\t\r";
			this->WriteRT_Part();
			//cout << "\b\bdone!\t\t\r";

			/* make new part */

			this->part++;
			this->part_am++;
			this->temp_size = 0;
			this->chain_am_part = 0;
			delete this->starts;
			this->starts = CreateUniqueArray(ch_am, this->min_l, this->max_l, this->mt, false, last, human, this->humanity);
			this->ends->clear();
			this->ends->reserve(ch_am);

			cout << "Generating part " << (int)(this->part) + 1 << "...\t\t\t\r";
		}

		/* generate new addition */

		if ((this->chain_am_part + blocksNum * threadsNum) <= ch_am)
			this->chain_am_part += blocksNum * threadsNum;
		else
			this->chain_am_part = ch_am;
		this->ends->resize(this->chain_am_part);

		Generate_RT_CUDA(this, blocksNum, threadsNum, human);
		if (this->chain_am_part < ch_am)
			this->chain_am += blocksNum * threadsNum;
		else
			this->chain_am = ch_am * this->part_am;
		

		t2 = high_resolution_clock::now();
		dur = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

		if (prev_dur < dur)
		{
			prev_dur = dur;
			cout << "Time remaining: " << (time - dur) / 3600 << "h " << ((time - dur) % 3600) / 60 << "m " << ((time - dur) % 60) << "s\t" << this->chain_am << "\t\t\t\r";
		}

		if (dur >= time)
		{
			/* sorting */

			//cout << "Sorting...\t\t";
			this->SortRTByEnd(0, this->chain_am_part);
			//cout << "\b\bdone!\t\t\r";

			/* write last part */

			cout << "Writing to file...\t\t\t\t\t\r";
			this->WriteRT_Part();
			//cout << "\b\bdone!\t\t" << endl;
		}

		/*
		cout << "Checking sorted...";
		this->CheckSorted();
		cout << "done!" << endl;
		*/
	}
}

void rainbow_t::GenerateUniqueRTbyTime_CUDA(int blocksNum, int threadsNum, bool human, uint64_t time)
{
	string last = "";
	string temp_end;
	vector<string> *addition;
	uint64_t counter, real_cntr;
	//uint64_t big_size = 256 * 1024 * 1024 / (uint64_t)(1.5 * this->min_l + 0.5 * this->max_l + 3);
	uint64_t big_size = 500000;
	high_resolution_clock::time_point t1, t2;
	uint64_t dur = 0, prev_dur = 0, prev_gen_dur = 0; //in ms
	/*
	ofstream graph;
	graph.open("rtables\\" + this->filename + "\\gen_graph.csv");
	graph << "Time:;Gen_dur:;Chain AM:;" << endl << endl;
	*/
	t1 = high_resolution_clock::now();

	while ((dur / 1000) < time)
	{
		if (dur == 0)
			cout << "Generating part " << (int)(this->part) + 1 << "...\t\t\t\t\r";

		this->starts->reserve(big_size);
		this->ends->reserve(big_size);

		while (this->temp_size < big_size)
		{
			addition = CreateUniqueArray(blocksNum * threadsNum, this->min_l, this->max_l, this->mt, false, last, human, this->humanity);
			last = addition->back();
			this->chain_am_part += blocksNum * threadsNum;
			this->starts->resize(this->chain_am_part);
			this->ends->resize(this->chain_am_part);
			for (counter = this->temp_size; counter < this->chain_am_part; counter++)
				this->starts->at(counter) = addition->at(counter - this->temp_size);

			delete addition;

			Generate_RT_CUDA(this, blocksNum, threadsNum, human);

			//cout << "Sorting...\t\t\t\t";
			this->SortRTByEnd(0, this->chain_am_part);
			//cout << "\b\b\b\bdone!\t\t\r";
			/*
			cout << "Checking sorted...";
			this->CheckSorted();
			cout << "done!" << endl;
			*/
			/* delete duplicates inside this part */

			//cout << "Deleting dups...\t\t\t\t";
			counter = 0;
			real_cntr = 0;
			while ((counter < this->chain_am_part))
			{
				temp_end = this->ends->at(counter);

				/* move distinct start and end up */

				this->starts->at(real_cntr) = this->starts->at(counter);
				this->ends->at(real_cntr) = this->ends->at(counter);

				counter++;
				real_cntr++;

				/* delete duplicates inside */

				while ((counter < this->chain_am_part) && (this->ends->at(counter) == temp_end))
				{
					this->starts->at(counter) = "";
					this->ends->at(counter) = "";
					this->temp_size--;
					counter++;
				}
			}
			//cout << "\b\b\b\bdone!\t\t\r";
			/*
			if (real_cntr == this->temp_size)
				cout << "This is OK!\t\t\t\t\r";
			*/
			this->chain_am_part = this->temp_size;

			t2 = high_resolution_clock::now();
			dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

			if ((prev_dur / 60000) < (dur / 60000))
			{
				prev_dur = dur;
				cout << "Time remaining: " << (time - (dur / 1000)) / 3600 << "h " << ((time - (dur / 1000)) % 3600) / 60 << "m " << (time - (dur / 1000)) % 60 << "s\t" << this->chain_am + this->chain_am_part << "\t\t\t\r";
				//graph << dur << ";" << this->gen_dur - prev_gen_dur << ";" << this->chain_am_part << endl;
			}

			if ((dur / 1000) >= time)
				break;
		}

		//cout << "Writing to file...\t\t\t\t";
		this->chain_am += this->temp_size;
		this->WriteRT_Part();

		//graph << endl;
		prev_gen_dur = this->gen_dur;
		//cout << "\b\b\b\bdone!\t\t\r";

		/* make new part if possible*/

		if ((dur / 1000) < time)
		{
			this->part++;
			this->part_am++;
			this->temp_size = 0;
			this->chain_am_part = 0;
			delete this->starts;
			delete this->ends;
			this->starts = new vector<string>();
			this->ends = new vector<string>();

			cout << "Generating part " << (int)(this->part) + 1 << "...\t\t\t\r";
		}
	}

	//graph.close();
}

void rainbow_t::SearchProb_CUDA(uint16_t amount, int blocksNum, int threadsNum)
{
	vector<string> *hashes = new vector<string>();
	vector<string> *plains = new vector<string>();

	/* check if it needs to read hash file */

	uint16_t tmp_amount;

	if (amount == 0)
	{
		tmp_amount = 400;

		hashes->assign(tmp_amount, "");
		plains->assign(tmp_amount, "");

		/* read hash file */

		ifstream file;
		file.open("hashes.txt");
		string str = "";
		uint16_t counter = 0;

		while (getline(file, str) && (str != ""))
		{
			hashes->at(counter) = str;
			counter++;
		}

		file.close();

		/* read plain file */

		file.open("plains.txt");
		str = "";
		counter = 0;

		while (getline(file, str) && (str != ""))
		{
			plains->at(counter) = str;
			counter++;
		}

		file.close();
	}
	else
	{
		/* create new random hash list with plain list*/

		tmp_amount = amount;

		if (this->mt)
			plains = CreateUniqueArray(tmp_amount, this->min_l, this->max_l, this->mt, true, "", true, this->humanity);
		else
			plains = CreateUniqueArray(tmp_amount, this->min_l, this->max_l, this->mt, true, "", false, this->humanity);

		hashes->assign(tmp_amount, "");

		uint16_t counter = 0;
		while (counter < tmp_amount)
		{
			hashes->at(counter) = picosha2::hash256_hex_string(plains->at(counter));
			counter++;
		}
	}

	/* transform to uint32_t array */

	uint32_t *hash_array = (uint32_t *)calloc(tmp_amount * 8, sizeof(uint32_t));
	for (uint16_t counter = 0; counter < (tmp_amount * 8); counter++)
		hash_array[counter] = std::stoul(hashes->at(counter / 8).substr((counter % 8) * 8, 8), nullptr, 16);

	/* run kernel (get vector)*/

	vector<string> *result = Search_Prob_CUDA(this, hash_array, tmp_amount, blocksNum, threadsNum);

	/* run end search */

	high_resolution_clock::time_point t1, t2;
	uint16_t found_hashes = 0;
	float temp_perc;
	float time_sum = 0;
	bool flag;
	string res_str;
	vector<string> *found_plains = new vector<string>();
	found_plains->assign(tmp_amount, "");

	cout << "Trying to compute successful search percentage..." << endl;
	for (uint8_t part_counter = this->part, iter = 0; iter < this->part_am; part_counter = (part_counter + 1) % (this->part_am), iter++)
	{
		if (part_counter != this->part)
			this->Read_Part(part_counter);

		cout << "Checking RT part " << (int)iter << "/" << this->part_am << "..." << endl;

		for (uint32_t i = 0; i < tmp_amount * this->chain_l; i++) //find_hashes
		{
			if ((!(i % this->chain_l)) || (!(i % 100)))
			{
				temp_perc = (float)((i) * 100) / (tmp_amount * this->chain_l);
				cout << "\t\t\t\t\t\t\t\t\r";
				cout << "Progress: " << temp_perc << "% (" << found_hashes << " found)\r";
			}

			flag = false;

			res_str = "";
			string end = result->at(i);

			if (found_plains->at(i / this->chain_l) == "")
			{
				t1 = high_resolution_clock::now();

				/*BINARY SEARCH*/

				uint64_t st_a = 0;
				uint64_t en_a = this->chain_am_part - 1;

				do
				{
					uint64_t counter_chains = (st_a + en_a) / 2; //будет отвечать за середину между началом и концом
					if (end.compare(ends->at(counter_chains)) < 0)
					{
						if (counter_chains == 0)
							break; //конец цепочки в сравнении меньше наименьшего конца
						else
						{
							en_a = counter_chains - 1;
							continue;
						}
					}
					if (end.compare(ends->at(counter_chains)) > 0)
					{
						if (counter_chains == chain_am_part - 1)
							break; //конец цепочки в сравнении больше наибольшего конца
						else
						{
							st_a = counter_chains + 1;
							continue;
						}
					}
					if (end.compare(ends->at(counter_chains)) == 0)
					{
						/*HASH FOUND PROBABLY IN THIS TYPE OF LINE; NEED TO SEARCH STRING*/

						while ((counter_chains < chain_am_part) && (end.compare(ends->at(counter_chains)) == 0))
							counter_chains--;

						if (counter_chains > chain_am_part)
							counter_chains = 0;
						else
							counter_chains++;

						/*SEARCH IN LINES WITH SAME END*/

						while ((!flag) && (counter_chains < chain_am_part) && (end.compare(ends->at(counter_chains)) == 0))
						{
							string str = starts->at(counter_chains);
							string hash_hex_str;
							uint32_t counter = i % this->chain_l;
							for (uint32_t increment = 0; increment < (this->chain_l - counter - 1); increment++)
							{
								hash_hex_str = hash256_hex_string(str);
								if (this->mt)
									str = reduction_human_cpu(hash_hex_str, increment, min_l, max_l, mt, this->part, this->chain_l, this->humanity);
								else
									str = reduction_rand_cpu(hash_hex_str, increment, min_l, max_l, this->part, this->chain_l);
							}
							hash_hex_str = hash256_hex_string(str);
							if (hashes->at(i / this->chain_l) == hash_hex_str)
							{
								flag = true;
								res_str = str;
							}
							else
								counter_chains++;
						}
						break;
					}
				} while ((st_a <= en_a) && (!flag));

				t2 = high_resolution_clock::now();
			}

			if (res_str != "")
			{
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
				time_sum += (float)duration;
				found_hashes++;
				if (plains->at(i / this->chain_l) == res_str)
				{
					found_plains->at(i / this->chain_l) = res_str;
					i = (i / this->chain_l + 1) * this->chain_l - 1;
				}
				else
				{
					cout << "Error: " << res_str << " != " << found_plains->at(i / this->chain_l) << endl;
					cout << picosha2::hash256_hex_string(res_str) << endl << "\t\t!=" << endl;
					cout << hashes->at(i / this->chain_l) << endl;

					delete plains;
					delete hashes;
					delete found_plains;
					free(hash_array);

					return;
				}
			}

			if ((!(i % this->chain_l + 1)) || (!(i % 100)))
			{
				temp_perc = (float)((i + 1) * 100) / (tmp_amount * this->chain_l);
				cout << "Progress: " << temp_perc << "% (" << found_hashes << " found)\t\t\r";
			}
		}
		cout << endl;
	}

	/* count stats and write */

	/* CHECK HASHES, THEIR PLAINS AND OUTPUT THEM */

	cout << "Hashes ---> Plains" << endl << endl;
	for (uint16_t i = 0; i < tmp_amount; i++)
		if (found_plains->at(i) != "")
			cout << hashes->at(i) << " ---> " << found_plains->at(i) << endl;

	cout << endl;
	float real_perc = (float)(found_hashes * 100) / tmp_amount;											//AVERAGE PERCENTAGE OF FOUND HASHES
	cout << "Real percentage of successful hash search: " << real_perc << "%" << endl;
	if (found_hashes)
		time_sum /= found_hashes;																	//AVERAGE SEARCH TIME
	cout << "Average successful search time: " << time_sum << " ms" << endl;

	this->percs->at(0) = real_perc;
	this->percs->at(1) = time_sum;

	if (amount == 0)
	{
		ofstream special_stats;
		special_stats.open("rtables\\" + this->filename + "\\special_stats.txt", std::ios::app);

		int block_found_hashes = 0;
		for (int i = 0; i < 50; i++)
			if (found_plains->at(i) != "")
				block_found_hashes++;

		temp_perc = (float)block_found_hashes * 2;

		special_stats << "SPLASHDATA BLOCK:\t" << temp_perc << "%\n";

		block_found_hashes = 0;
		for (int i = 50; i < 200; i++)
			if (found_plains->at(i) != "")
				block_found_hashes++;

		temp_perc = (float)block_found_hashes * 2 / 3;
		special_stats << "MT GENERATED BLOCK:\t" << temp_perc << "%\n";

		block_found_hashes = 0;
		for (int i = 200; i < 400; i++)
			if (found_plains->at(i) != "")
				block_found_hashes++;

		temp_perc = (float)block_found_hashes / 4;
		special_stats << "DICTIONARY BLOCK:\t" << temp_perc << "%\n";
		special_stats << endl;
		special_stats << "\tRECOVERED:" << endl << endl;

		for (int i = 0; i < 400; i++)
			if (found_plains->at(i) != "")
				special_stats << hashes->at(i) << "  --->  " << found_plains->at(i) << endl;

		special_stats << endl << endl << endl;
		special_stats.close();
	}
	else
	{
		ofstream recovered_pass;
		recovered_pass.open("rtables\\" + this->filename + "\\recovered_passwords.txt", std::ios::app);

		for (int i = 0; i < tmp_amount; i++)
			if (found_plains->at(i) != "")
				recovered_pass << hashes->at(i) << "  --->  " << found_plains->at(i) << endl;

		recovered_pass << endl << endl << endl;
		recovered_pass.close();
	}

	delete plains;
	delete hashes;
	delete found_plains;
	free(hash_array);
}

void rainbow_t::SearchProbExpress_CUDA(uint16_t amount, int blocksNum, int threadsNum)
{
	vector<string> *hashes = new vector<string>();
	vector<string> *plains = new vector<string>();
	uint64_t big_amount = 10000;
	uint64_t skip = 0;
	uint64_t found_hashes = 0;
	uint64_t skip_found = 0;

	/* check if it needs to read hash file */
	/*
	if (amount == 0)
	{

		amount = 1000;

		hashes->assign(amount, "");
		plains->assign(amount, "");

		// read additional file with found_hashes and skip amount

		string str = "";
		ifstream info;
		info.open("rtables\\" + this->filename + "\\spec_info.txt");
		getline(info, str);
		skip_found = std::stoull(str, nullptr, 0);
		getline(info, str);
		skip = std::stoull(str, nullptr, 0);
		info.close();

		// read hash file

		ifstream file;
		file.open("hashes.txt");
		str = "";
		//big_amount = 0;

		hashes->assign(big_amount, "");

		// skip
		/*
		uint64_t counter = 0;
		while (counter < skip)
		{
			getline(file, str);
			counter++;
		}

		// read

		uint64_t counter = 0;
		skip = 0;
		while ((counter < big_amount) && getline(file, str) && (str != ""))
		{
			if (!(skip % 270))
			{
				hashes->at(counter) = str;
				counter++;
			}
			skip++;
		}

		file.close();
		//hashes->shrink_to_fit();

		// read plain file

		file.open("plains.txt");
		plains->assign(big_amount, "");

		/* skip */
		/*
		counter = 0;
		while (counter < skip)
		{
			getline(file, str);
			counter++;
		}
		*/
		/* read

		counter = 0;
		skip = 0;
		while ((counter < big_amount) && getline(file, str) && (str != ""))
		{
			if (!(skip % 270))
			{
				plains->at(counter) = str;
				counter++;
			}
			skip++;
		}
		skip = 0;

		file.close();
	}
	*/

	uint16_t tmp_amount;

	if (amount == 0)
	{
		tmp_amount = 400;

		hashes->assign(tmp_amount, "");
		plains->assign(tmp_amount, "");
		ifstream file;

		// read hash file 

		file.open("hashes.txt");
		string str = "";
		uint64_t counter = 0;
		while (getline(file, str) && (str != ""))
		{
			hashes->at(counter) = str;
			counter++;
		}

		file.close();

		// read plain file

		file.open("plains.txt");
		str = "";
		counter = 0;
		while (getline(file, str) && (str != ""))
		{
			plains->at(counter) = str;
			counter++;
		}

		file.close();
	}
	else
	{
		/* create new random hash list with plain list*/

		tmp_amount = amount;

		if (this->mt)
			plains = CreateUniqueArray(tmp_amount, this->min_l, this->max_l, this->mt, true, "", true, this->humanity);
		else
			plains = CreateUniqueArray(tmp_amount, this->min_l, this->max_l, this->mt, true, "", false, this->humanity);

		hashes->assign(tmp_amount, "");

		uint16_t counter = 0;
		while (counter < tmp_amount)
		{
			hashes->at(counter) = picosha2::hash256_hex_string(plains->at(counter));
			counter++;
		}

		big_amount = tmp_amount;
	}

	/* transform to uint32_t array */

	uint32_t *hash_array = (uint32_t *)calloc(tmp_amount * 8, sizeof(uint32_t));
	for (uint64_t counter = 0; counter < (tmp_amount * 8); counter++)
		hash_array[counter] = std::stoul(hashes->at(counter / 8).substr((counter % 8) * 8, 8), nullptr, 16);

	/* run kernel (get vector)*/

	vector<string> *result;

	/* run end search */

	float time_sum = 0;
	string res_str;
	vector<string> *found_plains = new vector<string>();
	found_plains->assign(tmp_amount, "");

	uint64_t new_big_amount = tmp_amount;
	uint16_t loop_size = 10000;

	cout << "Trying to compute successful search percentage..." << endl;
	for (uint8_t part_counter = this->part, iter = 0; iter < this->part_am; part_counter = (part_counter + 1) % (this->part_am), iter++)
	{
		uint16_t loop_size_tmp = loop_size;
		uint16_t loops = (uint16_t)(new_big_amount / loop_size);
		if (new_big_amount % loop_size)
			loops++;

		if (part_counter != this->part)
			this->Read_Part(part_counter);

		cout << "Checking RT part " << (int)iter + 1 << "/" << this->part_am << "..." << endl;

		for (uint16_t loop_counter = 0; loop_counter < loops; loop_counter++)
		{
			uint32_t *hash_array_temp = &(hash_array[(uint64_t)loop_counter * (uint64_t)loop_size_tmp * 8]);

			if ((loop_counter == (loops - 1)) && (new_big_amount % loop_size))
				loop_size_tmp = new_big_amount % loop_size;

			result = Search_Prob_Express_CUDA(this, hash_array_temp, loop_size_tmp, blocksNum, threadsNum);

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
							found_hashes++;
						}
					}
				}
			}

			cout << "Found: " << found_hashes + skip_found << "/" << tmp_amount + skip << "\t\t" << endl;

			delete result;
		}

		/* renew hash array to calculate less */

		free(hash_array);

		new_big_amount = tmp_amount - found_hashes;
		uint64_t new_b_a_cntr = 0;
		hash_array = (uint32_t *)calloc(new_big_amount * 8, sizeof(uint32_t));

		for (uint64_t counter = 0; counter < tmp_amount; counter++)
		{
			if (found_plains->at(counter) == "")
			{
				for (uint16_t sub_cntr = 0; sub_cntr < 8; sub_cntr++)
					hash_array[new_b_a_cntr * 8 + sub_cntr] = std::stoul(hashes->at(counter).substr((sub_cntr) * 8, 8), nullptr, 16);
				new_b_a_cntr++;
			}
		}
	}

	/* count stats and write */

	/* CHECK HASHES, THEIR PLAINS AND OUTPUT THEM */
	/*
	cout << "Hashes ---> Plains" << endl << endl;
	for (uint64_t i = 0; i < big_amount; i++)
	{
		if (found_plains->at(i) != "")
		{
			cout << hashes->at(i) << endl;
			cout << "\t" << found_plains->at(i) << endl << endl;
		}
	}
	cout << endl;
	*/

	float real_perc = (float)((found_hashes + skip_found) * 100) / (tmp_amount + skip);											//AVERAGE PERCENTAGE OF FOUND HASHES
	cout << "Real percentage of successful hash search: " << real_perc << "%" << endl;
	if (found_hashes)
		time_sum /= found_hashes;																	//AVERAGE SEARCH TIME
	cout << "Average successful search time: " << time_sum << " ms" << endl;

	this->percs->at(0) = real_perc;
	this->percs->at(1) = time_sum;

	if (amount == 0)
	{
		ofstream spec_info_out;
		spec_info_out.open("rtables\\" + this->filename + "\\spec_info.txt", std::ios::trunc);

		spec_info_out << skip_found + found_hashes << endl;
		spec_info_out << skip + tmp_amount << endl;

		spec_info_out.close();

		ofstream special_stats;
		special_stats.open("rtables\\" + this->filename + "\\special_stats.txt", std::ios::app);

		int block_found_hashes = 0;
		for (int i = 0; i < 50; i++)
			if (found_plains->at(i) != "")
				block_found_hashes++;

		float temp_perc = (float)block_found_hashes * 2;
		special_stats << "SPLASHDATA BLOCK:\t" << temp_perc << "%\n";

		block_found_hashes = 0;
		for (int i = 50; i < 300; i++)
			if (found_plains->at(i) != "")
				block_found_hashes++;

		temp_perc = (float)block_found_hashes * 2 / 5;
		special_stats << "MT GENERATED BLOCK:\t" << temp_perc << "%\n";

		block_found_hashes = 0;
		for (int i = 300; i < 1000; i++)
			if (found_plains->at(i) != "")
				block_found_hashes++;

		temp_perc = (float)block_found_hashes / 7;
		special_stats << "DICTIONARY BLOCK:\t" << temp_perc << "%\n";
		special_stats << endl;
		special_stats << "\tRECOVERED:" << endl << endl;

		for (int i = 0; i < big_amount; i++)
		{
			if ((i == 50) || (i == 300))
				special_stats << endl;

			if (found_plains->at(i) != "")
				special_stats << hashes->at(i) << "  --->  " << found_plains->at(i) << endl;
		}

		special_stats << endl << "====================================================" << endl << endl << endl;
		special_stats.close();
	}
	else
	{
		ofstream spec_info_out;
		spec_info_out.open("rtables\\" + this->filename + "\\spec_info.txt", std::ios::trunc);

		spec_info_out << skip_found + found_hashes << endl;
		spec_info_out << skip + tmp_amount << endl;

		spec_info_out.close();

		ofstream recovered_pass;
		recovered_pass.open("rtables\\" + this->filename + "\\recovered_passwords.txt", std::ios::app);

		for (int i = 0; i < tmp_amount; i++)
			if (found_plains->at(i) != "")
				recovered_pass << hashes->at(i) << "  --->  " << found_plains->at(i) << endl;

		recovered_pass << endl << endl << endl;
		recovered_pass.close();
	}

	delete plains;
	delete hashes;
	delete found_plains;
	free(hash_array);
}

