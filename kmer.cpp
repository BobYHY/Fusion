#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "omp.h"
#include <ctime>
#include <cmath>

using namespace std;

int pow(int a,int b){
	int ans=1;
	for(int i=0;i<b;i++)
	{
		ans=ans*a; 
	}
	return ans;
}

vector<long double> Kmer(const string& sequence, int K) {
    int m = pow(4, K);
    vector<long double> na_vect(3*m, 0.0);
    vector<long double> pos_sum(m, 0.0);
    int n = sequence.length() - (K - 1);
    
    vector<int> index_map(128, -1);
    index_map['a'] = index_map['A'] = 0;
    index_map['c'] = index_map['C'] = 1;
    index_map['g'] = index_map['G'] = 2;
    index_map['t'] = index_map['T'] = 3;
    
    for (int i = 0; i < n; i++) {
        bool flag = true;
        int tem = index_map[sequence[i]];
        if (tem==-1) continue;
        for (int l = 1; l < K; l++) {
            if (index_map[sequence[i + l]] == -1) {
                flag = false;
                break;
            }
           tem = 4 * tem + index_map[sequence[i + l]];
        }
        if (flag) {
            na_vect[tem]++;
            pos_sum[tem] += (i + 1);
        }
    }
    for (int k = 0; k < m; k++) {
        if (na_vect[k] != 0) {
            na_vect[k + m] = pos_sum[k] / na_vect[k];
        } else {
            na_vect[k + m] = 0;
        }
    }
    
    for (int i = 0; i < n; i++) {
        bool flag = true;
        int tem = index_map[sequence[i]];
        if (tem==-1) continue;
        for (int l = 1; l < K; l++) {
            if (index_map[sequence[i + l]] == -1) {
                flag = false;
                break;
            }
           tem = 4 * tem + index_map[sequence[i + l]];
        }
        if (flag) {
            na_vect[tem+2*m] += (i + 1-na_vect[tem+m])*(i +1- na_vect[tem+m])/na_vect[tem]/n;
        }
    }
    return na_vect;
}



