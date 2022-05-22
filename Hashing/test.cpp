#include <Windows.h>
#include <bits/stdc++.h>

using namespace std;

int main()
{
    int i = 100;
    while(i--){
        cout << "\r" << i << flush;
        Sleep(1000);
    }
    return 0;
}