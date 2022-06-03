#include <stdio.h>
#include <stdlib.h>

typedef struct data{
    int value;
    struct data* next;
} data;

data* new_data(int i){   // A function to use as a 'Constructor' for 'data' variables
    data* new_Data = (data*)malloc(sizeof(data));
    new_Data->value = i;
    new_Data->next = NULL;
    return new_Data;
}

#define ARRAY_SIZE 100

unsigned int hash(int n){   // Hash Function
    return ((unsigned long)n * n) % ARRAY_SIZE;
}

void hash_init(data** hash_table){
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        hash_table[i] = NULL;
    }
    
}

void hash_print(data** hash_table){   // Printing The Hash Table
    for (int i = 0; i < ARRAY_SIZE; i++){
        printf("\t%u\t -->\t", i);
        data* itr = hash_table[i];
        while (itr != NULL)
        {
            printf("%u - ", itr->value);
            itr = itr->next;
        }
        printf("\n");
    }
    
}

void hash_insert(data** hash_table, data* d){    // To INSERT new data to Hash Table
    int idx = hash(d->value);
    if (hash_table[idx] == NULL){
        hash_table[idx] = d;
        return;
    }
    else{
        data* itr = hash_table[idx];
        while (itr->next != NULL){
            itr = itr->next;
        }
        itr->next = d;
    }
}

data* hash_delete(data** hash_table, data* d){   // To Delete data from Hash Table
    int idx = hash(d->value);
    if(hash_table[idx] == NULL) return NULL;
    else if(hash_table[idx]->value == d->value){
        hash_table[idx] = hash_table[idx]->next;
        return NULL;
    }
    else{
        data* itr = hash_table[idx];
        while (itr != NULL && itr->next != NULL && itr->next->value != d->value)
        {
            itr = itr->next;
        }
        if(itr == NULL) return NULL;
        else{
            data* to_delete = itr->next;
            itr->next = itr->next->next;
            return to_delete;
        }
    }
}

void main()
{
    data** hash_table = (data**)malloc(ARRAY_SIZE*sizeof(data*));
    hash_init(hash_table);
    data* data1 = new_data(1);
    data* data2 = new_data(2);
    data* data3 = new_data(3);
    data* data4 = new_data(4);
    data* data5 = new_data(5);
    data* data6 = new_data(6);
    data* data7 = new_data(7);
    data* data8 = new_data(8);
    data* data9 = new_data(9);
    data* data10 = new_data(10);
    data* data11 = new_data(11);
    data* data12 = new_data(12);
    data* data13 = new_data(13);
    data* data14 = new_data(14);
    data* data15 = new_data(15);
    hash_insert(hash_table, data1);
    hash_insert(hash_table, data2);
    hash_insert(hash_table, data3);
    hash_insert(hash_table, data4);
    hash_insert(hash_table, data5);
    hash_insert(hash_table, data6);
    hash_insert(hash_table, data7);
    hash_insert(hash_table, data8);
    hash_insert(hash_table, data9);
    hash_insert(hash_table, data10);
    hash_insert(hash_table, data11);
    hash_insert(hash_table, data12);
    hash_insert(hash_table, data13);
    hash_insert(hash_table, data14);
    hash_insert(hash_table, data15);
    hash_delete(hash_table, data1);
    hash_print(hash_table);
    return;
}