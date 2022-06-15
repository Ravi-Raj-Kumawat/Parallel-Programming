#include "Includes/stdafx.h"

using namespace System;
using namespace System::Management;

void printHardwareInfo(String^ hardwareClass, String^ propertyName){
    ManagementObjectSearcher^ searcher = gcnew ManagementObjectSearcher("rooot\\CIMV2", "SELECT * FROM" + hardwareClass);
    ManagementObjectCollection^ collection = searcher->Get();

    for each (ManagementObject^ object in collection){
        Console::WriteLine(object[propreyName]->ToString());
    }
}