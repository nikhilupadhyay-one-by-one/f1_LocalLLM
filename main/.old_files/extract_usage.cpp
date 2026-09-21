#include <windows.h>
#include <iostream>
#include <vector>
// #include <bits/stdc++.h>

// Function to get CPU usage for a process
double getProcessCpuUsage(HANDLE hProcess) {
    PROCESS_BASIC_INFORMATION processInfo;
    PROCESS_BASIC_INFORMATION* processInfoPtr = &processInfo;

    // QueryProcessInfo to get process information
    if (!QueryProcessInformation(hProcess, processInfoPtr, sizeof(processInfo), nullptr, nullptr, nullptr, nullptr, nullptr)) {
        return 0.0; // Return 0 if QueryProcessInformation fails
    }

    // Get the CPU usage from the process information
    return processInfo.CpuUsage;
}

// Function to print the CPU usage for a process
void printProcessCpuUsage(const std::string& processName, HANDLE hProcess) {
    double cpuUsage = getProcessCpuUsage(hProcess);
    std::cout << processName << ": " << cpuUsage << "%" << std::endl;
}

int main() {
    // Get a handle to the current process
    HANDLE hCurrentProcess = GetCurrentProcess();

    // Enumerate processes
    HANDLE hProcesses[1000];
    DWORD cbNeeded, cbAllocated;
    cbNeeded = sizeof(hProcesses);
    cbAllocated = cbNeeded;
    DWORD processCount = 0;
    if (!EnumProcesses(hProcesses, cbNeeded, &cbNeeded)) {
        std::cerr << "Failed to enumerate processes." << std::endl;
        return 1;
    }

    // Get process information for each process
    for (DWORD i = 0; i < processCount; i++) {
        HANDLE hProcess = hProcesses[i];
        if (hProcess != NULL) {
            // Get the process name
            HANDLE hProcessName = OpenProcessName(hProcess, FALSE, NULL);
            if (hProcessName != NULL) {
                std::wstring processNameW = GetProcessName(hProcessName);
                std::wstring processName = processNameW;
                std::wstring processNameUTF8 = processName.c_str();

                // Get process information
                PROCESS_BASIC_INFORMATION processInfo;
                if (!QueryProcessInformation(hProcess, &processInfo, sizeof(processInfo), nullptr, nullptr, nullptr, nullptr, nullptr)) {
                    std::cerr << "Failed to get process information." << std::endl;
                    continue;
                }

                // Get the CPU usage
                double cpuUsage = getProcessCpuUsage(hProcess);
                std::cout << processNameUTF8 << ": " << cpuUsage << "%" << std::endl;
                std::cout << "   Modules:" << std::endl;
                for (DWORD j = 0; j < processInfo.NumberOfModules; j++) {
                    HANDLE hModule = hProcess->ProcessModule[j];
                    if (hModule != NULL) {
                        HANDLE hModuleInfo = OpenProcessModule(hModule, FALSE, NULL);
                        if (hModuleInfo != NULL) {
                            std::wstring moduleInfoW = GetModuleInfo(hModuleInfo);
                            std::wstring moduleInfoUTF8 = moduleInfoW;
                            std::cout << "     " << moduleInfoUTF8 << std::endl;
                        }
                    }
                }
            }
            CloseHandle(hProcessName);
        }
    }

    return 0;
}