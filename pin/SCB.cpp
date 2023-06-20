#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <string.h>
#include "pin.H"
using std::ofstream;
using std::string;
using std::hex;
using std::ios;
using std::setw;
using std::cerr;
using std::dec;
using std::endl;
ofstream outFile;

string TARGET_IMG = "a.out";
// string TARGET_RTN = "";

static std::unordered_map<ADDRINT, std::string> str_of_ins_at;
static std::unordered_map<ADDRINT, std::string> str_of_func_at;
    
const char * StripPath(const char * path)
{
    const char * file = strrchr(path,'/');
    if (file)
        return file+1;
    else
        return path;
}

VOID RecordBranch(ADDRINT ip, ADDRINT target)
{
    outFile << str_of_func_at[ip] << "; " << str_of_ins_at[ip] << "; " << ip << " --> " << target << endl;
}

// Pin calls this function every time a new rtn is executed
VOID Routine(RTN rtn, VOID *v)
{
    string image_name = StripPath(IMG_Name(SEC_Img(RTN_Sec(rtn))).c_str());

    // if (true)
    if (strcmp(image_name.c_str(), TARGET_IMG.c_str()) == 0)
    {
        RTN_Open(rtn);
        string rtn_name = StripPath(RTN_Name(rtn).c_str());

        if (true) // can check the rtn here
        {
            for (INS ins = RTN_InsHead(rtn); INS_Valid(ins); ins = INS_Next(ins))
            {
                if (INS_IsXbegin(ins) || INS_IsXend(ins))
                    continue;
                
                if (INS_IsControlFlow(ins)) {
                    str_of_ins_at[INS_Address(ins)] = INS_Disassemble(ins);
                    str_of_func_at[INS_Address(ins)] = rtn_name;

                    INS_InsertCall(
                        ins, IPOINT_BEFORE, (AFUNPTR)RecordBranch,
                        IARG_INST_PTR,
                        IARG_BRANCH_TARGET_ADDR,
                        IARG_END);
                }
            }
        }
        RTN_Close(rtn);
    }
}

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "branch.out", "specify output file name");

KNOB<string> KnobTargetImg(KNOB_MODE_WRITEONCE, "pintool",
    "t", "a.out", "specify target img name");

// This function is called when the application exits
// It prints the name and count for each procedure
VOID Fini(INT32 code, VOID *v)
{
    outFile.setf(ios::showbase);
    outFile << "#eof" << endl;
    outFile.close();
}
/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */
INT32 Usage()
{
    cerr << "Trace" << endl;
    cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}
/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */
int main(int argc, char * argv[])
{
    // Initialize symbol table code, needed for rtn instrumentation
    PIN_InitSymbols();
    // Initialize pin
    if (PIN_Init(argc, argv)) return Usage();

    TARGET_IMG = KnobTargetImg.Value();
    outFile.open(KnobOutputFile.Value().c_str());
    
    // Register Routine to be called to instrument rtn
    RTN_AddInstrumentFunction(Routine, 0);
    // Register Fini to be called when the application exits
    PIN_AddFiniFunction(Fini, 0);
    
    // Start the program, never returns
    PIN_StartProgram();
    
    return 0;
}