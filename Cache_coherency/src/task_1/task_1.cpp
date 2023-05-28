#include "aca2009.h"
#include <systemc.h>
#include <math.h>
#include <iostream>

// MACROS
#define CACHE_SIZE  32768
#define ADDR_LINES  32
#define CACHE_BYTES 32
#define NUM_WAYS    8 

using namespace std;

// Define the cache structure
struct cacheBlock
{
    bool        valid;
    uint32_t    tag;
    int32_t     data[CACHE_BYTES / 4];
    uint8_t     age;
};

// Bit length caculation
static const int OFFSET_BITS = log2(CACHE_BYTES);
static const int INDEX_BITS  = log2(CACHE_SIZE / (CACHE_BYTES * NUM_WAYS));
static const int TAG_BITS    = (ADDR_LINES - INDEX_BITS - OFFSET_BITS);
// Mask generation
static const int TAG_MASK    = ((1 << TAG_BITS) - 1) << (INDEX_BITS + OFFSET_BITS);
static const int INDEX_MASK  = ((1 << INDEX_BITS) - 1) << OFFSET_BITS;
static const int WORD_MASK   = ((1 << OFFSET_BITS) - 1);

// Cache module
SC_MODULE(Cache) 
{
    public:
    // Enum declarations
    enum Function 
    {
        FUNC_READ,
        FUNC_WRITE
    };
    enum RetCode 
    {
        RET_READ_DONE,
        RET_WRITE_DONE,
        RET_FUNC_INVALID
    };
    enum CacheSignal
    {
        CACHE_HIT,
        CACHE_MISS
    };

    // Port declarations
    sc_in<bool>     Port_CLK;
    sc_in<Function> Port_Func;
    sc_in<int>      Port_Addr;
    sc_out<RetCode> Port_Done;
    sc_inout_rv<32> Port_Data;

    // Initialize cache
    void cache_init()
    {
        int i, j, k;
        // Loop through all ways
        for(i = 0; i < NUM_WAYS; i++)
        {
            // Loop through all indexes
            for (j = 0; j < (1 << INDEX_BITS); j++)
            {
                // Set all fields to 0
                for(k = 0; k < CACHE_BYTES / 4; k++)
                {
                    c_data[i][j].data[k]  = 0;
                }
                c_data[i][j].valid = false;
                c_data[i][j].tag   = 0;
                c_data[i][j].age   = 0;
            }
        }
    }

    // Cache module constructor
    SC_CTOR(Cache) 
    {
        SC_THREAD(execute);
        sensitive << Port_CLK.pos();
        dont_initialize();
        // Initialize all cache blocks to 0
        cache_init();
    }

    private:
    // Declare the cache structure
    cacheBlock c_data[NUM_WAYS][(1 << INDEX_BITS)];
    // Temporary variables
    uint32_t req_index, req_tag, req_word, way;

    // Function to update the age of the cache entries after a successful cache hit
    void update_age(uint32_t w, uint32_t idx)
    {
        int i;
        uint8_t prev_age = c_data[w][idx].age;
        for (i = 0; i < NUM_WAYS; i++)
        {
            // Increase the age of cache blocks younger than the current block
            if (c_data[i][idx].age < prev_age)
            {
                c_data[i][idx].age++;
            }
        }
        // Reset the age of current cache block making it the most recent entry
        c_data[w][idx].age = 0;
    }

    // Function to get a way which can be replaced for new cache entry
    uint32_t get_replacable_way(uint32_t idx)
    {
        int i, old = 0;
        // Loop through all the ways
        for (i = 0; i < NUM_WAYS; i++)
        {
            // Check for invalid cache blocks to update/replace
            if (c_data[i][idx].valid == false)
            {
                // Return if any cache block is invalid
                return i;
            }
            // Note down the oldest entry in the cache if all the blocks are valid
            else if (c_data[i][idx].age == NUM_WAYS - 1)
            {
                old = i;
            }
        }
        // Return the oldest entry to replace
        return old;
    }

    // Function to check for cache hit or miss
    CacheSignal check_hit(uint32_t t, uint32_t idx, uint32_t* w)
    {
        int i;
        // Loop through all ways to check for matching tag
        for(i = 0; i < NUM_WAYS; i++)
        {
            // Check if a macthing tag exists and is valid
            if ((c_data[i][idx].tag == t) && (c_data[i][idx].valid == true))
            {
                // Store the way number in the pointer variable
                *w = i;
                // Return the way number with a CACHE HIT signal
                return CACHE_HIT; 
            }
        }
        // Loop terminates if no cache hit occurs
        // Update way number which can be replaced for a CACHE MISS and return
        *w = get_replacable_way(idx);
        return CACHE_MISS;
    }

    void execute() 
    {
        while (true)
        {
            // This is fine since we use sc_buffer
            wait(Port_Func.value_changed_event());	
            // Read the port signals
            Function f = Port_Func.read();
            int addr   = Port_Addr.read();
            // Get the requested tag, index and word from the address
            req_tag    = ((addr & TAG_MASK) >> (OFFSET_BITS +  INDEX_BITS));
            req_index  = ((addr & INDEX_MASK) >> OFFSET_BITS);
            req_word   = (addr & WORD_MASK);
            int t_data   = 0;
            // Function WRITE to cache
            if (f == FUNC_WRITE) 
            {
                // Read the data to be written into cache
                t_data = Port_Data.read().to_int();
                cout << sc_time_stamp() << ": CACHE recieved write" << endl;
                // Check if the cache already exists
                if (CACHE_HIT == check_hit(req_tag, req_index, &way))
                {
                    // Update the cache hit entry and increement the hit count
                    cout << sc_time_stamp() << ": CACHE WRITE HIT" << endl;
                    stats_writehit(0);
                    c_data[way][req_index].data[req_word] = t_data;
                    // Update the age of all other entries in the cache set
                    update_age(way, req_index);
                }
                else
                {
                    // Update/replace the cache with new tag, data and update the age to 0
                    cout << sc_time_stamp() << ": CACHE WRITE MISS" << endl;
                    stats_writemiss(0);
                    c_data[way][req_index].data[req_word] = t_data;
                    c_data[way][req_index].tag            = req_tag;
                    c_data[way][req_index].valid          = true;
                    update_age(way, req_index);
                }
                // Simulate cache latency
                wait();
                // Write operation complete
                Port_Done.write(RET_WRITE_DONE);
            }
            // Function READ to cache
            else if (f == FUNC_READ) 
            {
                cout << sc_time_stamp() << ": CACHE received read" << endl;
                // Check if the cache already exists
                if (CACHE_HIT == check_hit(req_tag, req_index, &way))
                {
                    cout << sc_time_stamp() << ": CACHE READ HIT" << endl;
                    stats_readhit(0);
                    // Return the cache hit entry to the CPU
                    Port_Data.write((addr < CACHE_SIZE) ? c_data[way][req_index].data[req_word] : 0);
                    // Update the age of all other entries in the cache set
                    update_age(way, req_index);
                }
                else
                {
                    cout << sc_time_stamp() << ": CACHE READ MISS" << endl;
                    stats_readmiss(0);
                    // Wait time to simulate memory latency
                    wait(100);
                    // Generating random data to simulate memory read
                    int32_t rand_data = rand() + 1;
                    // Update/replace the cache line with new tag, data and update the age to 0
                    c_data[way][req_index].data[req_word] = rand_data;
                    c_data[way][req_index].tag            = req_tag;
                    c_data[way][req_index].valid          = true;
                    update_age(way, req_index);
                    // Send the data to CPU
                    Port_Data.write((addr < CACHE_SIZE) ? rand_data : 0);
                }
                // Read operation complete
                Port_Done.write(RET_READ_DONE);
                wait();
                Port_Data.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
            }
            else
            {
                // Ideally, this part of code is unreachable
                cout << sc_time_stamp() << ": INVALID CACHE FUNCTION" << endl;
                Port_Done.write(RET_FUNC_INVALID);
            }
        }
    }
}; 

// CPU module
SC_MODULE(CPU) 
{
    public:
    sc_in<bool>                Port_CLK;
    sc_in<Cache::RetCode>      Port_MemDone;
    sc_out<Cache::Function>    Port_MemFunc;
    sc_out<int>                Port_MemAddr;
    sc_inout_rv<32>            Port_MemData;

    // CPU module constructor
    SC_CTOR(CPU) 
    {
        SC_THREAD(execute);
        sensitive << Port_CLK.pos();
        dont_initialize();
    }

    private:
    void execute() 
    {
        TraceFile::Entry  tr_data;
        Cache::Function  f;

        // Loop until end of tracefile
        while(!tracefile_ptr->eof())
        {
            // Get the next action for the processor in the trace
            if(!tracefile_ptr->next(0, tr_data))
            {
                cerr << "Error reading trace for CPU" << endl;
                break;
            }

            // Switch to choose operation
            switch(tr_data.type)
            {
                case TraceFile::ENTRY_TYPE_READ:
                    f = Cache::FUNC_READ;
                    break;

                case TraceFile::ENTRY_TYPE_WRITE:
                    f = Cache::FUNC_WRITE;
                    break;

                case TraceFile::ENTRY_TYPE_NOP:
                    break;

                default:
                    cerr << "Error, got invalid data from Trace" << endl;
                    exit(0);
            }

            if(tr_data.type != TraceFile::ENTRY_TYPE_NOP)
            {
                Port_MemAddr.write(tr_data.addr);
                Port_MemFunc.write(f);

                if (f == Cache::FUNC_WRITE) 
                {
                    cout << sc_time_stamp() << ": CPU sends write" << endl;

                    uint32_t data = rand();
                    Port_MemData.write(data);
                    wait();
                    Port_MemData.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
                }
                else
                {
                    cout << sc_time_stamp() << ": CPU sends read" << endl;
                }

                wait(Port_MemDone.value_changed_event());

                if (f == Cache::FUNC_READ)
                {
                    cout << sc_time_stamp() << ": CPU reads: " << Port_MemData.read() << endl;
                }
            }
            else
            {
                cout << sc_time_stamp() << ": CPU executes NOP" << endl;
            }
            // Advance one cycle in simulated time            
            wait();
        }
        
        // Finished the Tracefile, now stop the simulation
        sc_stop();
    }
};


int sc_main(int argc, char* argv[])
{
    try
    {
        // Get the tracefile argument and create Tracefile object
        // This function sets tracefile_ptr and num_cpus
        init_tracefile(&argc, &argv);

        // Initialize statistics counters
        stats_init();

        // Instantiate Modules
        Cache  cache("cache");
        CPU    cpu("cpu");

        // Signals
        sc_buffer<Cache::Function>  sigMemFunc;
        sc_buffer<Cache::RetCode>   sigMemDone;
        sc_signal<int>              sigMemAddr;
        sc_signal_rv<32>            sigMemData;

        // The clock that will drive the CPU and Memory
        sc_clock clk;

        // Connecting module ports with signals
        cache.Port_Func(sigMemFunc);
        cache.Port_Addr(sigMemAddr);
        cache.Port_Data(sigMemData);
        cache.Port_Done(sigMemDone);

        cpu.Port_MemFunc(sigMemFunc);
        cpu.Port_MemAddr(sigMemAddr);
        cpu.Port_MemData(sigMemData);
        cpu.Port_MemDone(sigMemDone);

        cache.Port_CLK(clk);
        cpu.Port_CLK(clk);

        cout << "Running (press CTRL+C to interrupt)... " << endl;


        // Start Simulation
        sc_start();
        
        // Print statistics after simulation finished
        stats_print();
    }

    catch (exception& e)
    {
        cerr << e.what() << endl;
    }
    
    return 0;
}
