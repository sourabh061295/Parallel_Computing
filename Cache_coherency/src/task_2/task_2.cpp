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

// Enum declarations
enum Function 
{
    F_INVALID,
    F_READ,
    F_WRITE
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
enum Line_State 
{
    INVALID,
    VALID
};

// Define the cache structure
struct cacheBlock
{
    Line_State  state;
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


/* Bus interface, modified version from assignment. */
class Bus_if : public virtual sc_interface 
{
    public:
        virtual bool read(int writer, int addr) = 0;
        virtual bool write(int writer, int addr, int data) = 0;
};

// Cache module
SC_MODULE(Cache) 
{
    public:

    // Port declarations
    sc_in<bool>     Port_CLK;
    sc_in<Function> Port_Func;
    sc_in<int>      Port_Addr;
    sc_out<RetCode> Port_Done;
    sc_inout_rv<32> Port_Data;

    /* Bus snooping ports. */
    sc_in_rv<32>    Port_BusAddr;
    sc_in<int>      Port_BusWriter;
    sc_in<Function> Port_BusValid;

    /* Bus requests ports. */
    sc_port<Bus_if> Port_Bus;

    /* Variables. */
    int cache_id;

    /* Constructor. */
    SC_CTOR(Cache) 
    {
        // Snooping thread
        SC_THREAD(bus);
        // Cache functioning thread
        SC_THREAD(execute);
        /* Listen to clock.*/
        sensitive << Port_CLK.pos();
        dont_initialize();
    
        // Initialize the cache block
        // Loop through all ways
        for(int i = 0; i < NUM_WAYS; i++)
        {
            // Loop through all indexes
            for (int j = 0; j < (1 << INDEX_BITS); j++)
            {
                // Set all fields to 0
                for(int k = 0; k < CACHE_BYTES / 4; k++)
                {
                    c_data[i][j].data[k]  = 0;
                }
                c_data[i][j].state = INVALID;
                c_data[i][j].tag   = 0;
                c_data[i][j].age   = 0;
            }
        }
    }

    private:
    // Declare the cache structure
    cacheBlock c_data[NUM_WAYS][(1 << INDEX_BITS)];
    // Temporary variables
    uint32_t req_index, req_tag, req_word, way;
    
    // Function to update the age of the cache entries after a successful cache hit
    void update_age(uint32_t w, uint32_t idx)
    {
        uint8_t prev_age = c_data[w][idx].age;
        for (int i = 0; i < NUM_WAYS; i++)
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
        int old = 0;
        // Loop through all the ways
        for (int i = 0; i < NUM_WAYS; i++)
        {
            // Check for invalid cache blocks to update/replace
            if (c_data[i][idx].state == INVALID)
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
        // Loop through all ways to check for matching tag
        for(int i = 0; i < NUM_WAYS; i++)
        {
            // Check if a macthing tag exists and is valid
            if ((c_data[i][idx].tag == t) && (c_data[i][idx].state == VALID))
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
    
    /* Thread that handles the bus. */
    void bus() 
    {
        /* Continue while snooping is activated. */
        while(true)
        {
            /* Wait for work. */
            wait(Port_BusValid.value_changed_event());

            // Check if the request is not by the current cache itself
            if (Port_BusWriter.read() != this->cache_id)
            {
                /* Read address. */
                uint32_t inx, tg, wy;
                int addr = Port_BusAddr.read().to_int();
                // Get the requested tag, index and word from the address
                tg    = ((addr & TAG_MASK) >> (OFFSET_BITS +  INDEX_BITS));
                inx  = ((addr & INDEX_MASK) >> OFFSET_BITS);

                // Check if the cache block is already present and is valid
                if ((CACHE_HIT == check_hit(tg, inx, &wy)) && (this->c_data[wy][inx].state == VALID))
                {
                    switch(Port_BusValid.read())
                    {
                    // your code of what to do while snooping the bus
                    // keep in mind that a certain cache should distinguish between bus requests made by itself and requests made by other caches.
                    // count and report the total number of read/write operations on the bus, in which the desired address (by other caches) is found in the snooping cache (probe read hits and probe write hits).
                    case F_READ:
                        // Return the data of the cache hit in different cache
                        // Return the cache hit entry to the CPU
                        update_age(wy, inx);
                        Port_Data.write((addr < CACHE_SIZE) ? c_data[way][req_index].data[req_word] : 0);
                        break;
                    case F_WRITE:
                        // Invalidate the cache block
                        this->c_data[wy][inx].state = INVALID; 
                        break;
                    case F_INVALID:
                    default: break;
                    }
                }
	        }
        }
    }

    /* Thread that handles the data requests. */
    void execute() 
    {
        /* Begin loop. */
        while(true)
        {
            /* wait for something to do... */
            wait(Port_Func.value_changed_event());
            /* Read or write? */
            Function f = Port_Func.read();
            /* Read address. */
            int addr = Port_Addr.read();
            // Get the requested tag, index and word from the address
            req_tag    = ((addr & TAG_MASK) >> (OFFSET_BITS +  INDEX_BITS));
            req_index  = ((addr & INDEX_MASK) >> OFFSET_BITS);
            req_word   = (addr & WORD_MASK);
            int t_data   = 0;
            // Function WRITE to cache
            if (f == F_WRITE) 
            {
                // Read the data to be written into cache
                t_data = Port_Data.read().to_int();
                cout << sc_time_stamp() << ": CACHE recieved write" << endl;
                // Check if the cache already exists
                if (CACHE_HIT == check_hit(req_tag, req_index, &way))
                {
                    // Update the cache hit entry and increement the hit count
                    cout << sc_time_stamp() << ": CACHE WRITE HIT" << endl;
                    stats_writehit(this->cache_id);
                    c_data[way][req_index].data[req_word] = t_data;
                    // Update the age of all other entries in the cache set
                    update_age(way, req_index);
                }
                else
                {
                    // Update/replace the cache with new tag, data and update the age to 0
                    cout << sc_time_stamp() << ": CACHE WRITE MISS" << endl;
                    stats_writemiss(this->cache_id);
                    c_data[way][req_index].data[req_word] = t_data;
                    c_data[way][req_index].tag            = req_tag;
                    c_data[way][req_index].state          = VALID;
                    update_age(way, req_index);
                    // Send the bus signal for invalidation of same cache block
                    Port_Bus.write(this->cache_id, addr, t_data);
                }
                // Simulate cache latency
                wait();
                // Write operation complete
                Port_Done.write(RET_WRITE_DONE);
            }
            // Function READ to cache
            else if (f == F_READ) 
            {
                cout << sc_time_stamp() << ": CACHE received read" << endl;
                // Check if the cache already exists
                if (CACHE_HIT == check_hit(req_tag, req_index, &way))
                {
                    cout << sc_time_stamp() << ": CACHE READ HIT" << endl;
                    stats_readhit(this->cache_id);
                    // Return the cache hit entry to the CPU
                    Port_Data.write((addr < CACHE_SIZE) ? c_data[way][req_index].data[req_word] : 0);
                    // Update the age of all other entries in the cache set
                    update_age(way, req_index);
                }
                else
                {
                    cout << sc_time_stamp() << ": CACHE READ MISS" << endl;
                    stats_readmiss(this->cache_id);
                    // Prepare the cache data port to accept other data
                    Port_Data.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
                    // Send a bus read signal to check the data in other caches
                    Port_Bus.read(this->cache_id, addr);
                    // Wait until either any cache responds or the memory responds
                    wait(100);
                    // If no other cache responds, generate a random data simulating a memory module
                    if (Port_Data.read() == "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
                    {
                        // Generating random data to simulate memory read
                        int32_t rand_data = rand() + 1;
                        // Update/replace the cache line with new tag, data and update the age to 0
                        c_data[way][req_index].data[req_word] = rand_data;
                        c_data[way][req_index].tag            = req_tag;
                        c_data[way][req_index].state          = VALID;
                        update_age(way, req_index);
                        // Send the data to CPU
                        Port_Data.write((addr < CACHE_SIZE) ? rand_data : 0);
                    }
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

/* Bus class, provides a way to share one memory in multiple CPU + Caches. */
class Bus : public Bus_if, public sc_module 
{
    public:
    /* Ports and Signals. */
    sc_in<bool> Port_CLK;
    sc_out<Function> Port_BusValid;
    sc_out<int> Port_BusWriter;
    sc_signal_rv<32> Port_BusAddr;

    /* Bus mutex. */
    sc_mutex bus;

    /* Variables. */
    long waits;
    long reads;
    long writes;

    public:
    /* Constructor. */
    SC_CTOR(Bus) 
    {
        /* Handle Port_CLK to simulate delay */
        sensitive << Port_CLK.pos();

        // Initialize some bus properties
        Port_BusAddr.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");

        /* Update variables. */
        waits = 0;
        reads = 0;
        writes = 0;
    }

    /* Perform a read access to memory addr for CPU #writer. */
    virtual bool read(int writer, int addr)
    {
        /* Try to get exclusive lock on bus. */
        while(bus.trylock() == -1)
        {
            /* Wait when bus is in contention. */
            waits++;
            wait();
        }

        /* Update number of bus accesses. */
        reads++;

        /* Set lines. */
        Port_BusAddr.write(addr);
        Port_BusWriter.write(writer);
        Port_BusValid.write(F_READ);

        /* Wait for everyone to recieve. */
        wait();

        /* Reset. */
        Port_BusValid.write(F_INVALID);
        Port_BusAddr.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
        bus.unlock();

        return(true);
    };

    /* Write action to memory, need to know the writer, address and data. */
    virtual bool write(int writer, int addr, int data)
    {
        /* Try to get exclusive lock on the bus. */
        while(bus.trylock() == -1)
        {
            waits++;
            wait();
        }

        /* Update number of accesses. */
        writes++;

        /* Set. */
        Port_BusAddr.write(addr);
        Port_BusWriter.write(writer);
        Port_BusValid.write(F_WRITE);

        /* Wait for everyone to recieve. */
        wait();

        /* Reset. */
        Port_BusValid.write(F_INVALID);
        Port_BusAddr.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
        bus.unlock();

        return(true);
    }

    /* Bus output. */
    void output()
    {
        /* Write output as specified in the assignment. */
        double avg = (double)waits / double(reads + writes);
        printf("\n 1. Main memory access rates\n");
        printf("    Bus had %ld reads and %ld writes.\n", reads, writes);
        printf("    A total of %ld accesses.\n", reads + writes);
        printf("\n 2. Average time for bus acquisition\n");
        printf("    There were %ld waits for the bus.\n", waits);
        printf("    Average waiting time per access: %f cycles.\n", avg);
    }
};

// CPU module
SC_MODULE(CPU) 
{
    public:
    sc_in<bool>                Port_CLK;
    sc_in<RetCode>             Port_MemDone;
    sc_out<Function>           Port_MemFunc;
    sc_out<int>                Port_MemAddr;
    sc_inout_rv<32>            Port_MemData;

    /* Variables. */
    int cpu_id;

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
        Function  f;

        // Loop until end of tracefile
        while(!tracefile_ptr->eof())
        {
            // Get the next action for the processor in the trace
            if(!tracefile_ptr->next(this->cpu_id, tr_data))
            {
                cerr << "Error reading trace for CPU" << endl;
                break;
            }

            // Switch to choose operation
            switch(tr_data.type)
            {
                case TraceFile::ENTRY_TYPE_READ:
                    f = F_READ;
                    break;

                case TraceFile::ENTRY_TYPE_WRITE:
                    f = F_WRITE;
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

                if (f == F_WRITE) 
                {
                    cout << sc_time_stamp() << ": CPU " << this->cpu_id << "sends write" << endl;
                    uint32_t data = rand();
                    Port_MemData.write(data);
                    wait();
                    Port_MemData.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
                }
                else
                {
                    cout << sc_time_stamp() << ": CPU " << this->cpu_id << "sends read" << endl;
                }

                wait(Port_MemDone.value_changed_event());

                if (f == F_READ)
                {
                    cout << sc_time_stamp() << ": CPU " << this->cpu_id << "reads: " << Port_MemData.read() << endl;
                }
            }
            else
            {
                cout << sc_time_stamp() << ": CPU " << this->cpu_id << "executes NOP" << endl;
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
    // Get the tracefile argument and create Tracefile object
    // This function sets tracefile_ptr and num_cpus
    init_tracefile(&argc, &argv);

    // Initialize statistics counters
    stats_init();

    /* Create clock. */
    sc_clock clk;

    /* Create Bus and TraceFile Syncronizer. */
    Bus bus("bus");

    /* Connect to Clock. */
    bus.Port_CLK(clk);

    /* Cpu and Cache pointers. */
    Cache* cache[num_cpus];
    CPU* cpu[num_cpus];

    /* Signals Cache/CPU. */
    sc_buffer<Function>  sigMemFunc[num_cpus];
    sc_buffer<RetCode>   sigMemDone[num_cpus];
    sc_signal_rv<32>            sigMemData[num_cpus];
    sc_signal<int>              sigMemAddr[num_cpus];

    /* Signals Cache/Bus. */
    sc_signal<int>              sigBusWriter;
    sc_signal<Function>  sigBusValid;

    /* General Bus signals. */
    bus.Port_BusWriter(sigBusWriter);
    bus.Port_BusValid(sigBusValid);

    /* Create and connect all caches and cpu's. */
    for(int i = 0; i < (int)num_cpus; i++)
    {
        /* Each process should have a unique string name. */
        char name_cache[12];
        char name_cpu[12];

        /* Use number in unique string name. */
        //name_cache << "cache_" << i;
        //name_cpu   << "cpu_"   << i;
        sprintf(name_cache, "cache_%d", i);
        sprintf(name_cpu, "cpu_%d", i);

        /* Create CPU and Cache. */
        cache[i] = new Cache(name_cache);
        cpu[i] = new CPU(name_cpu);

        /* Set ID's. */
        cpu[i]->cpu_id = i;
        cache[i]->cache_id = i;
        // cache[i]->snooping = snooping;

        /* Cache to Bus. */
        cache[i]->Port_BusAddr(bus.Port_BusAddr);
        cache[i]->Port_BusWriter(sigBusWriter);
        cache[i]->Port_BusValid(sigBusValid);
        cache[i]->Port_Bus(bus);

        /* Cache to CPU. */
        cache[i]->Port_Func(sigMemFunc[i]);
        cache[i]->Port_Addr(sigMemAddr[i]);
        cache[i]->Port_Data(sigMemData[i]);
        cache[i]->Port_Done(sigMemDone[i]);

        /* CPU to Cache. */
        cpu[i]->Port_MemFunc(sigMemFunc[i]);
        cpu[i]->Port_MemAddr(sigMemAddr[i]);
        cpu[i]->Port_MemData(sigMemData[i]);
        cpu[i]->Port_MemDone(sigMemDone[i]);

        /* Set Clock */
        cache[i]->Port_CLK(clk);
        cpu[i]->Port_CLK(clk);
    }

    /* Start Simulation. */
    sc_start();

    return 0;
}