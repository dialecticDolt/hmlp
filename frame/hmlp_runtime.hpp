/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  


#ifndef HMLP_RUNTIME_HPP
#define HMLP_RUNTIME_HPP

#include <time.h>
#include <tuple>
#include <algorithm>
#include <vector>
#include <deque>
#include <map>
#include <unordered_map>
#include <limits>
#include <cstdint>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstddef>
#include <omp.h>

#ifdef USE_INTEL
#include <mkl.h>
#endif

#ifdef USE_PTHREAD_RUNTIME
#include <pthread.h>
#endif

#include <hmlp_device.hpp>
#include <hmlp_thread.hpp>
#include <hmlp_mpi.hpp>

#define MAX_WORKER 68


using namespace std;

namespace hmlp
{

//typedef enum 
//{
//  HMLP_SCHEDULE_DEFAULT,
//  HMLP_SCHEDULE_ROUND_ROBIN,
//  HMLP_SCHEDULE_UNIFORM,
//  HMLP_SCHEDULE_HEFT
//} SchedulePolicy;

/** @brief */
typedef enum { ALLOCATED, NOTREADY, QUEUED, RUNNING, EXECUTED, DONE, CANCELLED } TaskStatus;

/** @brief */
typedef enum { R, W, RW } ReadWriteType;


/**
 *  class Lock
 */ 

/** @brief Wrapper for omp or pthread mutex.  */ 
class Lock
{
  public:

    Lock();

    ~Lock();

    void Acquire();

    void Release();

  private:
#ifdef USE_PTHREAD_RUNTIME
    pthread_mutex_t lock;
#else
    omp_lock_t lock;
#endif
}; /** end class Lock */



/**
 *  class Event
 */ 

/** @brief Events are attached to tasks to record specific activities. */
class Event
{
  public:

    Event(); 

    void Set( string, double, double );

    void AddFlopsMops( double, double );

    void Begin( size_t );

    double GetBegin();

    double GetEnd();

    double GetDuration();

    double GetFlops();

    double GetMops();

    double GflopsPerSecond();

    void Normalize( double shift );

    void Terminate();

    void Print();

    void Timeline( bool isbeg, size_t tag );

    void MatlabTimeline( FILE *pFile );

  private:

    size_t tid = 0;

	  string label;

    double flops = 0.0;

    double mops = 0.0;

    double beg = 0.0;

    double end = 0.0;

    double sec = 0.0;

}; /** end class Event */





/**
 *  class Task
 */ 

/** @brief  */
class Task
{
  public:

    Task();

    ~Task();

    Worker *worker = NULL;

    string name;

    string label;

    int taskid;

    float cost = 0;

    bool priority = false;

    Event event;

    TaskStatus GetStatus();

    void SetStatus( TaskStatus status );
    void SetBatchStatus( TaskStatus status );

    void Submit();


    virtual void Set( string user_name, void (*user_function)(Task*), void *user_arg );

    virtual void Prefetch( Worker* );

    void Enqueue();

    void Enqueue( size_t tid );

    void TryEnqueue();

    void ForceEnqueue( size_t tid );

    void CallBackWhileWaiting();

    virtual void Execute( Worker* ) = 0;

    virtual void GetEventRecord();

    virtual void DependencyAnalysis();

    /* function ptr */
    void (*function)(Task*);

    /* function context */
    void *arg;

    /* Dependencies related members */
    volatile int n_dependencies_remaining = 0;
    void DependenciesUpdate();

    /** Read/write sets for dependency analysis */ 
    deque<Task*> in;
    deque<Task*> out;

    /** task lock */
    Lock task_lock;

    /** The next task in the batch job */
    Task *next = NULL;

    /** Preserve the current task in the call stack and context switch. */
    //bool ContextSwitchToNextTask( Worker* );

    /** If true, this task can be stolen */ 
    bool stealable = true;

    int created_by = 0;

    bool IsNested();

  private:

    volatile TaskStatus status;

    /** If true, then this is a nested task */
    bool is_created_in_epoch_session = false;

}; /** end class Task */


/** @brief This is a specific type of task that represents NOP. */
template<typename ARGUMENT>
class NULLTask : public Task
{
  public:

    void Set( ARGUMENT *arg ) {};

    void Execute( Worker* ) {};

}; /** end class NULLTask */





/**
 *  class MessageTask
 */ 

/** @brief This task is designed to take care MPI communications. */
class MessageTask : public Task
{
  public:

    /** MPI communicator will be provided during Submit(). */
    mpi::Comm comm;

    /** Provided by Set(). */
    int tar = 0;
    int src = 0;
    int key = 0;

    void Submit();
}; /** end class MessageTask */


/** @brief This task is the abstraction for all tasks handled by Listeners. */
class ListenerTask : public MessageTask
{
  public:

    void Submit();

    virtual void Listen() = 0;
}; /** end class ListenerTask */


/** @brief  */
template<typename T, typename ARG>
class SendTask : public MessageTask
{
  public:
    
    ARG *arg = NULL;

    vector<size_t> send_sizes;
    vector<size_t> send_skels;
    vector<T>      send_buffs;

    /** Override Set() */
    void Set( ARG *user_arg, int src, int tar, int key )
    {
      name = string( "Send" );
      label = to_string( tar );
      this->arg = user_arg;
      this->src = src;
      this->tar = tar;
      this->key = key;
      /** Compute FLOPS and MOPS */
      double flops = 0, mops = 0;
      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** "HIGH" priority */
      priority = true;
    };

    virtual void Pack() = 0;

    void Execute( Worker *user_worker ) 
    {
      Pack();
      mpi::Request req1, req2, req3;
      mpi::Isend( this->send_sizes.data(), this->send_sizes.size(),
          this->tar, this->key + 0, this->comm, &req1 ); 
      //mpi::Isend( this->send_skels.data(), this->send_skels.size(),
      //    this->tar, this->key + 1, this->comm, &req2 ); 
      mpi::Isend( this->send_buffs.data(), this->send_buffs.size(),
          this->tar, this->key + 2, this->comm, &req3 ); 
    };
}; /** end class SendTask */


template<typename T, typename ARG>
class RecvTask : public ListenerTask
{
	public:

    ARG *arg = NULL;

    vector<size_t> recv_sizes;
    vector<size_t> recv_skels;
    vector<T>      recv_buffs;

    /** Override Set() */
    void Set( ARG *user_arg, int src, int tar, int key )
    {
      name = string( "Listener" );
      label = to_string( src );
      this->arg = user_arg;
      this->src = src;
      this->tar = tar;
      this->key = key;
      /** Compute FLOPS and MOPS */
      double flops = 0, mops = 0;
      /** Setup the event */
      event.Set( label + name, flops, mops );
    };

    void DependencyAnalysis()
    {
      hmlp_msg_dependency_analysis( this->key, this->src, RW, this );
    };

    void Listen()
    {
      int src = this->src;
      int tar = this->tar;
      int key = this->key;
      mpi::Comm comm = this->comm;
      int cnt = 0;
      mpi::Status status;
      /** Probe the message that contains recv_sizes */
      mpi::Probe( src, key + 0, comm, &status );
      mpi::Get_count( &status, HMLP_MPI_SIZE_T, &cnt );
      recv_sizes.resize( cnt );
      mpi::Recv( recv_sizes.data(), cnt, src, key + 0, comm, &status );
      /** Calculate the total size of recv_buffs */
      cnt = 0;
      for ( auto c : recv_sizes ) cnt += c;
      recv_buffs.resize( cnt );
      /** Receive recv_buffs as well */
      mpi::Recv( recv_buffs.data(), cnt, src, key + 2, comm, &status );
    };

    virtual void Unpack() = 0;

    void Execute( Worker *user_worker ) { Unpack(); };

}; /** end class RecvTask */





/** @brief Recursive task sibmission (base case). */ 
template<typename ARG>
void RecuTaskSubmit( ARG *arg ) { /** do nothing */ }; 


/** @brief Recursive task sibmission. */ 
template<typename ARG, typename TASK, typename... Args>
void RecuTaskSubmit( ARG *arg, TASK& dummy, Args&... dummyargs )
{
  using NULLTASK = NULLTask<ARG>;
  /** Create the first normal task is it is not a NULLTask. */
  if ( !std::is_same<NULLTASK, TASK>::value )
  {
    auto task = new TASK();
    task->Submit();
    task->Set( arg );
    task->DependencyAnalysis();
  }
  /** now recurs to Args&... args, types are deduced automatically */
  RecuTaskSubmit( arg, dummyargs... );
}; /** end RecuDistTaskSubmit() */


/** @brief Recursive task execution (base case). */ 
template<typename ARG>
void RecuTaskExecute( ARG *arg ) { /** do nothing */ }; 


/** @brief Recursive task execution. */ 
template<typename ARG, typename TASK, typename... Args>
void RecuTaskExecute( ARG *arg, TASK& dummy, Args&... dummyargs )
{
  using NULLTASK = NULLTask<ARG>;
  /** Create the first normal task is it is not a NULLTask */
  if ( !std::is_same<NULLTASK, TASK>::value )
  {
    auto *task = new TASK();
    task->Set( arg );
    task->Execute( NULL );
    delete task;
  }
  /** Now recurs to Args&... args, types are deduced automatically */
  RecuTaskExecute( arg, dummyargs... );
}; /** end RecuDistTaskExecute() */





/**
 *  class ReadWrite
 */ 

/** @brief This class provides the ability to perform dependency analysis. */
class ReadWrite
{
  public:

    ReadWrite();

    /** Tracking the read set of the object. */
    deque<Task*> read;

    /** Tracking the write set of the object. */
    deque<Task*> write;

    void DependencyAnalysis( ReadWriteType type, Task *task );

    void DependencyCleanUp();

  private:

}; /** end class ReadWrite */





/**
 *  class MatrixReadWrite
 */ 

/** @brief This class creates 2D grids for 2D matrix partition. */
class MatrixReadWrite
{
  public:

    MatrixReadWrite();

    void Setup( size_t m, size_t n );

    bool HasBeenSetup();

    void DependencyAnalysis( size_t i, size_t j, ReadWriteType type, Task *task );

    void DependencyCleanUp();

  private:

    bool has_been_setup = false;

    size_t m = 0;

    size_t n = 0;

    vector<vector<ReadWrite> > Submatrices;

}; /** end class MatrixReadWrite */





/**
 *  class Scheduler
 */ 

/** @brief */
class Scheduler
{
  public:

    Scheduler();
    
    ~Scheduler();

    void Init( int n_worker );

    void Finalize();

    int n_worker = 0;

    size_t timeline_tag;

    double timeline_beg;

    /** Ready queues for normal tasks. */
    deque<Task*> ready_queue[ MAX_WORKER ];
    /** The tasklist records all tasks created in this epoch. */
    //deque<Task*> tasklist;
    /** Accessing ready_queue requires exclusive right to avoid race condition. */
    Lock ready_queue_lock[ MAX_WORKER ];

    /** The ready queue for nested tasks. */
    deque<Task*> nested_queue[ MAX_WORKER ];
    /** The tasklist records all nested tasks created in this epoch. */
    //deque<Task*> nested_tasklist;
    /** Accessing nested_ready_queue requires exclusive right to avoid race condition. */
    Lock nested_queue_lock[ MAX_WORKER ];


    /** The hashmap for asynchronous MPI tasks. */
    vector<unordered_map<int, ListenerTask*>> listener_tasklist;
    Lock listener_queue_lock;

    float time_remaining[ MAX_WORKER ];

    void ReportRemainingTime();

    /** Manually describe the dependencies */
    static void DependencyAdd( Task *source, Task *target );

    void MessageDependencyAnalysis( int key, int p, ReadWriteType type, Task *task );

    void NewTask( Task *task );

    void NewMessageTask( MessageTask *task );

    void NewListenerTask( ListenerTask *task );

    void ExecuteNestedTasksWhileWaiting( Worker *me, Task *waiting_task );

    void Summary();


  private:

    /** Main worker entry for the main iteration. */
    static void* EntryPoint( void* );

    /** The tasklist records all tasks created in this epoch. */
    deque<Task*> tasklist;

    /** The tasklist records all nested tasks created in this epoch. */
    deque<Task*> nested_tasklist;

    /** This mutex grants exclusive right to modify tasklists. */
    Lock tasklist_lock;

    /** Number of tasks and nested tasks that have been completed. */
    int n_task_completed = 0;
    int n_nested_task_completed = 0;

    /** Mutex for updating n_task_completed and n_nested_task_completed. */
    Lock n_task_lock;

    vector<Task*> DispatchFromNormalQueue( int tid );

    vector<Task*> DispatchFromNestedQueue( int tid );

    vector<Task*> StealFromOther();

    Task *StealFromQueue( size_t target );

    bool ConsumeTasks( Worker *me, vector<Task*> &batch );

    bool ConsumeTasks( Worker *me, Task *batch, bool is_nested );

    bool IsTimeToExit( int tid );

    void Listen( Worker* );

    Lock run_lock[ MAX_WORKER ];

    Lock pci_lock;

    Lock gpu_lock;

    /** Global communicator used between listeners. */
    mpi::Comm comm;

    /** This maps a key to a vector of p ReadWrite objects. */
    unordered_map<int, vector<ReadWrite>> msg_dependencies;

    /** If the flag is set, then there is a local conseneus. */
		bool do_terminate = false;

    /** If the flag is set, then an Ibarrier is post. */
    bool has_ibarrier = false;

    /** MPI request for Ibarrier. */
    mpi::Request ibarrier_request;

    /** If the flag is set, then there is a global consensus. */
    int ibarrier_consensus = 0;
}; /** end class Scheduler */





/**
 *  class RunTime
 */ 

/** @brief RunTime is statically created in hmlp_runtime.cpp. */
class RunTime
{
  public:

    RunTime();

    ~RunTime();

    void Init();

    void Run();

    void Finalize();

    /** Whether the runtime is in a epoch session. */
    bool IsInEpochSession();

    /** Consuming nested tasks while in a epoch session. */
    void ExecuteNestedTasksWhileWaiting( Task *waiting_task );

    //void pool_init();

    //void acquire_memory();

    //void release_memory( void* ptr );

    int n_worker = 0;

    int n_max_worker = 0;

    int n_background_worker = 1;

    thread_communicator *mycomm;

    Worker workers[ MAX_WORKER ];

    Device host;

    Device* device[ 1 ];

    Scheduler *scheduler;

  private:
   
    bool is_init = false;

    bool is_in_epoch_session = false;

}; /** end class Runtime */

}; /** end namespace hmlp */

hmlp::RunTime *hmlp_get_runtime_handle();

hmlp::Device *hmlp_get_device( int i );

bool hmlp_is_in_epoch_session();

//bool hmlp_is_nested_queue_empty();

void hmlp_set_num_background_worker( int n_background_worker );

void hmlp_msg_dependency_analysis( 
    int key, int p, hmlp::ReadWriteType type, hmlp::Task *task );


#endif /** define HMLP_RUNTIME_HPP */
