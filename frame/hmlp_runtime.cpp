#include <hmlp_runtime.hpp>

// #define DEBUG_RUNTIME 1
// #define DEBUG_SCHEDULER 1


struct 
{
  bool operator()( std::tuple<bool, double, size_t> a, std::tuple< bool , double, size_t> b )
  {   
    return std::get<1>( a ) < std::get<1>( b );
  }   
} EventLess;



namespace hmlp
{

static RunTime rt;

range::range( int beg, int end, int inc )
{
  info = std::make_tuple( beg, end, inc );
};

int range::beg()
{
  return std::get<0>( info );
};

int range::end()
{
  return std::get<1>( info );
};

int range::inc()
{
  return std::get<2>( info );
};

range GetRange
( 
  SchedulePolicy strategy, 
  int beg, int end, int nb, int tid, int nparts 
)
{
  switch ( strategy )
  {
    case HMLP_SCHEDULE_DEFAULT:
      {
        auto tid_beg = beg + tid * nb;
        auto tid_inc = nparts * nb;
        return range( tid_beg, end, tid_inc );
      }
    case HMLP_SCHEDULE_ROUND_ROBIN:
      {
        auto tid_beg = beg + tid * nb;
        auto tid_inc = nparts * nb;
        return range( tid_beg, end, tid_inc );
      }
    case HMLP_SCHEDULE_UNIFORM:
      printf( "GetRange(): HMLP_SCHEDULE_UNIFORM not yet implemented yet.\n" );
      exit( 1 );
    case HMLP_SCHEDULE_HEFT:
      {
        assert( nparts == 4 );
        int len = end - beg - 1;
        int big = ( len * 30 ) / 100 + 1;
        int sma = ( len * 20 ) / 100 + 1;

        int tid_beg, tid_end;

        if ( tid == 0 )
        {
          tid_beg = beg;
          tid_end = beg + big;
        }
        beg += big;

        if ( tid == 1 )
        {
          tid_beg = beg;
          tid_end = beg + sma;
        }
        beg += sma;

        if ( tid == 2 )
        {
          tid_beg = beg;
          tid_end = beg + sma;
        }
        beg += sma;

        if ( tid == 3 )
        {
          tid_beg = beg;
          tid_end = beg + big;
        }
        beg += big;

        if ( tid_end > end ) tid_end = end;
        return range( tid_beg, tid_end, nb );
      }
    default:
      printf( "GetRange(): not a legal scheduling strategy.\n" );
      exit( 1 );
  }
};

range GetRange( int beg, int end, int nb, int tid, int nparts )
{
  return GetRange( HMLP_SCHEDULE_DEFAULT, beg, end, nb, tid, nparts );
};

range GetRange( int beg, int end, int nb )
{
  return GetRange( HMLP_SCHEDULE_DEFAULT, beg, end, nb, 0, 1 );
};

/**
 *  @brief Lock
 */ 
Lock::Lock()
{
#ifdef USE_PTHREAD_RUNTIME
  if ( pthread_mutex_init( &lock, NULL ) )
  {
    printf( "pthread_mutex_init(): cannot initialize locks properly\n" );
  }
#else
  omp_init_lock( &lock );
#endif
};

Lock::~Lock()
{
#ifdef USE_PTHREAD_RUNTIME
  if ( pthread_mutex_destroy( &lock ) )
  {
    printf( "pthread_mutex_destroy(): cannot destroy locks properly\n" );
  }
#else
  omp_destroy_lock( &lock );
#endif
};

void Lock::Acquire()
{
#ifdef USE_PTHREAD_RUNTIME
  if ( pthread_mutex_lock( &lock ) )
  {
    printf( "pthread_mutex_lock(): cannot acquire locks properly\n" );
  }
#else
  omp_set_lock( &lock );
#endif
};

void Lock::Release()
{
#ifdef USE_PTHREAD_RUNTIME
  if ( pthread_mutex_unlock( &lock ) )
  {
    printf( "pthread_mutex_lock(): cannot release locks properly\n" );
  }
#else
  omp_unset_lock( &lock );
#endif
};


Event::Event() : flops( 0.0 ), mops( 0.0 ), beg( 0.0 ), end( 0.0 ), sec( 0.0 ) {};

//Event::Event( float _flops, float _mops ) : beg( 0.0 ), end( 0.0 ), sec( 0.0 )
//{
//  flops = _flops;
//  mops = _mops;
//};


void Event::Set( double _flops, double _mops )
{
  flops = _flops;
  mops = _mops;
};

void Event::Begin( size_t _tid )
{
  tid = _tid;
  beg = omp_get_wtime();
};

void Event::Normalize( double shift )
{
  beg -= shift;
  end -= shift;
};

void Event::Terminate()
{
  end = omp_get_wtime();
  sec = end - beg;
};

double Event::GetBegin()
{
  return beg;
};

double Event::GetEnd()
{
  return end;
};

double Event::GetDuration()
{
  return sec;
};

void Event::Print()
{
  printf( "beg %5.3lf end %5.3lf sec %5.3lf flops %E mops %E\n",
      beg, end, sec, flops, mops );
};

void Event::Timeline( bool isbeg, size_t tag )
{
  if ( isbeg )
  {
    printf( "@TIMELINE\n" );
    printf( "worker%lu, %lu, %E, %lf\n", tid, 2 * tag + 0, beg, (double)tid + 0.0 );
    printf( "@TIMELINE\n" );
    printf( "worker%lu, %lu, %E, %lf\n", tid, 2 * tag + 1, beg, (double)tid + 0.6 );
  }
  else
  {
    printf( "@TIMELINE\n" );
    printf( "worker%lu, %lu, %E, %lf\n", tid, 2 * tag + 0, beg, (double)tid + 0.6 );
    printf( "@TIMELINE\n" );
    printf( "worker%lu, %lu, %E, %lf\n", tid, 2 * tag + 1, beg, (double)tid + 0.0 );
  }

}






/**
 *  @brief Task
 */ 
Task::Task()
{
  status = ALLOCATED;
  //rt.scheduler->NewTask( this );
  status = NOTREADY;
};

Task::~Task()
{};

TaskStatus Task::GetStatus()
{
  return status;
};

void Task::SetStatus( TaskStatus next_status )
{
  status = next_status;
};

void Task::Submit()
{
  rt.scheduler->NewTask( this );
};

void Task::Set( std::string user_name, void (*user_function)(Task*), void *user_arg )
{
  name = user_name;
  function = user_function;
  arg = user_arg;
  status = NOTREADY;
};

void Task::DependenciesUpdate()
{
  while ( out.size() )
  {
    Task *child = out.front();

    child->task_lock.Acquire();
    {
      child->n_dependencies_remaining --;

      //std::cout << child->n_dependencies_remaining << std::endl;

      if ( !child->n_dependencies_remaining && child->status == NOTREADY )
      {
        child->Enqueue();
      }
    }
    child->task_lock.Release();
    out.pop_front();
  }
  status = DONE;
};

void Task::Execute( Worker *user_worker )
{
  function( this );
};

void Task::GetEventRecord() {};

/**
 *  @brief 
 */ 
void Task::Enqueue()
{
  float cost = 0.0;
  float earliest_t = -1.0;
  int assignment = -1;

  // Determine which work the task should go to using HEFT policy.
  for ( int i = 0; i < rt.n_worker; i ++ )
  {
    float terminate_t = rt.scheduler->time_remaining[ i ];
    float cost = rt.workers[ i ].EstimateCost( this );
    if ( earliest_t == -1.0 || terminate_t + cost < earliest_t )
    {
      earliest_t = terminate_t + cost;
      assignment = i;
    }
  }

  rt.scheduler->ready_queue_lock[ assignment ].Acquire();
  {
    status = QUEUED;
    rt.scheduler->time_remaining[ assignment ] += 
      rt.workers[ assignment ].EstimateCost( this );
    //rt.scheduler->ready_queue[ assignment ].push_back( this );
    rt.scheduler->ready_queue[ assignment ].push_front( this );
  }
  rt.scheduler->ready_queue_lock[ assignment ].Release();
};




/**
 *  @brief Scheduler
 */ 
Scheduler::Scheduler() : timeline_tag( 500 )
{
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler()\n" );
#endif
  timeline_beg = omp_get_wtime();
};

Scheduler::~Scheduler()
{
#ifdef DEBUG_SCHEDULER
  printf( "~Scheduler()\n" );
#endif
};


void Scheduler::Init( int user_n_worker )
{
  n_worker = user_n_worker;
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::Init()\n" );
#endif
  // Reset task counter.
  n_task = 0;
#ifdef USE_PTHREAD_RUNTIME
  for ( int i = 0; i < n_worker; i ++ )
  {
    time_remaining[ i ] = 0.0;
    rt.workers[ i ].tid = i;
    rt.workers[ i ].scheduler = this;
    pthread_create
    ( 
      &(rt.workers[ i ].pthreadid), NULL,
      EntryPoint, (void*)&(rt.workers[ i ])
    );
  }
  // Now the master thread
  EntryPoint( (void*)&(rt.workers[ 0 ]) );
#else
  #pragma omp parallel for num_threads( n_worker )
  for ( int i = 0; i < n_worker; i ++ )
  {
    time_remaining[ i ] = 0.0;
    rt.workers[ i ].tid = i;
    rt.workers[ i ].scheduler = this;
    EntryPoint( (void*)&(rt.workers[ i ]) );
  }
#endif
};


void Scheduler::NewTask( Task *task )
{
  tasklist_lock.Acquire();
  {
    tasklist.push_back( task );
  }
  tasklist_lock.Release();
};

void Scheduler::Finalize()
{
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::Finalize()\n" );
#endif
#ifdef USE_PTHREAD_RUNTIME
  for ( int i = 0; i < rt.n_worker; i ++ )
  {
    pthread_join( rt.workers[ i ].pthreadid, NULL );
  }
#else
#endif
#ifdef DUMP_ANALYSIS_DATA
  Summary();
#endif
  // Reset tasklist
  for ( auto it = tasklist.begin(); it != tasklist.end(); it ++ )
  {
    delete *it; 
  }
  tasklist.clear();
};


/**
 *  @brief Add an direct edge (dependency) from source to target. 
 *         That is to say, target depends on source.
 *
 */ 
void Scheduler::DependencyAdd( Task *source, Task *target )
{
  // Update the source list.
  source->task_lock.Acquire();
  {
    source->out.push_back( target );
  }
  source->task_lock.Release();

  // Update the target list.
  target->task_lock.Acquire();
  {
    target->in.push_back( source );
    if ( source->GetStatus() != DONE )
    {
      target->n_dependencies_remaining ++;
    }
  }
  target->task_lock.Release();
}; // end DependencyAdd()


void* Scheduler::EntryPoint( void* arg )
{
  Worker *me = reinterpret_cast<Worker*>( arg );
  Scheduler *scheduler = me->scheduler;

#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::EntryPoint()\n" );
  printf( "pthreadid %d\n", me->tid );
#endif

  while ( 1 )
  {
    Task *task = NULL;

    scheduler->ready_queue_lock[ me->tid ].Acquire();
    {
      if ( scheduler->ready_queue[ me->tid ].size() )
      {
        task = scheduler->ready_queue[ me->tid ].front();
        scheduler->ready_queue[ me->tid ].pop_front();
      }
    }
    scheduler->ready_queue_lock[ me->tid ].Release();

    if ( task )
    {
      task->SetStatus( RUNNING );
      if ( me->Execute( task ) )
      {
        scheduler->ready_queue_lock[ me->tid ].Acquire();
        {
          scheduler->time_remaining[ me->tid ] -= task->cost;
          if ( scheduler->time_remaining[ me->tid ] < 0.0 )
            scheduler->time_remaining[ me->tid ] = 0.0;
        }
        scheduler->ready_queue_lock[ me->tid ].Release();

        task->DependenciesUpdate();
        scheduler->n_task_lock.Acquire();
        {
          scheduler->n_task ++;
        }
        scheduler->n_task_lock.Release();
      }
    }
    else // No task in my ready_queue. Steal from others.
    {
      //scheduler->time_remaining[ me->tid ] = 0.0;
     
      //for ( int p = 0; p < scheduler->n_worker; p ++ )
      //{
      //  printf( "worker %d try to steal from worker %d\n", me->tid, p );  
      //}
    }

    if ( scheduler->n_task >= scheduler->tasklist.size() ) 
    {
      break;
    }
  }

  return NULL;
};


void Scheduler::Summary()
{
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[ 80 ];

  time( &rawtime );
  timeinfo = localtime( &rawtime );
  strftime( buffer, 80, "%T.", timeinfo );

  //printf( "%s\n", buffer );

  std::deque<std::tuple<bool, double, size_t>> timeline;


  if ( tasklist.size() )
  {
    for ( size_t i = 0; i < tasklist.size(); i ++ )
    {
      tasklist[ i ]->event.Normalize( timeline_beg );
    }

    for ( size_t i = 0; i < tasklist.size(); i ++ )
    {
      auto &event = tasklist[ i ]->event;
      timeline.push_back( std::make_tuple( true,  event.GetBegin(), i ) );
      timeline.push_back( std::make_tuple( false, event.GetEnd(),   i ) );
    }

    std::sort( timeline.begin(), timeline.end(), EventLess );

    for ( size_t i = 0; i < timeline.size(); i ++ )
    {
      auto &data = timeline[ i ];
      auto &event = tasklist[ std::get<2>( data ) ]->event;  
      event.Timeline( std::get<0>( data ), i + timeline_tag );
    }

    timeline_tag += timeline.size();
  }

}; // end void Schediler::Summary()




RunTime::RunTime() :
  n_worker( 0 )
{
#ifdef DEBUG_RUNTIME
  printf( "Runtime()\n" );
#endif
};

RunTime::~RunTime()
{
#ifdef DEBUG_RUNTIME
  printf( "~Runtime()\n" );
#endif
};

void RunTime::Init()
{
  #pragma omp critical (init)
  {
    if ( !is_init )
    {
      //n_worker = 4;
      n_worker = omp_get_max_threads();
      scheduler = new Scheduler();
      is_init = true;
    }
  }
};

void RunTime::Run()
{
  if ( !is_init ) 
  {
    Init();
  }
  scheduler->Init( n_worker );
  scheduler->Finalize();
};

void RunTime::Finalize()
{
  #pragma omp critical (init)
  {
    if ( is_init )
    {
      scheduler->Finalize();
      delete scheduler;
      is_init = false;
    }
  }
};

//void hmlp_runtime::pool_init()
//{
//
//};
//
//void hmlp_runtime::acquire_memory()
//{
//
//};
//
//void hmlp_runtime::release_memory( void *ptr )
//{
//
//};

}; // end namespace hmlp

void hmlp_init()
{
  hmlp::rt.Init();
};

void hmlp_run()
{
  hmlp::rt.Run();
};

void hmlp_finalize()
{
  hmlp::rt.Finalize();
};

