def example():
    ENVIRONMENT = "test"
    PS = 0                                  # Parameter servers
    N_WORKERS = 16                          # Parallel workers
    N_DROPPED = 2                           # Dropped gradients
    N_AGG = N_WORKERS - N_DROPPED           # Gradients to aggregate
    for p in range(PS):
        print("Starting Parameter Server", p)
        start_process(ENVIRONMENT, "ps", p, N_WORKERS, N_AGG, PS)
    for w in range(N_WORKERS):
        print("Starting Worker", w)
        start_process(ENVIRONMENT, "worker", w, N_WORKERS, N_AGG, PS)
        