def chunk_tasks(n_tasks, n_batches, arr=None, args=None, start_idx=0):
    """Split the tasks into some number of batches to sent out to MPI workers.
    Parameters
    ----------
    n_tasks : int
        The total number of tasks to divide.
    n_batches : int
        The number of batches to split the tasks into. Often, you may want to do
        ``n_batches=pool.size`` for equal sharing amongst MPI workers.
    arr : iterable (optional)
        Instead of returning indices that specify the batches, you can also
        directly split an array into batches.
    args : iterable (optional)
        Other arguments to add to each task.
    start_idx : int (optional)
        What index in the tasks to start from?
    """
    if args is None:
        args = []
    args = list(args)

    tasks = []
    if n_batches > 0 and n_tasks > n_batches:
        # chunk by the number of batches, often the pool size
        base_chunk_size = n_tasks // n_batches
        rmdr = n_tasks % n_batches

        i1 = start_idx
        for i in range(n_batches):
            i2 = i1 + base_chunk_size
            if i < rmdr:
                i2 += 1

            if arr is None: # store indices
                tasks.append([(i1, i2), i1] + args)

            else: # store sliced array
                tasks.append([arr[i1:i2], i1] + args)

            i1 = i2

    else:
        if arr is None: # store indices
            tasks.append([(start_idx, n_tasks+start_idx), start_idx] + args)

        else: # store sliced array
            tasks.append([arr[start_idx:n_tasks+start_idx], start_idx] + args)

    return tasks
