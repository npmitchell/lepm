from timeit import default_timer as timer
import sys

'''Custom functions for handling timing of function execution
'''


def timefunc(func, *args, **kwargs):
    """Time a function.

    Parameters
    ----------
    func : function
        the function to time, which accepts arguments *args and **kwargs. Note that the function cannot have a kwarg
        called 'iterations'
    *args : arguments passed to func
        arguments passed to func
    **kwargs : keyword arguments passed to func
        keyword arguments passed to func
    iterations : int or None (default=3)
        The number of times to run the function, to get best and worst time (similar to timeit)

    Returns
    -------
    result : output of func
    shortest : float
        the shortest time to execute, out of 'iterations' iterations
    longest : float
        the longest time to execute, out of 'iterations' iterations
    """
    try:
        iterations = kwargs.pop('iterations')
    except KeyError:
        iterations = 3
    shortest = sys.maxsize
    longest = 0.0
    for _ in range(iterations):
        start = timer()
        result = func(*args, **kwargs)
        shortest = min(timer() - start, shortest)
        longest = max(timer() - start, longest)
    print('Best of {} {}(): {:.9f}'.format(iterations, func.__name__, shortest) + ', ' +
          'Worst of {} {}(): {:.9f}'.format(iterations, func.__name__, longest))
    return result, shortest, longest
