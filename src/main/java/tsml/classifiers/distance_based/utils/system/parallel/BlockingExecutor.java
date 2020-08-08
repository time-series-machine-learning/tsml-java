package tsml.classifiers.distance_based.utils.system.parallel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Purpose: run several tasks concurrently with no queuing / queue overflow
 * <p>
 * Contributors: goastler
 */
public class BlockingExecutor implements ExecutorService {

    final Semaphore semaphore;
    final ExecutorService service;

    public Executor getService() {
        return service;
    }

    public BlockingExecutor(final int concurrentTasksLimit, final ExecutorService service) {
        semaphore = new Semaphore(concurrentTasksLimit);
        this.service = service;
    }

    public BlockingExecutor(ThreadPoolExecutor service) {
        int maximumPoolSize = service.getMaximumPoolSize();
        this.semaphore = new Semaphore(maximumPoolSize);
        this.service = service;
    }

    private void acquire() {
        try {
            semaphore.acquire();
        } catch (InterruptedException e) {
            IllegalStateException ex = new IllegalStateException();
            ex.addSuppressed(e);
            throw ex;
        }
    }

    private boolean tryAcquire(long time, TimeUnit unit) {
        try {
            return semaphore.tryAcquire(time, unit);
        } catch (InterruptedException e) {
            IllegalStateException ex = new IllegalStateException();
            ex.addSuppressed(e);
            throw ex;
        }
    }

    private <A> Callable<A> acquireThenReleaseWrap(Callable<A> callable) {
        acquire();
        return releaseWrap(callable);
    }

    private <A> Callable<A> releaseWrap(Callable<A> callable) {
        return () -> {
            A result;
            try {
                result = callable.call();
            } finally {
                semaphore.release();
            }
            return result;
        };
    }

    @Override
    public void execute(final Runnable runnable) {
        submit(runnable);
    }

    @Override
    public void shutdown() {
        service.shutdown();
    }

    @Override
    public List<Runnable> shutdownNow() {
        return service.shutdownNow();
    }

    @Override
    public boolean isShutdown() {
        return service.isShutdown();
    }

    @Override
    public boolean isTerminated() {
        return service.isTerminated();
    }

    @Override
    public boolean awaitTermination(final long l, final TimeUnit timeUnit) throws InterruptedException {
        return service.awaitTermination(l, timeUnit);
    }

    @Override
    public <T> Future<T> submit(final Callable<T> callable) {
        return service.submit(acquireThenReleaseWrap(callable));
    }

    @Override
    public <T> Future<T> submit(final Runnable runnable, final T t) {
        return service.submit(acquireThenReleaseWrap(() -> {
            runnable.run();
            return t;
        }));
    }

    @Override
    public Future<?> submit(final Runnable runnable) {
        return service.submit(acquireThenReleaseWrap(() -> {
            runnable.run();
            return null;
        }));
    }

    @Override
    public <T> List<Future<T>> invokeAll(final Collection<? extends Callable<T>> collection)
        throws InterruptedException {
        List<Future<T>> list = new ArrayList<>();
        for(Callable<T> callable : collection) {
            list.add(submit(callable));
        }
        for(Future<T> future : list) {
            try {
                future.get();
            } catch(ExecutionException ignored) {

            }
        }
        return list;
    }

    // only for internal use, returns an empty future in the case a permit couldn't be acquired in the time interval
    private <A> Future<A> submit(Callable<A> callable, long time, TimeUnit unit) throws TimeoutException {
        boolean acquired;
        if(time > 0) {
            acquired = tryAcquire(time, unit);
            if(acquired) {
                callable = releaseWrap(callable);
                return submit(callable);
            }
        }
        throw new TimeoutException();
    }

    @Override
    public <T> List<Future<T>> invokeAll(final Collection<? extends Callable<T>> collection, final long l,
        final TimeUnit timeUnit)
        throws InterruptedException {
        long timestamp = System.nanoTime();
        long timeLimit = TimeUnit.NANOSECONDS.convert(l, timeUnit);
        List<Future<T>> futures = new ArrayList<>();
        for(Callable<T> callable : collection) {
            Future<T> future = null;
            try {
                future = submit(callable, timeLimit - (System.nanoTime() - timestamp), TimeUnit.NANOSECONDS);
            } catch(TimeoutException e) {
                future = new FutureTask<>(callable);
                future.cancel(true);
            }
            futures.add(future);
        }
        for(Future<T> future : futures) {
            long remainingTime = timeLimit - (System.nanoTime() - timestamp);
            if(remainingTime > 0 && !future.isCancelled()) {
                try {
                    future.get(timeLimit - (System.nanoTime() - timestamp), TimeUnit.NANOSECONDS);
                } catch(CancellationException | InterruptedException | ExecutionException | TimeoutException ignored) {

                }
            }
            future.cancel(true);
        }
        return futures;
    }

    @Override
    public <T> T invokeAny(final Collection<? extends Callable<T>> collection)
        throws InterruptedException, ExecutionException {
        for(Callable<T> callable : collection) {
            try {
                Future<T> future = submit(callable);
                T result = future.get();
                if(!future.isCancelled() && future.isDone()) {
                    return result;
                }
            } catch(CancellationException | ExecutionException ignored) {}
        }
        throw new ExecutionException(new IllegalStateException("no task completed."));
    }

    @Override
    public <T> T invokeAny(final Collection<? extends Callable<T>> collection, final long l, final TimeUnit timeUnit)
        throws InterruptedException, ExecutionException, TimeoutException {
        long timestamp = System.nanoTime();
        long timeLimit = TimeUnit.NANOSECONDS.convert(l, timeUnit);
        for(Callable<T> callable : collection) {
            try {
                Future<T> future = submit(callable, timeLimit - (System.nanoTime() - timestamp), TimeUnit.NANOSECONDS);
                T result = future.get(timeLimit - (System.nanoTime() - timestamp), TimeUnit.NANOSECONDS);
                if(!future.isCancelled() && future.isDone()) {
                    return result;
                }
            } catch(CancellationException | ExecutionException ignored) {}
        }
        throw new ExecutionException(new IllegalStateException("no task completed."));
    }
}
