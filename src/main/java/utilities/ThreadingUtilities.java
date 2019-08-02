/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package utilities;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Some utility methods for threading, currently assumes that all threaded jobs
 * are independent etc. 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ThreadingUtilities {

    public static ExecutorService buildExecutorService(int numThreads) { 
        //todo look into queues etc
        return Executors.newFixedThreadPool(numThreads);
    }
        
    public static void shutdownExecutor(ExecutorService executor) {
        //todo maybe add timer to while, for general expected usecase in this codebase 
        //this should be fine though
        executor.shutdown();
        while (!executor.isTerminated()) { }
    }
    
    /**
     * Submits all given jobs that each return an object to the executor, waits for them 
     * all to finish and returns all the results. The returned list of results is 
     * in the same order as the jobs, i.e. results.get(0) is the results for jobs.get(0), 
     * etc.
     */
    public static <T> List<T> computeAll(ExecutorService executor, List<Callable<T>> jobs, boolean shutdownExecutorOnCompletion) throws InterruptedException, ExecutionException {
        List<T> results = gatherAll(submitAll(executor, jobs));
        
        if (shutdownExecutorOnCompletion)
            shutdownExecutor(executor);
        
        return results;
    }
    
    /**
     * Submits all given jobs that do NOT return an object to the executor, wait for them all to 
     * finish and returns any Exceptions thrown in a list parallel with the jobs. This can be inspected
     * for failed executions if desired. If there was no exception and the job completed successfully, 
     * the Exception will be null
     */
    public static List<Exception> runAll(ExecutorService executor, List<Runnable> jobs, boolean shutdownExecutorOnCompletion) throws InterruptedException, ExecutionException {
        List<Future<Exception>> futureResults = new ArrayList<>();
        
        for (Runnable job : jobs) {
            Callable<Exception> wrappedJob = () -> {
                try {
                    job.run();
                    return null;
                } catch (Exception e) {
                    return e;
                }
            };
            
            futureResults.add(executor.submit(wrappedJob));
        }
        
        List<Exception> results = gatherAll(futureResults);
        
        if (shutdownExecutorOnCompletion)
            shutdownExecutor(executor);
        
        return results;
    }
    
    public static <T> List<Future<T>> submitAll(ExecutorService executor, List<Callable<T>> jobs) throws InterruptedException, ExecutionException {
        List<Future<T>> futureResults = new ArrayList<>();
        for (Callable<T> job : jobs)
            futureResults.add(executor.submit(job));
        return futureResults;
    }
    
    public static <T> List<T> gatherAll(List<Future<T>> futures) throws InterruptedException, ExecutionException {
        List<T> results = new ArrayList<>();
        
        for (Future<T> future : futures)
            results.add(future.get());
        
        return results;
    }

}
