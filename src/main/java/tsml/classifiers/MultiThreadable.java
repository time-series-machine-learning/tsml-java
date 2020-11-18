/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.classifiers;

/**
 * Interface that allows the user to allow a classifier to use multiple threads, how this happens is determined by the
 * classifier. Exact API for this and how threading is handled codebase-wide is still to be decided 02/08/2019
 * 
 * Known classifiers: AbstractEnsemble, BOSS, cBOSS, BOSSIndividual, MultiSamplingEvaluator
 * 
 * @author Matthew Middlehurst, James Large (james.large@uea.ac.uk)
 */
public interface MultiThreadable {
    
    /**
     * Enables multithreading, and allows the class to spawn numThreads threads
     */
    void enableMultiThreading(int numThreads);
    
    /**
     * Enables multithreading, and allows the class to spawn a number of threads equal to the number of available
     * processors minus one.
     */
    default void enableMultiThreading() {
        enableMultiThreading(Runtime.getRuntime().availableProcessors()-1);
    }
}
