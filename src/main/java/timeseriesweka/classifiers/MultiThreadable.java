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
package timeseriesweka.classifiers;

/**
 * Interface that allows the user to allow a classifier to use multiple threads, how this happens is determined by the
 * classifier.
 * 
 * Known classifiers: AbstractEnsemble
 * 
 * @author pfm15hbu, James Large (james.large@uea.ac.uk) 
 */
public interface MultiThreadable {
    
    /**
     * Sets how many threads this object may spawn. 
     */
    void setThreadAllowance(int numThreads);
    
    /**
     * Details how many threads this object could actually make use of and 
     * gain benefit from. 
     * 
     * This might be a fixed number, or it might change depending on other parameters
     * of the object outside this interface (e.g. an ensemble that spawns a thread
     * for each classifier might have a numUtilisableThreads equal to the number
     * of base classifiers). 
     * 
     * Intended usage is to more intelligently allocate spare threads to different
     * (sub-)processes. This is a utility tool rather than a precise measurement, 
     * there may be diminishing returns for additional threads etc.
     */
    int getNumUtilisableThreads();
}
