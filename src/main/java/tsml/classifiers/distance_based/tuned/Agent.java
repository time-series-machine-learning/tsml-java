/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.tuned;

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.collections.iteration.DefaultIterator;

import java.util.Set;

/*
    Explores the tuning space (usually a parameter space). It is an iterator selects the next classifier to examine.
    The classifier is examined by the IncTuner and returned using the feedback function, i.e. reinforcement learning
    style.
 */
public interface Agent extends DefaultIterator<EnhancedAbstractClassifier> {
    default long predictNextTimeNanos() {
        return -1;
    }
    Set<EnhancedAbstractClassifier> getFinalClassifiers();

    boolean feedback(EnhancedAbstractClassifier classifier); // true == can be exploited (again)

    default boolean isExploringOrExploiting() { // must be called after hasNext !!
        return true; // explore == true; exploit == false
    }

}
