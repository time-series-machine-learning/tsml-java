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
 
package tsml.classifiers.distance_based.knn.neighbour_iteration;


import tsml.classifiers.distance_based.knn.KNN;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;

import java.util.Iterator;

public class RandomNeighbourIteratorBuilder
 extends LinearNeighbourIteratorBuilder {

    public RandomNeighbourIteratorBuilder() {}

    public RandomNeighbourIteratorBuilder(KNNLOOCV knn) {
        super(knn);
    }

    @Override
    public Iterator<KNN.NeighbourSearcher> build() {
        if(knn == null) throw new IllegalStateException("knn not set");
        // set this to a random seed sourced from the knn's rng
        return new RandomIterator<>(knn.getRandom(), knn.getSearchers());
    }


}
