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
import utilities.ArrayUtilities;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public interface Ensembler extends Serializable {
    List<Double> weightVotes(Iterable<EnhancedAbstractClassifier> benchmarks);

    interface Scorer extends Function<EnhancedAbstractClassifier, Double>, Serializable {

    }

    static Ensembler byScore(Scorer scorer) {
        return (benchmarks) -> {
            List<Double> weights = new ArrayList<>();
            for(EnhancedAbstractClassifier benchmark : benchmarks) {
                weights.add(scorer.apply(benchmark));
            }
            ArrayUtilities.normalise(weights);
            return weights;
        };
    }

    static Ensembler single() {
        return (benchmarks) -> {
            if(CollectionUtils.size(benchmarks) != 1) {
                throw new IllegalArgumentException("was only expecting 1 benchmark");
            }
            return new ArrayList<>(java.util.Collections.singletonList(1.0));
        };
    }
}
