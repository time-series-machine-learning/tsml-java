/* Copyright (C) 2019 Chang Wei Tan
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
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. */
package tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds;

import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import weka.core.Instance;

/**
 * A class to compute LB Yi for DTW distance
 *@inproceedings{yi2000fast,
 *   title={Fast time sequence indexing for arbitrary Lp norms},
 *   author={Yi, Byoung-Kee and Faloutsos, Christos},
 *   booktitle={VLDB},
 *   volume={385},
 *   number={394},
 *   pages={99},
 *   year={2000}
 * }
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LbYi {
    public static double distance(final Instance candidate, final SequenceStatsCache queryCache, final int indexQuery) {
        double lb = 0;

        for (int i = 0; i < candidate.numAttributes()-1; i++) {
            if (candidate.value(i) < queryCache.getMin(indexQuery)) {
                lb += queryCache.getMin(indexQuery);
            } else if (queryCache.getMax(indexQuery) < candidate.value(i)) {
                lb += queryCache.getMax(indexQuery);
            }
        }

        return lb;
    }
}
