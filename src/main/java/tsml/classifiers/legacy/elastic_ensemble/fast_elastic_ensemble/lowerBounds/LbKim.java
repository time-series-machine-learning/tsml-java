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
 * A class to compute Lb Kim for DTW distance. Slightly modified from the original paper.
 * @inproceedings{kim2001index,
 *   title={An index-based approach for similarity search supporting time warping in large sequence databases},
 *   author={Kim, Sang-Wook and Park, Sanghyun and Chu, Wesley W},
 *   booktitle={Proceedings 17th International Conference on Data Engineering},
 *   pages={607--614},
 *   year={2001},
 *   organization={IEEE}
 * }
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LbKim {
    public static double distance(final Instance query, final Instance reference) {
        double maxQ = query.value(0), maxR = reference.value(0);
        double minQ = query.value(0), minR = reference.value(0);

        final double diffFirsts = maxQ - maxR;
        final double diffLasts = query.value(query.numAttributes() - 2) - reference.value(reference.numAttributes() - 2);
        double minDist = diffFirsts * diffFirsts + diffLasts * diffLasts;

        boolean minFirstLastQ = true, minFirstLastR = true;
        boolean maxFirstLastQ = true, maxFirstLastR = true;
        for (int i = 1; i < query.numAttributes() - 1; i++) {
            if (query.value(i) > maxQ) {
                maxQ = query.value(i);
                maxFirstLastQ = false;
            } else if (query.value(i) < minQ) {
                minQ = query.value(i);
                minFirstLastQ = false;
            }

            if (reference.value(i) > maxR) {
                maxR = reference.value(i);
                maxFirstLastR = false;
            } else if (reference.value(i) < minR) {
                minR = reference.value(i);
                minFirstLastR = false;
            }
        }

        if (!(minFirstLastQ && minFirstLastR)) {
            final double diffMin = minQ - minR;
            minDist += diffMin * diffMin;
        }
        if (!(maxFirstLastQ && maxFirstLastR)) {
            final double diffMax = maxQ - maxR;
            minDist += diffMax * diffMax;
        }

        return minDist;
    }

    public static double distance(final Instance query, final Instance reference,
                                  final SequenceStatsCache queryCache, final SequenceStatsCache referenceCache,
                                  final int indexQuery, final int indexReference) {
        final double diffFirsts = query.value(0) - reference.value(0);
        final double diffLasts = query.value(query.numAttributes() - 2) - reference.value(reference.numAttributes() - 2);
        double minDist = diffFirsts * diffFirsts + diffLasts * diffLasts;

        if (!queryCache.isMinFirst(indexQuery) && !referenceCache.isMinFirst(indexReference) &&
                !queryCache.isMinLast(indexQuery) && !referenceCache.isMinLast(indexReference)) {
            final double diffMin = queryCache.getMin(indexQuery) - referenceCache.getMin(indexReference);
            minDist += diffMin * diffMin;
        }
        if (!queryCache.isMaxFirst(indexQuery) && !referenceCache.isMaxFirst(indexReference) &&
                !queryCache.isMaxLast(indexQuery) && !referenceCache.isMaxLast(indexReference)) {
            final double diffMax = queryCache.getMax(indexQuery) - referenceCache.getMax(indexReference);
            minDist += diffMax * diffMax;
        }

        return minDist;
    }

    public static double distance(final Instance query, final Instance reference, final SequenceStatsCache cache,
                                  final int indexQuery, final int indexReference) {
        final double diffFirsts = query.value(0) - reference.value(0);
        final double diffLasts = query.value(query.numAttributes() - 2) - reference.value(reference.numAttributes() - 2);
        double minDist = diffFirsts + diffLasts;

        if (!cache.isMinFirst(indexQuery) && !cache.isMinFirst(indexReference) &&
                !cache.isMinLast(indexQuery) && !cache.isMinLast(indexReference)) {
            final double diffMin = cache.getMin(indexQuery) - cache.getMin(indexReference);
            minDist += diffMin * diffMin;
        }
        if (!cache.isMaxFirst(indexQuery) && !cache.isMaxFirst(indexReference) &&
                !cache.isMaxLast(indexQuery) && !cache.isMaxLast(indexReference)) {
            final double diffMax = cache.getMax(indexQuery) - cache.getMax(indexReference);
            minDist += diffMax * diffMax;
        }

        return minDist;
    }
}
