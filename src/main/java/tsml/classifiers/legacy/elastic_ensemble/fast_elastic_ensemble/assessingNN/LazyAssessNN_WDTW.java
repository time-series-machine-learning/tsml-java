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
package tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.assessingNN;

import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.WeightedDTW;
import weka.core.Instance;

/**
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LazyAssessNN_WDTW extends LazyAssessNN {
    private double[] currentWeightVector;

    public LazyAssessNN_WDTW(final SequenceStatsCache cache) {
        super(cache);
    }

    public LazyAssessNN_WDTW(final Instance query, final int index,
                             final Instance reference, final int indexReference,
                             final SequenceStatsCache cache) {
        super(query, index, reference, indexReference, cache);
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    public void set(final Instance query, final int index, final Instance reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        // --- From constructor
        if (index < indexReference) {
            this.query = query;
            this.indexQuery = index;
            this.reference = reference;
            this.indexReference = indexReference;
        } else {
            this.query = reference;
            this.indexQuery = indexReference;
            this.reference = query;
            this.indexReference = index;
        }
        this.minDist = 0.0;
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    private void tryContinueLBWDTWQR(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double QMAX = cache.getMax(indexQuery);
        final double QMIN = cache.getMin(indexQuery);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < QMIN) {
                final double diff = QMIN - c;
                minDist += diff * diff * currentWeightVector[0];
            } else if (QMAX < c) {
                final double diff = QMAX - c;
                minDist += diff * diff * currentWeightVector[0];
            }
            indexStoppedLB++;
        }
    }

    private void tryContinueLBWDTWRQ(final double scoreToBeat) {
        final int length = reference.numAttributes() - 1;
        final double QMAX = cache.getMax(indexReference);
        final double QMIN = cache.getMin(indexReference);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < QMIN) {
                final double diff = QMIN - c;
                minDist += diff * diff * currentWeightVector[0];
            } else if (QMAX < c) {
                final double diff = QMAX - c;
                minDist += diff * diff * currentWeightVector[0];
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBWDTWQR() {
        final int length = query.numAttributes() - 1;
        final double QMAX = cache.getMax(indexQuery);
        final double QMIN = cache.getMin(indexQuery);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < QMIN) {
                final double diff = QMIN - c;
                minDist += diff * diff;
            } else if (QMAX < c) {
                final double diff = QMAX - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
        this.minDist *= currentWeightVector[0];
    }

    private void tryFullLBWDTWRQ() {
        final int length = reference.numAttributes() - 1;
        final double QMAX = cache.getMax(indexReference);
        final double QMIN = cache.getMin(indexReference);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < QMIN) {
                final double diff = QMIN - c;
                minDist += diff * diff;
            } else if (QMAX < c) {
                final double diff = QMAX - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
        this.minDist *= currentWeightVector[0];
    }

    private void setCurrentWeightVector(final double[] weightVector) {
        this.currentWeightVector = weightVector;
        if (status == LBStatus.Full_WDTW) {
            this.status = LBStatus.Previous_WDTW;
        } else {
            this.status = LBStatus.Previous_LB_WDTW;
            this.oldIndexStoppedLB = indexStoppedLB;
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final double[] weightVector) {
        setCurrentWeightVector(weightVector);
        switch (status) {
            case None:
            case Previous_LB_WDTW:
            case Previous_WDTW:
                if (bestMinDist * weightVector[0] >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_WDTWQR:
                tryContinueLBWDTWQR(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_WDTWQR;
                    else status = LBStatus.Full_LB_WDTWQR;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_WDTWQR;
            case Full_LB_WDTWQR:
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_WDTWRQ:
                tryContinueLBWDTWRQ(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_WDTWRQ;
                    else status = LBStatus.Full_LB_WDTWRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_WDTWRQ;
            case Full_LB_WDTWRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = WeightedDTW.distance(query, reference, weightVector);
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_WDTW;
            case Full_WDTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    public double getDistance() {
        return minDist;
    }

    @Override
    public void setFullDistStatus() {
        this.status = LBStatus.Full_WDTW;
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        switch (status) {
            // WDTW
            case Full_WDTW:
            case Full_LB_WDTWQR:
            case Full_LB_WDTWRQ:
                return thisD / (query.numAttributes() - 1);
            case Partial_LB_WDTWQR:
            case Partial_LB_WDTWRQ:
                return thisD / indexStoppedLB;
            case Previous_WDTW:
                return 0.8 * thisD / (query.numAttributes() - 1);
            case Previous_LB_WDTW:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
