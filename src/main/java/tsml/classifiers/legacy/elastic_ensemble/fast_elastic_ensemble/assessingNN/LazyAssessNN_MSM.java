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
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.MSMDistance;
import weka.core.Instance;

/**
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LazyAssessNN_MSM extends LazyAssessNN {
    private double currentC;
    private MSMDistance distanceComputer = new MSMDistance();

    public LazyAssessNN_MSM(final SequenceStatsCache cache) {
        super(cache);
    }

    public LazyAssessNN_MSM(final Instance query, final int index,
                            final Instance reference, final int indexReference,
                            final SequenceStatsCache cache) {
        super(query, index, reference, indexReference, cache);
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    public void set(final Instance query, final int index, final Instance reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        currentC = 0;
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

    private void tryContinueLBMSM(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double QMAX = cache.getMax(indexQuery);
        final double QMIN = cache.getMin(indexQuery);
        this.minDist = Math.abs(query.value(0) - reference.value(0));
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist < scoreToBeat) {
            int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (index > 0 && index < length - 1) {
                final double curr = reference.value(index);
                final double prev = reference.value(index - 1);
                if (prev <= curr && curr < QMIN) {
                    minDist += Math.min(Math.abs(reference.value(index) - QMIN), this.currentC);
                } else if (prev >= curr && curr >= QMAX) {
                    minDist += Math.min(Math.abs(reference.value(index) - QMAX), this.currentC);
                }
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBMSM() {
        final int length = query.numAttributes() - 1;
        final double QMAX = cache.getMax(indexQuery);
        final double QMIN = cache.getMin(indexQuery);
        this.minDist = Math.abs(query.value(0) - reference.value(0));
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (index > 0 && index < length - 1) {
                final double curr = reference.value(index);
                final double prev = reference.value(index - 1);
                if (prev <= curr && curr < QMIN) {
                    minDist += Math.min(Math.abs(reference.value(index) - QMIN), this.currentC);
                } else if (prev >= curr && curr >= QMAX) {
                    minDist += Math.min(Math.abs(reference.value(index) - QMAX), this.currentC);
                }
            }
            indexStoppedLB++;
        }
    }

    private void setCurrentC(final double c) {
        if (this.currentC != c) {
            this.currentC = c;
            if (status == LBStatus.Full_MSM) {
                this.status = LBStatus.Previous_MSM;
            } else {
                this.status = LBStatus.Previous_LB_MSM;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final double c) {
        setCurrentC(c);
        switch (status) {
            case None:
            case Previous_LB_MSM:
            case Previous_MSM:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_MSM:
                tryContinueLBMSM(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_MSM;
                    else status = LBStatus.Full_LB_MSM;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_MSM;
            case Full_LB_MSM:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                distanceComputer.setC(currentC);
                minDist = distanceComputer.distance(query, reference);
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_MSM;
            case Full_MSM:
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
        this.status = LBStatus.Full_MSM;
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        switch (status) {
            // MSM
            case Full_MSM:
            case Full_LB_MSM:
                return thisD / (query.numAttributes() - 1);
            case Partial_LB_MSM:
                return thisD / indexStoppedLB;
            case Previous_MSM:
                return 0.8 * thisD / (query.numAttributes() - 1);
            case Previous_LB_MSM:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
