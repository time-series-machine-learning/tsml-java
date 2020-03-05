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

import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.WarpingPathResults;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.LCSSDistance;
import weka.core.Instance;

/**
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LazyAssessNN_LCSS extends LazyAssessNN {
    private double currentEpsilon;
    private int currentDelta;
    private int minWindowValidity;

    public LazyAssessNN_LCSS(final SequenceStatsCache cache) {
        super(cache);
    }

    public LazyAssessNN_LCSS(final Instance query, final int index,
                             final Instance reference, final int indexReference,
                             final SequenceStatsCache cache) {
        super(query, index, reference, indexReference, cache);
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    public void set(final Instance query, final int index, final Instance reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        minWindowValidity = 0;
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

    private void tryContinueLBLCSS(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double ub = Math.abs(1.0 - scoreToBeat) * length;
        final double[] LEQ = cache.getLE(indexQuery, currentDelta, currentEpsilon);
        final double[] UEQ = cache.getUE(indexQuery, currentDelta, currentEpsilon);
        double lcs = 0;
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && lcs > ub) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (reference.value(index) <= UEQ[index] && reference.value(index) >= LEQ[index]) {
                lcs++;
            }
            indexStoppedLB++;
        }
        this.minDist = 1 - lcs / length;
    }

    private void tryFullLBLCSS() {
        final int length = query.numAttributes() - 1;
        final double[] LEQ = cache.getLE(indexQuery, currentDelta, currentEpsilon);
        final double[] UEQ = cache.getUE(indexQuery, currentDelta, currentEpsilon);
        double lcs = 0;
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (reference.value(index) <= UEQ[index] && reference.value(index) >= LEQ[index]) {
                lcs++;
            }
            indexStoppedLB++;
        }
        this.minDist = 1 - lcs / length;
    }

    private void setCurrentDeltaAndEpsilon(final int delta, final double epsilon) {
        if (this.currentEpsilon != epsilon) {
            this.currentEpsilon = epsilon;
            this.currentDelta = delta;
            this.minDist = 0.0;
            this.bestMinDist = minDist;
            indexStoppedLB = oldIndexStoppedLB = 0;
            this.status = LBStatus.Previous_LB_LCSS;
        } else if (this.currentDelta != delta) {
            this.currentDelta = delta;
            if (status == LBStatus.Full_LCSS) {
                if (this.currentDelta < minWindowValidity) {
                    this.status = LBStatus.Previous_LCSS;
                }
            } else {
                this.status = LBStatus.Previous_LB_LCSS;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final int delta, final double epsilon) {
        setCurrentDeltaAndEpsilon(delta, epsilon);
        switch (status) {
            case None:
            case Previous_LB_LCSS:
            case Previous_LCSS:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_LCSS:
                tryFullLBLCSS();
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist > scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_LCSS;
                    else status = LBStatus.Full_LB_LCSS;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_LCSS;
            case Full_LB_LCSS:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = LCSSDistance.distanceExt(query, reference, epsilon, delta);
                minDist = res.distance;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_LCSS;
                minWindowValidity = res.distanceFromDiagonal;
            case Full_LCSS:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    public double getDistance(final int window) {
        if ((status == LBStatus.Full_LCSS) && minWindowValidity <= window) {
            return minDist;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is argMin3 valid already-computed Distance");
    }

    public int getMinWindowValidityForFullDistance() {
        if (status == LBStatus.Full_LCSS) {
            return minWindowValidity;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is argMin3 valid already-computed Distance");
    }

    public int getMinwindow() {
        return minWindowValidity;
    }

    public void setMinwindow(final int w) {
        minWindowValidity = w;
    }

    @Override
    public void setFullDistStatus() {
        this.status = LBStatus.Full_LCSS;
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        switch (status) {
            // LCSS
            case Full_LCSS:
            case Full_LB_LCSS:
                return thisD / (query.numAttributes() - 1);
            case Partial_LB_LCSS:
                return thisD / indexStoppedLB;
            case Previous_LCSS:
                return 0.8 * thisD / (query.numAttributes() - 1);
            case Previous_LB_LCSS:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
