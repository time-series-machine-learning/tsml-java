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
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.DTW;
import weka.core.Instance;

/**
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LazyAssessNN_DTW extends LazyAssessNN {
    private int currentW;                           // Current warping window for DTW
    private int minWindowValidity;                  // Minimum window validity for DTW, ERP, LCSS
    private int nOperationsLBKim;                   // Number of operations for LB Kim
    private double EuclideanDist;                   // euclidean distance

    public LazyAssessNN_DTW(final SequenceStatsCache cache) {
        super(cache);
    }

    public LazyAssessNN_DTW(final Instance query, final int index,
                            final Instance reference, final int indexReference,
                            final SequenceStatsCache cache) {
        super(query, index, reference, indexReference, cache);
        tryLBKim();
        this.bestMinDist = minDist;
        this.status = LBStatus.LB_Kim;
    }

    public void set(final Instance query, final int index, final Instance reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        currentW = 0;
        minWindowValidity = 0;
        nOperationsLBKim = 0;
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
        tryLBKim();
        this.bestMinDist = minDist;
        this.status = LBStatus.LB_Kim;
    }

    private void tryLBKim() {
        final double diffFirsts = query.value(0) - reference.value(0);
        final double diffLasts = query.value(query.numAttributes() - 2) - reference.value(reference.numAttributes() - 2);
        minDist = diffFirsts * diffFirsts + diffLasts * diffLasts;
        nOperationsLBKim = 2;
        if (!cache.isMinFirst(indexQuery) && !cache.isMinFirst(indexReference) && !cache.isMinLast(indexQuery) && !cache.isMinLast(indexReference)) {
            final double diffMin = cache.getMin(indexQuery) - cache.getMin(indexReference);
            minDist += diffMin * diffMin;
            nOperationsLBKim++;
        }
        if (!cache.isMaxFirst(indexQuery) && !cache.isMaxFirst(indexReference) && !cache.isMaxLast(indexQuery) && !cache.isMaxLast(indexReference)) {
            final double diffMax = cache.getMax(indexQuery) - cache.getMax(indexReference);
            minDist += diffMax * diffMax;
            nOperationsLBKim++;
        }

        status = LBStatus.LB_Kim;
    }

    private void tryContinueLBKeoghQR(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double[] LEQ = cache.getLE(indexQuery, currentW);
        final double[] UEQ = cache.getUE(indexQuery, currentW);
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < LEQ[index]) {
                final double diff = LEQ[index] - c;
                minDist += diff * diff;
            } else if (UEQ[index] < c) {
                final double diff = UEQ[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    private void tryContinueLBKeoghRQ(final double scoreToBeat) {
        final int length = reference.numAttributes() - 1;
        final double[] LER = cache.getLE(indexReference, currentW);
        final double[] UER = cache.getUE(indexReference, currentW);
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < LER[index]) {
                final double diff = LER[index] - c;
                minDist += diff * diff;
            } else if (UER[index] < c) {
                final double diff = UER[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBKeoghQR() {
        final int length = query.numAttributes() - 1;
        final double[] LEQ = cache.getLE(indexQuery, currentW);
        final double[] UEQ = cache.getUE(indexQuery, currentW);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < LEQ[index]) {
                final double diff = LEQ[index] - c;
                minDist += diff * diff;
            } else if (UEQ[index] < c) {
                final double diff = UEQ[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBKeoghRQ() {
        final int length = reference.numAttributes() - 1;
        final double[] LER = cache.getLE(indexReference, currentW);
        final double[] UER = cache.getUE(indexReference, currentW);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < LER[index]) {
                final double diff = LER[index] - c;
                minDist += diff * diff;
            } else if (UER[index] < c) {
                final double diff = UER[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    private void setCurrentW(final int currentW) {
        if (this.currentW != currentW) {
            this.currentW = currentW;
            if (this.status == LBStatus.Full_DTW) {
                if (this.currentW < minWindowValidity) {
                    this.status = LBStatus.Previous_DTW;
                }
            } else {
                this.status = LBStatus.Previous_LB_DTW;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final int w) {
        setCurrentW(w);
        switch (status) {
            case Previous_LB_DTW:
            case Previous_DTW:
            case LB_Kim:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_KeoghQR:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBKeoghQR(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_KeoghQR;
                    else status = LBStatus.Full_LB_KeoghQR;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_KeoghQR;
            case Full_LB_KeoghQR:
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_KeoghRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBKeoghRQ(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < reference.numAttributes() - 1) status = LBStatus.Partial_LB_KeoghRQ;
                    else status = LBStatus.Full_LB_KeoghRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_KeoghRQ;
            case Full_LB_KeoghRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = DTW.distanceExt(query, reference, currentW);
                minDist = res.distance;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_DTW;
                minWindowValidity = res.distanceFromDiagonal;
            case Full_DTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    public double getDistance(final int window) {
        if ((status == LBStatus.Full_DTW) && minWindowValidity <= window) {
            return minDist;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is argMin3 valid already-computed Distance");
    }

    public int getMinWindowValidityForFullDistance() {
        if (status == LBStatus.Full_DTW) {
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

    public double getEuclideanDistance() {
        return EuclideanDist;
    }

    @Override
    public void setFullDistStatus() {
        this.status = LBStatus.Full_DTW;
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        switch (status) {
            // DTW
            case Full_DTW:
            case Full_LB_KeoghQR:
            case Full_LB_KeoghRQ:
                return thisD / (query.numAttributes() - 1);
            case LB_Kim:
                return thisD / nOperationsLBKim;
            case Partial_LB_KeoghQR:
            case Partial_LB_KeoghRQ:
                return thisD / indexStoppedLB;
            case Previous_DTW:
                return 0.8 * thisD / (query.numAttributes() - 1);    // DTW(w+1) should be tighter
            case Previous_LB_DTW:
                if (indexStoppedLB == 0) {
                    //lb kim
                    return thisD / nOperationsLBKim;
                } else {
                    //lbkeogh
                    return thisD / oldIndexStoppedLB;
                }
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
