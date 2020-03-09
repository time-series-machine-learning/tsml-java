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
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.ERPDistance;
import weka.core.Instance;

/**
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LazyAssessNN_ERP extends LazyAssessNN {
    private double currentG;
    private double currentBandSize;
    private int minWindowValidity;

    public LazyAssessNN_ERP(final SequenceStatsCache cache) {
        super(cache);
    }

    public LazyAssessNN_ERP(final Instance query, final int index,
                            final Instance reference, final int indexReference,
                            final SequenceStatsCache cache) {
        super(query, index, reference, indexReference, cache);
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    public void set(final Instance query, final int index, final Instance reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        currentG = 0;
        currentBandSize = 0;
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

    void tryContinueLBERPQR(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double[] LEQ = cache.getLE(indexQuery, currentG, currentBandSize);
        final double[] UEQ = cache.getUE(indexQuery, currentG, currentBandSize);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
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

    private void tryContinueLBERPRQ(final double scoreToBeat) {
        final int length = reference.numAttributes() - 1;
        final double[] LER = cache.getLE(indexReference, currentG, currentBandSize);
        final double[] UER = cache.getUE(indexReference, currentG, currentBandSize);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
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

    private void tryFullLBERPQR() {
        final int length = query.numAttributes() - 1;
        final double[] LEQ = cache.getLE(indexQuery, currentG, currentBandSize);
        final double[] UEQ = cache.getUE(indexQuery, currentG, currentBandSize);
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

    private void tryFullLBERPRQ() {
        final int length = reference.numAttributes() - 1;
        final double[] LER = cache.getLE(indexReference, currentG, currentBandSize);
        final double[] UER = cache.getUE(indexReference, currentG, currentBandSize);
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


    private void setCurrentGandBandSize(final double g, final double bandSize) {
        if (this.currentG != g) {
            this.currentBandSize = bandSize;
            this.currentG = g;
            this.minDist = 0.0;
            this.bestMinDist = minDist;
            indexStoppedLB = oldIndexStoppedLB = 0;
            this.status = LBStatus.Previous_G_LB_ERP;
        } else if (this.currentBandSize != bandSize) {
            this.currentBandSize = bandSize;
            if (status == LBStatus.Full_ERP) {
                if (this.currentBandSize < minWindowValidity) {
                    this.status = LBStatus.Previous_Band_ERP;
                }
            } else {
                this.status = LBStatus.Previous_Band_LB_ERP;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final double g, final double bandSize) {
        setCurrentGandBandSize(g, bandSize);
        switch (status) {
            case None:
            case Previous_G_LB_ERP:
            case Previous_Band_LB_ERP:
            case Previous_Band_ERP:
            case LB_Kim:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_ERPQR:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBERPQR(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_ERPQR;
                    else status = LBStatus.Full_LB_ERPQR;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_ERPQR;
            case Full_LB_ERPQR:
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_ERPRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBERPRQ(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < reference.numAttributes() - 1) status = LBStatus.Partial_LB_ERPRQ;
                    else status = LBStatus.Full_LB_ERPRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_ERPRQ;
            case Full_LB_ERPRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = ERPDistance.distanceExt(query, reference, currentG, currentBandSize);
                minDist = res.distance;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_ERP;
                minWindowValidity = res.distanceFromDiagonal;
            case Full_ERP:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    public double getDistance(final int window) {
        if ((status == LBStatus.Full_ERP) && minWindowValidity <= window) {
            return minDist;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is argMin3 valid already-computed Distance");
    }

    public int getMinWindowValidityForFullDistance() {
        if (status == LBStatus.Full_ERP) {
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
        this.status = LBStatus.Full_ERP;
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        switch (status) {
            // ERP
            case Full_ERP:
            case Full_LB_ERPQR:
            case Full_LB_ERPRQ:
                return thisD / (query.numAttributes() - 1);
            case Partial_LB_ERPQR:
            case Partial_LB_ERPRQ:
                return thisD / indexStoppedLB;
            case Previous_Band_ERP:
                return 0.8 * thisD / (query.numAttributes() - 1);
            case Previous_G_LB_ERP:
            case Previous_Band_LB_ERP:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
