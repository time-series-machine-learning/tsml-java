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
import weka.core.Instance;

/**
 * A super class to access if a query is the nearest neighbour of a reference time series, given a cache.
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public abstract class LazyAssessNN implements Comparable<LazyAssessNN> {
    public int indexQuery;
    public int indexReference;              // Index for query and reference
    SequenceStatsCache cache;               // Cache to store the information for the sequences
    Instance query, reference;              // Query and reference sequences
    int indexStoppedLB, oldIndexStoppedLB;  // Index where we stop LB

    double minDist;                         // distance
    double bestMinDist;                     // best so far distance

    LBStatus status;                        // Status of Lower Bound

    public LazyAssessNN(final Instance query, final int index,
                        final Instance reference, final int indexReference,
                        final SequenceStatsCache cache) {
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
        this.cache = cache;
        this.bestMinDist = minDist;
    }

    public LazyAssessNN(final SequenceStatsCache cache) {
        this.cache = cache;
    }

    public abstract void set(final Instance query, final int index, final Instance reference, final int indexReference);

    public void setBestMinDist(final double bestMinDist) {
        this.bestMinDist = bestMinDist;
    }

    public double getBestMinDist() {
        return this.bestMinDist;
    }

    @Override
    public String toString() {
        return "" + indexQuery + " - " + indexReference + " - " + bestMinDist;
    }

    public int getOtherIndex(final int index) {
        if (index == indexQuery) {
            return indexReference;
        } else {
            return indexQuery;
        }
    }

    public Instance getSequenceForOtherIndex(final int index) {
        if (index == indexQuery) {
            return reference;
        } else {
            return query;
        }
    }

    public double getMinDist() {
        return minDist;
    }

    public void setMinDist(final double minDist) {
        this.minDist = minDist;
    }

    @Override
    public int compareTo(final LazyAssessNN o) {
        return this.compare(o);
    }

    private int compare(final LazyAssessNN o) {
        double num1 = this.getDoubleValueForRanking();
        double num2 = o.getDoubleValueForRanking();
        return Double.compare(num1, num2);
    }

    public abstract double getDoubleValueForRanking();

    @Override
    public boolean equals(final Object o) {
        LazyAssessNN d = (LazyAssessNN) o;
        return (this.indexQuery == d.indexQuery && this.indexReference == d.indexReference);
    }

    public LBStatus getStatus() {
        return status;
    }

    public abstract void setFullDistStatus();

    public double getBestLB() {
        return bestMinDist;
    }

    public Instance getQuery() {
        return query;
    }

    public Instance getReference() {
        return reference;
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Internal types
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    public enum RefineReturnType {
        Pruned_with_LB, Pruned_with_Dist, New_best
    }

    public enum LBStatus {
        None, LB_Kim,
        Partial_LB_KeoghQR, Full_LB_KeoghQR, Partial_LB_KeoghRQ, Full_LB_KeoghRQ,   // DTW
        Previous_LB_DTW, Previous_DTW, Full_DTW, Partial_DTW,                       // DTW
        Partial_LB_WDTWQR, Partial_LB_WDTWRQ, Full_LB_WDTWQR, Full_LB_WDTWRQ,       // WDTW
        Previous_LB_WDTW, Previous_WDTW, Full_WDTW,                                 // WDTW
        Partial_LB_MSM, Full_LB_MSM, Previous_LB_MSM, Previous_MSM, Full_MSM,       // MSM
        Partial_LB_ERPQR, Partial_LB_ERPRQ, Full_LB_ERPQR, Full_LB_ERPRQ,           // ERP
        Previous_G_LB_ERP, Previous_Band_LB_ERP, Previous_Band_ERP, Full_ERP,       // ERP
        Partial_LB_TWE, Full_LB_TWE, Previous_LB_TWE, Previous_TWE, Full_TWE,       // TWE
        Partial_LB_LCSS, Full_LB_LCSS, Previous_LB_LCSS, Previous_LCSS, Full_LCSS   // LCSS
    }
}
