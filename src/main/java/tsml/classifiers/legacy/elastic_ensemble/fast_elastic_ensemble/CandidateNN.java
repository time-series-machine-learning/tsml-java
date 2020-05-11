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
package tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble;

/**
 * This is a class for a candidate nearest neighbour that is used in FastEE.
 * It stores some of the meta data of the nearest neighbour that will make the NN search more efficient.
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class CandidateNN {
    public enum Status {
        NN,                         // This is the Nearest Neighbour
        BC,                         // Best Candidate so far
    }

    public int index;               // Index of the sequence in train[]
    public int r;                   // Window validity
    public double distance;         // Computed lower bound

    private Status status;

    public CandidateNN() {
        this.index = Integer.MIN_VALUE;                 // Will be an invalid, negative, index.
        this.r = Integer.MAX_VALUE;						// Max: stands for "haven't found yet"
        this.distance = Double.POSITIVE_INFINITY;       // Infinity: stands for "not computed yet".
        this.status = Status.BC;                        // By default, we don't have any found NN.
    }

    public void set(final int index, final int r, final double distance, final Status status) {
        this.index = index;
        this.r = r;
        this.distance = distance;
        this.status = status;
    }

    public void set(final int index, final double distance, final Status status) {
        this.index = index;
        this.r = -1;
        this.distance = distance;
        this.status = status;
    }

    public boolean isNN() {
        return this.status == Status.NN;
    }

    @Override
    public String toString() {
        return "" + this.index;
    }

    @Override
    public boolean equals(final Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        CandidateNN that = (CandidateNN) o;

        return index == that.index;
    }

    public int compareTo(CandidateNN potentialNN) {
        return Double.compare(this.distance, potentialNN.distance);
    }
}
