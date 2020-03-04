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
 * This class stores some results from computing DTW.
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */


public class WarpingPathResults {
    public double distance;
    public int distanceFromDiagonal; //The smallest window that would give the same distance

    public WarpingPathResults() {
    }

    public WarpingPathResults(double d, int distanceFromDiagonal) {
        this.distance = d;
        this.distanceFromDiagonal = distanceFromDiagonal;
    }
}
