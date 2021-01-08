/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.distances;

public abstract class MatrixBasedDistanceMeasure extends BaseDistanceMeasure {

    private boolean generateDistanceMatrix = false;

    public boolean isGenerateDistanceMatrix() {
        return generateDistanceMatrix;
    }

    public void setGenerateDistanceMatrix(final boolean generateDistanceMatrix) {
        this.generateDistanceMatrix = generateDistanceMatrix;
    }

    public abstract void cleanDistanceMatrix();
}
