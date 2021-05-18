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

package tsml.clusterers;

import weka.clusterers.AbstractClusterer;

import java.util.ArrayList;

/**
 * @author pfm15hbu
 */
public abstract class EnhancedAbstractClusterer extends AbstractClusterer {

    protected int seed = 0;
    protected boolean seedClusterer = false;
    protected boolean copyInstances = true;

    protected double[] assignments;
    protected ArrayList<Integer>[] clusters;

    public double[] getAssignments() {
        return assignments;
    }

    public ArrayList<Integer>[] getClusters() {
        return clusters;
    }

    public void setSeed(int seed) {
        this.seed = seed;
        seedClusterer = true;
    }

    public void setCopyInstances(boolean b) {
        copyInstances = b;
    }
}
