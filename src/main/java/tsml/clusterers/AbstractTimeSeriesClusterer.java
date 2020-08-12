/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.clusterers;

import weka.clusterers.AbstractClusterer;

import java.util.ArrayList;

/**
 *
 * @author pfm15hbu
 */
public abstract class AbstractTimeSeriesClusterer extends AbstractClusterer{

    protected boolean copyInstances = true;

    protected int[] assignments;
    protected ArrayList<Integer>[] clusters;

    public int[] getAssignments(){
        return assignments;
    }

    public ArrayList<Integer>[] getClusters(){
        return clusters;
    }

    public void setCopyInstances(boolean b){
        copyInstances = b;
    }
}
