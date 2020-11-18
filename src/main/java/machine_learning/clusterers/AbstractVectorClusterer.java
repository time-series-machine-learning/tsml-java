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
package machine_learning.clusterers;

import weka.clusterers.AbstractClusterer;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Abstract class for vector based clusterers.
 *
 * @author Matthew Middlehurst
 */
public abstract class AbstractVectorClusterer extends AbstractClusterer{

    protected DistanceFunction distFunc = new EuclideanDistance();
    protected boolean normaliseData = true;
    protected boolean copyInstances = true;

    protected int[] assignments;
    protected ArrayList<Integer>[] clusters;

    //mean and stdev of each attribute for normalisation.
    protected double[] attributeMeans;
    protected double[] attributeStdDevs;

    public int[] getAssignments(){
        return assignments;
    }

    public ArrayList<Integer>[] getClusters(){
        return clusters;
    }

    public void setDistanceFunction(DistanceFunction distFunc){
        this.distFunc = distFunc;
    }

    public void setNormaliseData(boolean b){
        this.normaliseData = b;
    }

    public void setCopyInstances(boolean b){
        copyInstances = b;
    }

    //Normalise instances and save the means and standard deviations.
    protected void normaliseData(Instances data) throws Exception{
        if (data.classIndex() >= 0 && data.classIndex() != data.numAttributes()-1){
            throw new Exception("Class attribute is available and not the final attribute.");
        }

        attributeMeans = new double[data.numAttributes()-1];
        attributeStdDevs = new double[data.numAttributes()-1];

        for (int i = 0; i < data.numAttributes()-1; i++){
            attributeMeans[i] = data.attributeStats(i).numericStats.mean;
            attributeStdDevs[i] = data.attributeStats(i).numericStats
                    .stdDev;

            for (int n = 0; n < data.size(); n++){
                Instance instance = data.get(n);
                instance.setValue(i, (instance.value(i) - attributeMeans[i])
                        /attributeStdDevs[i]);
            }
        }
    }
}
