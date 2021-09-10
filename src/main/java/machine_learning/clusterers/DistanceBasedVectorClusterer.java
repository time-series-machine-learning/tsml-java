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

package machine_learning.clusterers;

import tsml.clusterers.EnhancedAbstractClusterer;
import weka.core.*;

import static utilities.InstanceTools.deleteClassAttribute;

/**
 * Abstract class for vector based clusterers.
 *
 * @author Matthew Middlehurst
 */
public abstract class DistanceBasedVectorClusterer extends EnhancedAbstractClusterer {

    protected DistanceFunction distFunc = new EuclideanDistance();
    protected boolean symmetricDistance = true;
    protected boolean normaliseData = true;

    //mean and stdev of each attribute for normalisation.
    protected double[] attributeMeans;
    protected double[] attributeStdDevs;

    @Override
    public void buildClusterer(Instances data) throws Exception {
        super.buildClusterer(data);

        if (normaliseData)
            normaliseData(train);

        distFunc.setInstances(train);
    }

    //Find the closest train instance and return its cluster
    @Override
    public int clusterInstance(Instance inst) throws Exception {
        Instance newInst = copyInstances ? new DenseInstance(inst) : inst;
        deleteClassAttribute(newInst);
        if (normaliseData)
            normaliseData(newInst);
        double minDist = Double.MAX_VALUE;
        int closestCluster = 0;

        for (int i = 0; i < train.size(); ++i) {
            double dist = distFunc.distance(newInst, train.get(i));

            if (dist < minDist) {
                minDist = dist;
                closestCluster = (int) assignments[i];
            }
        }

        return closestCluster;
    }

    public void setDistanceFunction(DistanceFunction distFunc) {
        this.distFunc = distFunc;
    }

    public void setSymmetricDistance(boolean b) { this.symmetricDistance = b; }

    public void setNormaliseData(boolean b) {
        this.normaliseData = b;
    }

    //Normalise instances and save the means and standard deviations.
    protected void normaliseData(Instances data) throws Exception {
        if (data.classIndex() >= 0 && data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute is available and not the final attribute.");
        }

        int cls = data.classIndex() >= 0 ? 1 : 0;
        attributeMeans = new double[data.numAttributes() - cls];
        attributeStdDevs = new double[data.numAttributes() - cls];

        for (int i = 0; i < data.numAttributes() - cls; i++) {
            attributeMeans[i] = data.attributeStats(i).numericStats.mean;
            attributeStdDevs[i] = data.attributeStats(i).numericStats
                    .stdDev;

            if (attributeStdDevs[i] == 0) {
                attributeStdDevs[i] = 0.0000001;
            }

            for (int n = 0; n < data.size(); n++) {
                Instance instance = data.get(n);
                instance.setValue(i, (instance.value(i) - attributeMeans[i]) / attributeStdDevs[i]);
            }
        }
    }

    protected void normaliseData(Instance inst){
        int cls = inst.classIndex() >= 0 ? 1 : 0;

        for (int i = 0; i < inst.numAttributes() - cls; i++) {
            inst.setValue(i, (inst.value(i) - attributeMeans[i]) / attributeStdDevs[i]);
        }
    }
}
