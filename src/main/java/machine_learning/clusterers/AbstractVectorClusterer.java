/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
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
 * @author pfm15hbu
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
