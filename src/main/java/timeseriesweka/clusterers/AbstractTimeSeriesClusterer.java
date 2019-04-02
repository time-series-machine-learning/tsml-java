/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.clusterers;

import weka.clusterers.AbstractClusterer;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author pfm15hbu
 */
public abstract class AbstractTimeSeriesClusterer extends AbstractClusterer{
    
    protected boolean dontCopyInstances = false;
    
    public void setDontCopyInstances(boolean b){
        dontCopyInstances = b;
    }

    protected void zNormalise(Instances data) {
        for (Instance inst: data){
            zNormalise(inst);
        }
    }

    protected void zNormalise(Instance inst){
        double meanSum = 0;
        int length = inst.numAttributes();

        for (int i = 0; i < length; i++){
            meanSum += inst.value(i);
        }

        double mean = meanSum / length;

        double squareSum = 0;

        for (int i = 0; i < length; i++){
            double temp = inst.value(i) - mean;
            squareSum += temp * temp;
        }

        double stdev = Math.sqrt(squareSum/(length-1));

        if (stdev == 0){
            stdev = 1;
        }

        for (int i = 0; i < length; i++){
            inst.setValue(i, (inst.value(i) - mean) / stdev);
        }
    }
}
