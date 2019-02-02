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
    
    protected boolean changeOriginalInstances = true;
    protected boolean hasClassValue = false;

    private boolean checkedClass = false;
    
    public void setChangeOriginalInstances(boolean b){
        changeOriginalInstances = b;
    }
    
    public void setHasClassValue(boolean b){
        hasClassValue = b;
        checkedClass = true;
    }

    protected void zNormalise(Instances data, boolean hasClass) throws Exception {
        int length;
        
        if (hasClass){
            length = data.numAttributes()-1;
        }
        else{
            length = data.numAttributes();
        }
        
        for (Instance inst: data){
            zNormalise(inst, length);
        }
    }

    protected void zNormalise(Instance inst, boolean hasClass){
        int length;

        if (hasClass){
            length = inst.numAttributes()-1;
        }
        else{
            length = inst.numAttributes();
        }

        zNormalise(inst, length);
    }

    private void zNormalise(Instance inst, int length){
        double meanSum = 0;

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

    protected void checkClass(Instances data) throws Exception {
        if (data.classIndex() >= 0 && data.classIndex() != data.numAttributes()-1){
            throw new Exception("Class attribute is available and not the final attribute.");
        }

        if (data.classIndex() >= 0){
            hasClassValue = true;
        }
        else{
            hasClassValue = false;
        }
    }
}
