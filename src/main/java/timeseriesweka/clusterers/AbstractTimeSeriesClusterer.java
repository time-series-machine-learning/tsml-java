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
    
    public void setChangeOriginalInstances(boolean b){
        changeOriginalInstances = b;
    }
    
    public void setHasClassValue(boolean b){
        hasClassValue = b;
    }
    
    protected void zNormalise(Instances data) throws Exception{  
        if (data.classIndex() >= 0 && data.classIndex() != data.numAttributes()-1){
            throw new Exception("Class attribute is available and not the final attribute.");
        }
        
        int length;
        
        if (data.classIndex() >= 0){
            length = data.numAttributes()-1;
            hasClassValue = true;
        }
        else{
            length = data.numAttributes();
            hasClassValue = false;
        }
        
        for (Instance inst: data){
            double meanSum = 0;

            for (int i = 0; i < inst.numAttributes(); i++){
                meanSum += inst.value(i);
            }

            double mean = meanSum / length;

            double squareSum = 0;

            for (int i = 0; i < length; i++){
                double temp = inst.value(i) - mean;
                squareSum += temp * temp;
            }

            double stdev = Math.sqrt(squareSum/length);

            if (stdev == 0){
                stdev = 1;
            }
            
            for (int i = 0; i < length; i++){
                inst.setValue(i, (inst.value(i) - mean) / stdev);
            }
        }
    }
}
