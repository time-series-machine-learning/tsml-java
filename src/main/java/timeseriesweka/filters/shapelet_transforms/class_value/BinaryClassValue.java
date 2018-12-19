/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.class_value;

import utilities.class_distributions.ClassDistribution;
import utilities.class_distributions.TreeSetClassDistribution;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class BinaryClassValue extends NormalClassValue{

    
    ClassDistribution[] binaryClassDistribution;
    
    @Override
    public void init(Instances inst)
    {
        //this inits the classDistributions.
        super.init(inst);
        binaryClassDistribution = createBinaryDistributions();
    }
    
    @Override
    public ClassDistribution getClassDistributions() {
        return binaryClassDistribution[(int)shapeletValue];        
    }

    @Override
    public double getClassValue(Instance in) {
        return in.classValue() == shapeletValue ? 0.0 : 1.0;
    }
    
    
    private ClassDistribution[] createBinaryDistributions()
    {
        ClassDistribution[] binaryMapping = new ClassDistribution[classDistributions.size()];
        
        //for each classVal build a binary distribution map.
        int i=0;
        for(double key : classDistributions.keySet())
        {
            binaryMapping[i++] = binariseDistributions(key);
        }
        return binaryMapping;
    }
    
    private ClassDistribution binariseDistributions(double shapeletClassVal)
    {
        //binary distribution only needs to be size 2.
        ClassDistribution binaryDistribution = new TreeSetClassDistribution();

        Integer shapeletClassCount = classDistributions.get(shapeletClassVal);
        binaryDistribution.put(0.0, shapeletClassCount);
        
        int sum = 0;
        for(double i : classDistributions.keySet()) 
        {
            sum += classDistributions.get(i);
        }
        
        //remove the shapeletsClass count. Rest should be all the other classes.
        sum -= shapeletClassCount; 
        binaryDistribution.put(1.0, sum);
        return binaryDistribution;
    }
}
