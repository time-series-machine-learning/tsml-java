/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.class_value;

import utilities.class_counts.ClassCounts;
import utilities.class_counts.TreeSetClassCounts;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class BinaryClassValue extends NormalClassValue{

    
    ClassCounts[] binaryClassDistribution;
    
    @Override
    public void init(Instances inst)
    {
        //this inits the classDistributions.
        super.init(inst);
        binaryClassDistribution = createBinaryDistributions();
    }
    
    @Override
    public ClassCounts getClassDistributions() {
        return binaryClassDistribution[(int)shapeletValue];        
    }

    @Override
    public double getClassValue(Instance in) {
        return in.classValue() == shapeletValue ? 0.0 : 1.0;
    }
    
    
    private ClassCounts[] createBinaryDistributions()
    {
        ClassCounts[] binaryMapping = new ClassCounts[classDistributions.size()];
        
        //for each classVal build a binary distribution map.
        int i=0;
        for(double key : classDistributions.keySet())
        {
            binaryMapping[i++] = binariseDistributions(key);
        }
        return binaryMapping;
    }
    
    private ClassCounts binariseDistributions(double shapeletClassVal)
    {
        //binary distribution only needs to be size 2.
        ClassCounts binaryDistribution = new TreeSetClassCounts();

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
