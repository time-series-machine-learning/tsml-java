/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.class_value;

import java.io.Serializable;
import utilities.class_counts.ClassCounts;
import utilities.class_counts.TreeSetClassCounts;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class NormalClassValue implements Serializable{
    
    double shapeletValue;
    ClassCounts classDistributions;
    
    public void init(Instances inst)
    {
        classDistributions = new TreeSetClassCounts(inst);
    }
    
    public ClassCounts getClassDistributions()
    {
        return classDistributions;
    }
    
    //this will get updated as and when we work with a new shapelet.
    public void setShapeletValue(Instance shapeletSeries)
    {
        shapeletValue = shapeletSeries.classValue();
    }
    
    public double getClassValue(Instance in){
        return in.classValue();
    }

    public double getShapeletValue() {
        return shapeletValue;
    }
    
}
