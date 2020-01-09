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
package tsml.transformers.shapelet_tools.class_value;

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
