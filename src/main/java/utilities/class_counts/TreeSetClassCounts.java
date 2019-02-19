/*
  This is used by Aarons shapelet code and is down for depreciation

 */
package utilities.class_counts;

import java.util.Collection;
import java.util.ListIterator;
import java.util.Set;
import java.util.TreeMap;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class TreeSetClassCounts extends ClassCounts{

    TreeMap<Double,Integer> classDistribution;
    
    public TreeSetClassCounts(Instances data) {
        
        classDistribution = new TreeMap<>();
        
        ListIterator<Instance> it = data.listIterator();
        double classValue;
        while (it.hasNext())
        {
            classValue = it.next().classValue();

            Integer val = classDistribution.get(classValue);

            val = (val != null) ? val + 1 : 1;
            classDistribution.put(classValue, val);
        }
    }
    
    public TreeSetClassCounts() {
        classDistribution = new TreeMap<>();
    }
    
    public TreeSetClassCounts(ClassCounts in){
        
        //copy over the data.
        classDistribution = new TreeMap<>();
        for(double val : in.keySet())
        {
            classDistribution.put(val, in.get(val));
        }
    }


    @Override
    public int get(double classValue) {
        return classDistribution.getOrDefault(classValue, 0);
    }

    @Override
    public void put(double classValue, int value) {
       classDistribution.put(classValue, value);
    }

    @Override
    public int size() {
        return classDistribution.size();
    }

    @Override
    public int get(int accessValue) {
        return classDistribution.getOrDefault((double) accessValue, 0);
    }
    
    @Override
    public void addTo(double classVal, int value)
    {
        put(classVal, get(classVal)+value);
    }
    
    @Override
    public Set<Double> keySet()
    {
        return classDistribution.keySet();
    }
    
    @Override
    public Collection<Integer> values()
    {
        return classDistribution.values();
    }
    
}
