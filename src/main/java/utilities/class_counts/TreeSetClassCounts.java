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
package utilities.class_counts;

import java.util.Collection;
import java.util.ListIterator;
import java.util.Set;
import java.util.TreeMap;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This is used by Aarons shapelet code and is down for depreciation
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

    public TreeSetClassCounts(TimeSeriesInstances data) {
        classDistribution = new TreeMap<>();
        for(TimeSeriesInstance inst : data){
            classDistribution.merge((double)inst.getLabelIndex(), 1, Integer::sum);
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
