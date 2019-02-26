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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This is used by Aarons shapelet code and may be depreciated with new light 
 *   weight shapelets
 * @author raj09hxu
 */


public class SimpleClassCounts extends ClassCounts {
    private final Integer[] classDistribution;
    private final Set<Double> keySet;

    public SimpleClassCounts(Instances data) {
        
        classDistribution = new Integer[data.numClasses()];
        Arrays.fill(classDistribution, 0);
        
        keySet = new TreeSet<>();
        
        for (Instance data1 : data) {
            int thisClassVal = (int) data1.classValue();
            keySet.add(data1.classValue());
            
            classDistribution[thisClassVal]++;
        }
    }
    
    //clones the object 
    public SimpleClassCounts(ClassCounts in){
        
        //copy over the data.
        classDistribution = new Integer[in.size()];
        for(int i=0; i<in.size(); i++)
        {
            classDistribution[i] = in.get(i);
        }
        keySet = in.keySet();
    }
   
    
    //creates an empty distribution of specified size.
    public SimpleClassCounts(int size)
    {
        classDistribution = new Integer[size];
        Arrays.fill(classDistribution, 0);
        keySet = new TreeSet<>();
    }

    @Override
    public int get(double classValue) {
        return classDistribution[(int)classValue];
    }

    //Use this with caution.
    @Override
    public void put(double classValue, int value) {
        classDistribution[(int)classValue] = value;
        keySet.add(classValue);
    }

    @Override
    public int size() {
        return classDistribution.length;
    }   

    @Override
    public int get(int accessValue) {
        return classDistribution[accessValue];
    }
    
    @Override
    public String toString(){
        String temp = "";
        for(int i=0; i<classDistribution.length; i++){
            temp+="["+i+" "+classDistribution[i]+"] ";
        }
        return temp;
    }
    
    @Override
    public void addTo(double classValue, int value)
    {
        classDistribution[(int)classValue]+= value;
    }

    @Override
    public Set<Double> keySet() {
        return keySet;
    }
    
    @Override
    public Collection<Integer> values()
    {
        return new ArrayList<>(Arrays.asList(classDistribution));
    }
}
