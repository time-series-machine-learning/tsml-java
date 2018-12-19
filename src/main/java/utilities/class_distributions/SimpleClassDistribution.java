/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.class_distributions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */


public class SimpleClassDistribution extends ClassDistribution {
    private final Integer[] classDistribution;
    private final Set<Double> keySet;

    public SimpleClassDistribution(Instances data) {
        
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
    public SimpleClassDistribution(ClassDistribution in){
        
        //copy over the data.
        classDistribution = new Integer[in.size()];
        for(int i=0; i<in.size(); i++)
        {
            classDistribution[i] = in.get(i);
        }
        keySet = in.keySet();
    }
   
    
    //creates an empty distribution of specified size.
    public SimpleClassDistribution(int size)
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
