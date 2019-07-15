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
package evaluation.tuning;

import utilities.ArrayUtilities;

import java.util.*;
import java.util.Map.Entry;
import java.util.function.Function;

import static utilities.ArrayUtilities.removeDuplicatesInPlace;
import static utilities.Utilities.fromPermutation;
import static utilities.Utilities.numPermutations;

/**
 *
 * Wraps/contains what is essentially a Map<String, List<String>>, which maps parameter names
 * to lists of possible values (stored as strings). The names should align to the names of different 
 * set-able options via the setOptions(String[]) method of the classifier to be tuned. 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ParameterSpace implements Iterable<Entry<String, List<Object>>>{
    public Map<String, List<Object>> parameterLists = new TreeMap<>();
       
    public int numParas() { 
        return parameterLists.size();
    }
    
    public int numUniqueParameterSets() { 
        int total = 1;
        for (Map.Entry<String, List<Object>> entry : parameterLists.entrySet())
            total *= entry.getValue().size();
        return total;
    }

    public List<Object> getValues(String key)  {
        return parameterLists.get(key);
    }

    public List<Integer> getParameterSizes() {
        List<Integer> sizes = new ArrayList<>();
        for(Map.Entry<String, List<Object>> entry : parameterLists.entrySet()) {
            sizes.add(entry.getValue().size());
        }
        return sizes;
    }

    public int size() {
        return numPermutations(getParameterSizes());
    }

    public ParameterSet get(int index) {
        List<Integer> indices = fromPermutation(index, getParameterSizes());
        ParameterSet parameterSet = new ParameterSet();
        int i = 0;
        for(Map.Entry<String, List<Object>> entry : parameterLists.entrySet()) {
            parameterSet.addParameter(entry.getKey(), String.valueOf(entry.getValue().get(indices.get(i))));
            i++;
        }
        return parameterSet;
    }
    
    /**
     * Adder for *list* of any object (including string)
     * *arrays* of object will use this method by making the call 
     * space.addParater(paraName, Arrays.asList(values));
     */
    public void addParameter(String paraName, Collection<? extends Object> paraValues) {
        parameterLists.put(paraName, new ArrayList<>(paraValues));
    }

    /**
     * Adder for *array* of strings themselves
     */
    public void addParameter(String paraName, String[] paraValues) {
        parameterLists.put(paraName, new ArrayList<>(Arrays.asList(paraValues)));
    }

    //Adders for the primitive types
    public void addParameter(String paraName, int[] paraValues) {
        parameterLists.put(paraName, new ArrayList<>(Arrays.asList(ArrayUtilities.box(paraValues))));
    }
    public void addParameter(String paraName, double[] paraValues) {
        parameterLists.put(paraName, new ArrayList<>(Arrays.asList(ArrayUtilities.box(paraValues))));
    }
    public void addParameter(String paraName, float[] paraValues) {
        parameterLists.put(paraName, new ArrayList<>(Arrays.asList(ArrayUtilities.box(paraValues))));
    }
    public void addParameter(String paraName, boolean[] paraValues) {
        parameterLists.put(paraName, new ArrayList<>(Arrays.asList(ArrayUtilities.box(paraValues))));
    }
    public void addParameter(String paraName, long[] paraValues) {
        parameterLists.put(paraName, new ArrayList<>(Arrays.asList(ArrayUtilities.box(paraValues))));
    }
    public void addParameter(String paraName, short[] paraValues) {
        parameterLists.put(paraName, new ArrayList<>(Arrays.asList(ArrayUtilities.box(paraValues))));
    }

    @Override
    public Iterator<Entry<String, List<Object>>> iterator() {
        return parameterLists.entrySet().iterator();
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("{\n");
        
        for (Map.Entry<String, List<Object>> paras : parameterLists.entrySet()) {
            sb.append("\t" + paras.getKey() + ": [ ");
            for (Object val : paras.getValue())
                sb.append(String.valueOf(val) + ", ");
            sb.append("]\n");
        }
        sb.append("}");
        
        return sb.toString();
    }

    public static void main(String[] args) {
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter("p1", new double[] {1.0,2.0,3.0,4.0,3.0,2.0,1.0});
        parameterSpace.addParameter("p2", new String[] {"d", "e", "f", "g"});
        parameterSpace.addParameter("p3", new String[] {"h", "i", "j", "k"});
        int size = parameterSpace.size();
        for(int i = 0; i < size; i++) {
            ParameterSet parameterSet = parameterSpace.get(i);
            System.out.println(parameterSet);
        }
        parameterSpace.removeDuplicateValues();
        System.out.println("----");
        size = parameterSpace.size();
        for(int i = 0; i < size; i++) {
            ParameterSet parameterSet = parameterSpace.get(i);
            System.out.println(parameterSet);
        }
    }


    public void removeDuplicateValues() {
        for(List<Object> values : parameterLists.values()) {
            removeDuplicatesInPlace(values);
        }
    }

    public void addParameterValue(String key, Object value) {
        List<Object> values = parameterLists.computeIfAbsent(key, s -> new ArrayList<>());
        values.add(value);
    }

    public void addParameterValue(Map.Entry<String, Object> entry) {
        addParameterValue(entry.getKey(), entry.getValue());
    }

    public void addParameter(ParameterSet parameterSet) {
        for(Map.Entry<String, Object> entry : parameterSet.getAllParameters()) {
            addParameterValue(entry);
        }
    }
}
