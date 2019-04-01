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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 *
 * Wraps/contains what is essentially a Map<String, List<String>>, which maps parameter names
 * to lists of possible values (stored as strings). The names should align to the names of different 
 * set-able options via the setOptions(String[]) method of the classifier to be tuned. 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ParameterSpace implements Iterable<Entry<String, List<String>>>{
    public Map<String, List<String>> parameterLists = new HashMap<>();
       
    public int numParas() { 
        return parameterLists.size();
    }
    
    public int numUniqueParameterSets() { 
        int total = 1;
        for (Map.Entry<String, List<String>> entry : parameterLists.entrySet())
            total *= entry.getValue().size();
        return total;
    }

    public List<String> getValues(String key)  {
        return parameterLists.get(key);
    }
    
    
    
    
    
    /**
     * Adder for *list* of any object (including string)
     * *arrays* of object will use this method by making the call 
     * space.addParater(paraName, Arrays.asList(values));
     */
    public void addParameter(String paraName, List<? extends Object> paraValues) {
        List<String> stringValues = new ArrayList<>(paraValues.size());
        for (int i = 0; i < paraValues.size(); i++)
            stringValues.add(paraValues.get(i).toString());
        parameterLists.put(paraName, stringValues);
    }

    /**
     * Adder for *array* of strings themselves
     */
    public void addParameter(String paraName, String[] paraValues) {
        parameterLists.put(paraName, Arrays.asList(paraValues));
    }

    //Adders for the primitive types
    public void addParameter(String paraName, int[] paraValues) {
        List<String> stringValues = new ArrayList<>(paraValues.length);
        for (int i = 0; i < paraValues.length; i++)
            stringValues.add(paraValues[i]+"");
        parameterLists.put(paraName, stringValues);
    }
    public void addParameter(String paraName, double[] paraValues) {
        List<String> stringValues = new ArrayList<>(paraValues.length);
        for (int i = 0; i < paraValues.length; i++)
            stringValues.add(paraValues[i]+"");
        parameterLists.put(paraName, stringValues);
    }
    public void addParameter(String paraName, float[] paraValues) {
        List<String> stringValues = new ArrayList<>(paraValues.length);
        for (int i = 0; i < paraValues.length; i++)
            stringValues.add(paraValues[i]+"");
        parameterLists.put(paraName, stringValues);
    }
    public void addParameter(String paraName, boolean[] paraValues) {
        List<String> stringValues = new ArrayList<>(paraValues.length);
        for (int i = 0; i < paraValues.length; i++)
            stringValues.add(paraValues[i]+"");
        parameterLists.put(paraName, stringValues);
    }
    public void addParameter(String paraName, long[] paraValues) {
        List<String> stringValues = new ArrayList<>(paraValues.length);
        for (int i = 0; i < paraValues.length; i++)
            stringValues.add(paraValues[i]+"");
        parameterLists.put(paraName, stringValues);
    }
    public void addParameter(String paraName, short[] paraValues) {
        List<String> stringValues = new ArrayList<>(paraValues.length);
        for (int i = 0; i < paraValues.length; i++)
            stringValues.add(paraValues[i]+"");
        parameterLists.put(paraName, stringValues);
    }

    @Override
    public Iterator<Entry<String, List<String>>> iterator() {
        return parameterLists.entrySet().iterator();
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("{\n");
        
        for (Map.Entry<String, List<String>> paras : parameterLists.entrySet()) {
            sb.append("\t" + paras.getKey() + ": [ ");
            for (String val : paras.getValue())
                sb.append(val + ", ");
            sb.append("]\n");
        }
        sb.append("}");
        
        return sb.toString();
    }
}
