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

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class ParameterSet {
    public Map<String, String> parameterSet = new HashMap<>();
        
    private static String startParaLineDelimiter = "parasStart";
    private static String endParaLineDelimiter = "parasEnd";
    
    public int size() {
        return parameterSet.size();
    }
    
    public boolean contains(String paraName) {
        return parameterSet.containsKey(paraName); 
    }
    
    public Set<Entry<String,String>> getAllParameters() {
        return parameterSet.entrySet();
    }
    
    public String getParameterValue(String paraName) {
        return parameterSet.get(paraName);
    }
    
    public void addParameter(String paraName, String paraValue) { 
        parameterSet.put(paraName, paraValue);
    }
    
    @Override
    public String toString() { 
        StringBuilder sb = new StringBuilder("{");
        
        for (Map.Entry<String, String> para : parameterSet.entrySet())
            sb.append(para.getKey() + ": " + para.getValue() + ", ");
        sb.append("}");
        
        return sb.toString();
    }
    
    public String[] toOptionsList() { 
        String[] ps = new String[parameterSet.size() * 2];

        int i = 0;
        for (Map.Entry<String, String> entry : parameterSet.entrySet()) {
            ps[i] = "-" + entry.getKey();
            ps[i+1] = entry.getValue();
            i+=2;
        }

        return ps;
    }
    
    /**
     * Assumes that this is just a list of flag-value pairs, e.g 
     * [ flag1, value1, flag2, value2....] 
     * and does not contain any independent flags (maybe representing that a boolean 
     * flag should be set to true, for example), and that all the flag/value pairs 
     * are parameter to be read in (e.g no debug flags)
     */
    public void readOptionsList(String[] options) { 
        parameterSet = new HashMap<String, String>();
        
        for (int i = 0; i < options.length; i+=2)
            parameterSet.put(options[i], options[i+1]);
        
        //todo error checking etc
    }
    
    public static String toFileNameString(int[] inds) {
        StringBuilder sb = new StringBuilder();
        sb.append(inds[0]);

        for (int i = 1; i < inds.length; i++)
            sb.append("_").append(inds[i]);

        return sb.toString();
    }
    
    public String toClassifierResultsParaLine() {
        return toClassifierResultsParaLine(false); 
    }
    
    public String toClassifierResultsParaLine(boolean includeStartEndMarkers) {
        StringBuilder sb = new StringBuilder();
        
        if (includeStartEndMarkers)
            sb.append(startParaLineDelimiter).append(",");
        
        boolean first = true;
        for (Map.Entry<String, String> entry : parameterSet.entrySet()) {
            if (first) {
                //no initial comma
                sb.append(entry.getKey()).append(",").append(entry.getValue());
                first = false;
            }
            else 
                sb.append(",").append(entry.getKey()).append(",").append(entry.getValue());
            
        }
        
        if (includeStartEndMarkers)
            sb.append(",").append(endParaLineDelimiter);
        
        return sb.toString();
    }
    
    /**
     * Assumes that this is just a list of flag-value pairs, e.g 
     * flag1, value1, flag2, value2.... 
     */
    public void readClassifierResultsParaLine(String line) { 
        parameterSet = new HashMap<>();
        String[] parts = line.split(",");
       
        int length = parts.length;
        if (parts.length % 2 == 1) {
            //odd number of parts, assumed to be an empty string as the last entry,
            //trailing comma. todo revisit this if there are problems
            length -= 1;
        }
        
        //note: i+=2
        for (int i = 0; i < parts.length; i+=2) {
            String key = parts[i];
            String value = parts[i+1];
            
            parameterSet.put(key, value);
        }
    }
    
    /**
     * Will strip the line to only the parts between the parameterset start and end 
     * delimeters and interpret those, useful if the classifierresults parameter line 
     * has other info like buildtimes,accs,etc. 
     */
    public void readClassifierResultsParaLine(String line, boolean startEndMarkersIncluded) { 
        
        if (startEndMarkersIncluded) { 
            String[] parts = line.split(",");
            
            StringBuilder sb = new StringBuilder();
            
            boolean reading = false;
            for (String part : parts) {
                if (part.equals(startParaLineDelimiter)) {
                    reading = true;
                    continue;
                }
                
                if (part.equals(endParaLineDelimiter))
                    break;
                
                if (reading)
                    sb.append(part + ",");
                
                //else 
                //  ignore. 
            }
            
            line = sb.toString();
        }
        
        readClassifierResultsParaLine(line);
    }
}
