
package evaluation.tuning;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import utilities.generic_storage.Pair;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class ParameterSet {
    public Map<String, String> parameterSet = new HashMap<>();
        
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
}
